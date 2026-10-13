"""Trajectory-level gap analysis against real pedestrian datasets (ETH/EWAP, ATC).

Compares three per-step scalars between the recorded pedestrians of a sweep,
split by driver, and real tracking data: walking speed, clearance to the
nearest other pedestrian, and turn rate. Every side is put on the EWAP
annotation grid of 0.4 s before the scalars are taken, so speed and clearance
are measured at the same time resolution. Turn rate uses 0.8 s heading chords
on that grid.

Datasets: EWAP (seq_eth and seq_hotel, obsmat.txt) and optionally one day of
the ATC shopping-centre tracking data (Brscic et al. 2013, atc-YYYYMMDD.csv),
of which one clock hour is taken.

    evaluate eth --recordings_dir <sweep> --ewap <dir with seq_eth/ seq_hotel/> [--atc <atc-YYYYMMDD.csv>] --out_dir <dir>

Writes eth_gap.csv (one row per driver and per real-data source with the
median, mean and IQR of each scalar), eth_gap_samples.parquet (the per-step
samples the rows summarize), and eth_gap.pdf.
"""

from __future__ import annotations

import math
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from arena_humansim.utils.evaluation.analyze import iter_trials
from arena_humansim.utils.evaluation.plots import DRIVER_COLOR, DRIVER_ORDER

EWAP_URL = "https://icu.ee.ethz.ch/content/dam/ethz/special-interest/itet/computer-vision-dam/documents/datasets/ewap_dataset_light.tgz"
EWAP_STEP_S = 0.4
EWAP_FRAME_STRIDE = {"seq_eth": 6, "seq_hotel": 10}  # video frames per annotation step
MIN_MOVING_SPEED = 0.1  # m/s
TURN_BASELINE_STEPS = 2  # heading chord length in grid steps
ATC_UTC_OFFSET_H = 9
ATC_HOUR = 12  # local clock hour
ATC_MAX_PEOPLE = 4000  # in order of appearance
SCALARS = ("speed", "clearance", "turn_rate")


def _ewap_dir(path: Path | None) -> Path:
    """Directory containing seq_eth/ and seq_hotel/, fetched into ~/.cache when absent."""
    if path is not None and (path / "seq_eth" / "obsmat.txt").exists():
        return path
    cache = Path.home() / ".cache" / "arena_humansim" / "ewap"
    if not (cache / "ewap_dataset" / "seq_eth" / "obsmat.txt").exists():
        cache.mkdir(parents=True, exist_ok=True)
        tgz = cache / "ewap_dataset_light.tgz"
        if not tgz.exists():
            urllib.request.urlretrieve(EWAP_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(cache, filter="data")
    return cache / "ewap_dataset"


def load_ewap(root: Path) -> pd.DataFrame:
    """[frame, ped, x, y] for both sequences, frame converted to seconds on the 0.4 s grid."""
    frames = []
    for seq in ("seq_eth", "seq_hotel"):
        mat = np.loadtxt(root / seq / "obsmat.txt")
        df = pd.DataFrame({"frame": mat[:, 0].astype(int), "agent_id": mat[:, 1].astype(int), "x": mat[:, 2], "y": mat[:, 4]})
        df["sequence"] = seq
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = df["frame"] * (EWAP_STEP_S / df["sequence"].map(EWAP_FRAME_STRIDE))
    return df


def load_atc(csv: Path, hour: int = ATC_HOUR, max_people: int = ATC_MAX_PEOPLE) -> pd.DataFrame:
    """[time, agent_id, x, y, sequence] for one local clock hour of an ATC day file."""
    df = pd.read_csv(csv, header=None, usecols=[0, 1, 2, 3], names=["time", "agent_id", "x", "y"], dtype={"time": float, "agent_id": int, "x": float, "y": float})
    local_hour = ((df["time"] / 3600.0 + ATC_UTC_OFFSET_H) % 24).astype(int)
    df = df[local_hour == hour]
    if df.empty:
        raise ValueError(f"{csv}: no rows in local hour {hour}")
    first_seen = df.groupby("agent_id")["time"].min().sort_values()
    keep = first_seen.index[:max_people]
    df = df[df["agent_id"].isin(keep)].copy()
    df["time"] = df["time"] - df["time"].min()
    df["x"] = df["x"] / 1000.0
    df["y"] = df["y"] / 1000.0
    df["sequence"] = f"{csv.stem}_h{hour:02d}"
    return df[["time", "agent_id", "x", "y", "sequence"]]


def _per_step_scalars(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Speed, nearest-neighbour clearance and turn rate per (agent, step) on a 0.4 s grid."""
    out = []
    for gkey, g in df.groupby(group_col, sort=False):
        g = g.sort_values(["agent_id", "time"])
        step = np.round(g["time"].to_numpy() / EWAP_STEP_S).astype(int)
        g = g.assign(step=step).drop_duplicates(["agent_id", "step"])
        clear = np.full(len(g), np.nan)
        g = g.reset_index(drop=True)
        for _, fr in g.groupby("step"):
            if len(fr) < 2:
                continue
            xy = fr[["x", "y"]].to_numpy()
            d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
            np.fill_diagonal(d, np.inf)
            clear[fr.index.to_numpy()] = d.min(axis=1)
        g["clearance"] = clear
        for aid, tr in g.groupby("agent_id"):
            if len(tr) < 3:
                continue
            xy = tr[["x", "y"]].to_numpy()
            st = tr["step"].to_numpy()
            dxy = np.diff(xy, axis=0)
            dt = np.diff(st) * EWAP_STEP_S
            ok = dt > 0
            speed = np.hypot(dxy[:, 0], dxy[:, 1]) / np.where(ok, dt, np.nan)
            turn = _turn_rate(xy, st)
            out.append(
                pd.DataFrame(
                    {
                        "group": gkey,
                        "agent_id": aid,
                        "speed": speed[1:],
                        "clearance": tr["clearance"].to_numpy()[1:-1],
                        "turn_rate": turn,
                    }
                )
            )
    if not out:
        return pd.DataFrame(columns=["group", "agent_id", *SCALARS])
    return pd.concat(out, ignore_index=True)


def _turn_rate(xy: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Turn rate at interior steps 1..n-2, NaN where a chord is missing, spans a gap or is slower than MIN_MOVING_SPEED."""
    n, b = len(xy), TURN_BASELINE_STEPS
    turn = np.full(max(n - 2, 0), np.nan)
    span = b * EWAP_STEP_S
    for k in range(b, n - b):
        if st[k] - st[k - b] != b or st[k + b] - st[k] != b:
            continue
        back, fwd = xy[k] - xy[k - b], xy[k + b] - xy[k]
        if np.hypot(*back) / span < MIN_MOVING_SPEED or np.hypot(*fwd) / span < MIN_MOVING_SPEED:
            continue
        dh = np.arctan2(fwd[1], fwd[0]) - np.arctan2(back[1], back[0])
        turn[k - 1] = abs(np.arctan2(np.sin(dh), np.cos(dh))) / span
    return turn


def sim_samples(recordings_dirs: list[Path]) -> pd.DataFrame:
    """Per-step scalars for every human trajectory in the sweep, labelled by driver."""
    parts = []
    for df, meta in iter_trials(recordings_dirs):
        if meta["robot_policy"]:
            continue
        humans = df[df["agent_id"] >= 0][["time", "agent_id", "x", "y"]].copy()
        if humans.empty:
            continue
        humans["trial"] = meta["trial_name"]
        s = _per_step_scalars(humans, "trial")
        s["source"] = meta["planner"]
        s["bucket"] = meta["bucket"]
        parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["source", *SCALARS])


def ewap_samples(root: Path) -> pd.DataFrame:
    df = load_ewap(root)
    s = _per_step_scalars(df, "sequence")
    s["source"] = "ETH-" + s["group"].astype(str).str.replace("seq_", "")
    s["bucket"] = "eth"
    return s


def atc_samples(csv: Path) -> pd.DataFrame:
    df = load_atc(csv)
    s = _per_step_scalars(df, "sequence")
    s["source"] = "ATC"
    s["bucket"] = "atc"
    return s


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in samples.groupby("source", sort=False):
        row: dict[str, object] = {"source": src, "n_steps": int(len(g))}
        for k in SCALARS:
            v = g[k].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            row[f"{k}_median"] = float(np.median(v)) if v.size else math.nan
            row[f"{k}_mean"] = float(v.mean()) if v.size else math.nan
            row[f"{k}_q25"] = float(np.percentile(v, 25)) if v.size else math.nan
            row[f"{k}_q75"] = float(np.percentile(v, 75)) if v.size else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


SIZE_GAP = (6.753, 2.657)  # inches


def _iqr(v: np.ndarray) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))


def plot(samples: pd.DataFrame, out_pdf: Path, order: list[str]) -> None:
    """Per-driver median with IQR whiskers against the IQR band and median of each real dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = {"speed": "Speed (m/s)", "clearance": "Clearance (m)", "turn_rate": "Turn rate (rad/s)"}
    drivers = [d for d in DRIVER_ORDER if d in order]
    eth = samples[samples["source"].astype(str).str.startswith("ETH")]
    atc = samples[samples["source"] == "ATC"]
    fig, axes = plt.subplots(1, 3, figsize=SIZE_GAP)
    for ax, k in zip(axes, SCALARS, strict=True):
        lo, med, hi = _iqr(eth[k].to_numpy(dtype=float))
        ax.axhspan(lo, hi, color="#d9d9d9", alpha=0.7, lw=0)
        ax.axhline(med, color="k", ls="--", lw=0.8)
        if not atc.empty:
            lo, med, hi = _iqr(atc[k].to_numpy(dtype=float))
            ax.axhspan(lo, hi, facecolor="none", edgecolor="#8c564b", hatch="///", lw=0, alpha=0.6)
            ax.axhline(med, color="#8c564b", ls=":", lw=0.9)
        for i, d in enumerate(drivers):
            lo, med, hi = _iqr(samples.loc[samples["source"] == d, k].to_numpy(dtype=float))
            ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="o", ms=4, color=DRIVER_COLOR[d], capsize=2, lw=1.0)
        ax.set_xticks(range(len(drivers)))
        ax.set_xticklabels(drivers, rotation=45, ha="right")
        ax.set_ylabel(labels[k])
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Patch(facecolor="#d9d9d9", alpha=0.7, label="ETH BIWI IQR"), Line2D([], [], color="k", ls="--", lw=0.8, label="ETH BIWI median")]
    if not atc.empty:
        handles += [Patch(facecolor="none", edgecolor="#8c564b", hatch="///", label="ATC IQR"), Line2D([], [], color="#8c564b", ls=":", lw=0.9, label="ATC median")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_pdf)
    plt.close(fig)


def run_eth_gap(recordings_dirs: list[Path], out_dir: Path, ewap: Path | None = None, atc: Path | None = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = _ewap_dir(ewap)
    print(f"EWAP at {root}")
    real = [ewap_samples(root)]
    print(f"  ETH: {len(real[0])} steps from {real[0]['agent_id'].nunique()} pedestrians")
    if atc is not None:
        real.append(atc_samples(atc))
        print(f"  ATC: {len(real[-1])} steps from {real[-1]['agent_id'].nunique()} pedestrians ({atc.name}, local hour {ATC_HOUR})")
    sim = sim_samples(recordings_dirs)
    print(f"  sim: {len(sim)} steps, drivers {sorted(sim['source'].unique())}")
    samples = pd.concat([sim, *real], ignore_index=True)
    samples.to_parquet(out_dir / "eth_gap_samples.parquet", index=False)
    table = summarize(samples)
    table.to_csv(out_dir / "eth_gap.csv", index=False)
    order = sorted(sim["source"].unique()) + [src for r in real for src in r["source"].unique()]
    plot(samples, out_dir / "eth_gap.pdf", order)
    print(table.to_string(index=False, float_format="{:.3f}".format))
    print(f"  wrote {out_dir / 'eth_gap.csv'}, {out_dir / 'eth_gap.pdf'}")
    return table
