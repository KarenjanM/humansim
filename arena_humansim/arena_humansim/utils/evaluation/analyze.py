import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from arena_humansim.utils.evaluation.buckets import (
    DRIVER_CLASS_FINE,
    KNOWN_BUCKETS,
)
from arena_humansim.utils.evaluation.framing import pick_framing, render_framing
from arena_humansim.utils.evaluation.metrics import (
    calculate_kinematic_metrics,
    calculate_run_collisions,
    pairwise_hausdorff,
)
from arena_humansim.utils.evaluation.variance import (
    decompose_scalar_variance,
    headline_seed_ratio,
    self_divergence_table,
)


def parse_trial_dir(name: str) -> tuple[str, str, str, int]:
    parts = name.split("__")
    if len(parts) == 4:
        scenario, planner, robot_policy, seed = parts
    elif len(parts) == 3:
        scenario, planner, seed = parts
        robot_policy = ""
    else:
        raise ValueError(f"trial dir {name!r} not in <scenario>__<planner>[__<robot_policy>]__<seed> form")
    return scenario, planner, robot_policy, int(seed)


from arena_humansim.utils.scenario_discovery import parse_scenario_id  # noqa: E402, F401


def infer_bucket(scenario_yaml: dict[str, Any], scenario_id: str = "") -> str:
    """Classify a (resolved) scenario YAML into nav/bt/het.

    'het' if the scenario id is under het/;
    'bt' if any agent_type uses mode==behavior_tree (interaction-layered humans);
    'nav' otherwise (pure waypoint navigation).
    """
    if scenario_id.startswith("het_") or scenario_id.startswith("het/"):
        return "het"
    agent_types = scenario_yaml.get("agent_types") or {}
    for atype in agent_types.values():
        if isinstance(atype, dict) and atype.get("mode") == "behavior_tree":
            return "bt"
    return "nav"


def iter_trials(recordings_dirs: Path | list[Path]) -> Iterator[tuple[pd.DataFrame, dict[str, Any]]]:
    """Yield (annotated_df, meta) per unique trial; one trial's df is resident at a time.
    meta keys: scenario, planner, robot_policy, seed, bucket, density, modality,
    is_robot_scenario, source_dir, recordings_dir, trial_name."""
    from arena_humansim.utils.evaluation.bag_cache import load_trial

    if isinstance(recordings_dirs, Path):
        recordings_dirs = [recordings_dirs]
    seen: set[tuple[str, str, str, int]] = set()
    for recordings_dir in recordings_dirs:
        for trial_dir in sorted(recordings_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            try:
                scenario, planner, robot_policy, seed = parse_trial_dir(trial_dir.name)
            except ValueError:
                continue
            key = (scenario, planner, robot_policy, seed)
            if key in seen:
                print(f"skipping {recordings_dir.name}/{trial_dir.name} (duplicate of trial already loaded from earlier sweep)")
                continue
            df = load_trial(trial_dir)
            if df.empty:
                print(f"skipping {trial_dir.name} (empty bag)")
                continue
            seen.add(key)
            snapshot = trial_dir / "scenario.yaml"
            if snapshot.exists():
                try:
                    bucket = infer_bucket(yaml.safe_load(snapshot.read_text()) or {}, scenario)
                except Exception as exc:
                    print(f"warning: {trial_dir.name}: failed to parse scenario.yaml ({exc}); bucket=unknown")
                    bucket = "unknown"
            else:
                print(f"warning: {trial_dir.name}: no scenario.yaml snapshot; bucket=unknown (run backfill_scenario_snapshots)")
                bucket = "unknown"
            _, density, modality, is_robot_scenario = parse_scenario_id(scenario)
            df = df.copy()
            df["scenario"] = scenario
            df["seed"] = seed
            df["planner"] = planner
            df["robot_policy"] = robot_policy
            df["bucket"] = bucket
            df["density"] = density
            df["modality"] = modality
            df["is_robot_scenario"] = is_robot_scenario
            df["source_dir"] = recordings_dir.name
            yield (
                df,
                {
                    "scenario": scenario,
                    "planner": planner,
                    "robot_policy": robot_policy,
                    "seed": seed,
                    "bucket": bucket,
                    "density": density,
                    "modality": modality,
                    "is_robot_scenario": is_robot_scenario,
                    "source_dir": recordings_dir.name,
                    "recordings_dir": recordings_dir,
                    "trial_name": trial_dir.name,
                },
            )


def _enumerate_unique_trials(recordings_dirs: list[Path]) -> list[tuple[Path, str]]:
    """Walk sweep dirs, dedup by parsed (scenario, planner, robot_policy, seed)."""
    seen: set[tuple[str, str, str, int]] = set()
    out: list[tuple[Path, str]] = []
    for recordings_dir in recordings_dirs:
        if not recordings_dir.is_dir():
            continue
        for trial_dir in sorted(recordings_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            try:
                key = parse_trial_dir(trial_dir.name)
            except ValueError:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append((recordings_dir, trial_dir.name))
    return out


def _kinematics_row(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    per_agent = df.sort_values(["agent_id", "time"]).groupby("agent_id").apply(calculate_kinematic_metrics).reset_index()
    return {
        "bucket": meta["bucket"],
        "scenario": meta["scenario"],
        "planner": meta["planner"],
        "robot_policy": meta["robot_policy"],
        "seed": meta["seed"],
        "jerk": per_agent["mean_jerk"].mean(),
        "curvature": per_agent["mean_curvature"].mean(),
        "collisions": calculate_run_collisions(df),
    }


def compute_pairwise_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["scenario", "robot_policy", "seed", "agent_id"])
    for (scenario, robot_policy, seed, agent_id), group in tqdm(grouped, total=grouped.ngroups, unit="grp", desc="pairwise"):
        planners = sorted(group["planner"].unique())
        if len(planners) < 2:
            continue
        traj = {p: group[group["planner"] == p][["x", "y"]].values for p in planners}
        bucket = group["bucket"].iloc[0]
        for p1, p2 in itertools.combinations(planners, 2):
            t1, t2 = traj[p1], traj[p2]
            if len(t1) == 0 or len(t2) == 0:
                continue
            d = pairwise_hausdorff(t1, t2)
            rows.append(
                {
                    "bucket": bucket,
                    "scenario": scenario,
                    "robot_policy": robot_policy,
                    "seed": seed,
                    "agent_id": agent_id,
                    "p1": p1,
                    "p2": p2,
                    "class1": DRIVER_CLASS_FINE.get(p1, "unknown"),
                    "class2": DRIVER_CLASS_FINE.get(p2, "unknown"),
                    "same_class": DRIVER_CLASS_FINE.get(p1) == DRIVER_CLASS_FINE.get(p2),
                    "hausdorff": d,
                }
            )
    columns = ["bucket", "scenario", "robot_policy", "seed", "agent_id", "p1", "p2", "class1", "class2", "same_class", "hausdorff"]
    return pd.DataFrame(rows, columns=columns)


def _scenario_equal_weight(slice_df: pd.DataFrame) -> float:
    if slice_df.empty:
        return float("nan")
    return float(slice_df.groupby("scenario")["hausdorff"].mean().mean())


def _bootstrap_K(slice_df: pd.DataFrame, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    within = slice_df[slice_df["same_class"]]
    across = slice_df[~slice_df["same_class"]]
    scenarios = sorted(slice_df["scenario"].unique())
    if not scenarios or within.empty or across.empty:
        return float("nan"), float("nan")

    if len(scenarios) <= 1:
        w_arr = within["hausdorff"].to_numpy()
        a_arr = across["hausdorff"].to_numpy()
        if len(w_arr) == 0 or len(a_arr) == 0:
            return float("nan"), float("nan")
        ks = []
        for _ in range(n_iter):
            xb = w_arr[rng.integers(0, len(w_arr), len(w_arr))].mean()
            yb = a_arr[rng.integers(0, len(a_arr), len(a_arr))].mean()
            if xb > 0:
                ks.append(yb / xb)
    else:
        x_per = within.groupby("scenario")["hausdorff"].mean()
        y_per = across.groupby("scenario")["hausdorff"].mean()
        common = sorted(set(x_per.index) & set(y_per.index))
        if not common:
            return float("nan"), float("nan")
        x_arr = x_per.loc[common].to_numpy()
        y_arr = y_per.loc[common].to_numpy()
        n = len(common)
        ks = []
        for _ in range(n_iter):
            idx = rng.integers(0, n, n)
            xb = x_arr[idx].mean()
            yb = y_arr[idx].mean()
            if xb > 0:
                ks.append(yb / xb)

    if not ks:
        return float("nan"), float("nan")
    return float(np.percentile(ks, 5)), float(np.percentile(ks, 95))


def _bootstrap_mean(values: np.ndarray, group_keys: np.ndarray, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    """Scenario-clustered bootstrap CI on a mean. Mirrors `_bootstrap_K` clustering choice."""
    if values.size == 0:
        return float("nan"), float("nan")
    groups = sorted(set(group_keys.tolist()))
    if len(groups) <= 1:
        means = []
        for _ in range(n_iter):
            idx = rng.integers(0, values.size, values.size)
            means.append(values[idx].mean())
        return float(np.percentile(means, 5)), float(np.percentile(means, 95))
    per_group = pd.Series(values).groupby(pd.Series(group_keys)).mean()
    arr = per_group.loc[groups].to_numpy()
    n = len(groups)
    means = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        means.append(arr[idx].mean())
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def within_class_pairs(pairwise_df: pd.DataFrame, n_bootstrap: int = 1000, ci_seed: int = 0) -> pd.DataFrame:
    """Per-pair Hausdorff with scenario-clustered 90% bootstrap CIs, restricted to driver pairs in the same DRIVER_CLASS_FINE class. Generalizes the Sec.4.5 SFM~HSFM check to NSP/SocialGAIL and any future intra-class pair."""
    columns = ["bucket", "class", "p1", "p2", "mean_hausdorff", "ci_lo", "ci_hi", "n_pairs", "n_scenarios"]
    if pairwise_df.empty:
        return pd.DataFrame(columns=columns)

    df = pairwise_df.copy()
    df["class1_fine"] = df["p1"].map(DRIVER_CLASS_FINE)
    df["class2_fine"] = df["p2"].map(DRIVER_CLASS_FINE)
    intra = df[df["class1_fine"].notna() & (df["class1_fine"] == df["class2_fine"])].copy()
    if intra.empty:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(ci_seed)
    rows = []
    bucket_slices: list[tuple[str, pd.DataFrame]] = [(b, intra[intra["bucket"] == b]) for b in KNOWN_BUCKETS]
    pooled = intra[intra["bucket"].isin(KNOWN_BUCKETS)]
    if not pooled.empty:
        bucket_slices.append(("all", pooled))

    for label, slice_df in bucket_slices:
        if slice_df.empty:
            continue
        for (cls, p1, p2), grp in slice_df.groupby(["class1_fine", "p1", "p2"]):
            vals = grp["hausdorff"].to_numpy()
            scen = grp["scenario"].to_numpy()
            lo, hi = _bootstrap_mean(vals, scen, n_bootstrap, rng)
            point = float(grp.groupby("scenario")["hausdorff"].mean().mean())
            rows.append(
                {
                    "bucket": label,
                    "class": cls,
                    "p1": p1,
                    "p2": p2,
                    "mean_hausdorff": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_pairs": int(vals.size),
                    "n_scenarios": int(grp["scenario"].nunique()),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["bucket", "class", "p1", "p2"]).reset_index(drop=True)


def headline(pairwise_df: pd.DataFrame, n_bootstrap: int = 1000, ci_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(ci_seed)

    def _agg(slice_df: pd.DataFrame, label: str) -> dict:
        within = slice_df[slice_df["same_class"]]
        across = slice_df[~slice_df["same_class"]]
        x = _scenario_equal_weight(within)
        y = _scenario_equal_weight(across)
        k = y / x if x and not np.isnan(x) else float("nan")
        k_lo, k_hi = _bootstrap_K(slice_df, n_bootstrap, rng)
        return {
            "bucket": label,
            "within_class_mean": x,
            "across_class_mean": y,
            "ratio_K": k,
            "K_lo": k_lo,
            "K_hi": k_hi,
            "n_pairs": len(slice_df),
            "n_scenarios": slice_df["scenario"].nunique(),
        }

    rows = []
    for bucket in KNOWN_BUCKETS:
        slice_df = pairwise_df[pairwise_df["bucket"] == bucket]
        if slice_df.empty:
            continue
        rows.append(_agg(slice_df, bucket))

    pooled = pairwise_df[pairwise_df["bucket"].isin(KNOWN_BUCKETS)]
    if not pooled.empty:
        rows.append(_agg(pooled, "all"))

    columns = ["bucket", "within_class_mean", "across_class_mean", "ratio_K", "K_lo", "K_hi", "n_pairs", "n_scenarios"]
    return pd.DataFrame(rows, columns=columns)


def run_analysis(
    recordings_dirs: Path | list[Path],
    out_dir: Path,
    n_bootstrap: int = 1000,
    ci_seed: int = 0,
    framing_threshold: float = 1.2,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(recordings_dirs, Path):
        recordings_dirs = [recordings_dirs]

    total_trials = len(_enumerate_unique_trials(recordings_dirs))
    print(f"Streaming {total_trials} trials from {len(recordings_dirs)} dir(s): {[d.name for d in recordings_dirs]}")
    frames: list[pd.DataFrame] = []
    kin_rows: list[dict[str, Any]] = []
    bar = tqdm(iter_trials(recordings_dirs), total=total_trials, unit="trial", desc="load")
    for trial_df, meta in bar:
        frames.append(trial_df)
        kin_rows.append(_kinematics_row(trial_df, meta))
        bar.set_postfix_str(f"{meta['scenario']}/{meta['planner']}")
    bar.close()

    if not frames:
        raise RuntimeError(f"no usable trials under {[str(d) for d in recordings_dirs]}")

    df = pd.concat(frames, ignore_index=True)
    frames.clear()
    print(f"  {len(df)} rows | {df['scenario'].nunique()} scenarios x {df['planner'].nunique()} planners x {df['seed'].nunique()} seeds | buckets: {sorted(df['bucket'].unique())}")

    kin_df = pd.DataFrame(kin_rows)
    kin_path = out_dir / "kinematics_per_trial.csv"
    kin_df.to_csv(kin_path, index=False)
    print(f"  wrote {kin_path}")

    print("Pairwise Hausdorff...")
    pair_df = compute_pairwise_table(df)
    pair_path = out_dir / "pairwise_distance.csv"
    pair_df.to_csv(pair_path, index=False)
    print(f"  wrote {pair_path} ({len(pair_df)} pairs)")

    print("Self-divergence (within-driver across-seed Hausdorff)...")
    self_div_df = self_divergence_table(df)
    self_div_path = out_dir / "self_divergence.csv"
    self_div_df.to_csv(self_div_path, index=False)
    print(f"  wrote {self_div_path} ({len(self_div_df)} pairs)")

    print(f"Headline (n_bootstrap={n_bootstrap})...")
    head_df = headline(pair_df, n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    seed_df = headline_seed_ratio(pair_df, self_div_df, n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    if not seed_df.empty:
        head_df = head_df.merge(seed_df[["bucket", "within_driver_seed_mean", "K_seed", "K_seed_lo", "K_seed_hi", "n_self_pairs"]], on="bucket", how="left")
    head_path = out_dir / "headline.csv"
    head_df.to_csv(head_path, index=False)
    print(f"  wrote {head_path}")

    print("Variance decomposition (scenario x driver x seed on per-trial scalars)...")
    var_df = decompose_scalar_variance(kin_df)
    var_path = out_dir / "variance_decomposition.csv"
    var_df.to_csv(var_path, index=False)
    print(f"  wrote {var_path} ({len(var_df)} rows)")

    print("Within-class pair CIs...")
    intra_df = within_class_pairs(pair_df, n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    intra_path = out_dir / "within_class_pairs.csv"
    intra_df.to_csv(intra_path, index=False)
    print(f"  wrote {intra_path} ({len(intra_df)} rows)")

    print("Picking abstract framing...")
    framing = pick_framing(head_df, threshold=framing_threshold)
    framing_path = out_dir / "framing.md"
    framing_path.write_text(render_framing(framing))
    print(f"  wrote {framing_path}")

    print()
    print("=== Headline (X = within-class mean, Y = across-class mean, K = Y/X, [K_lo, K_hi] = 90% bootstrap CI) ===")
    print(head_df.to_string(index=False, float_format="{:.4f}".format))

    if not var_df.empty:
        print()
        print("=== Variance decomposition (per-trial scalars, share of total SS) ===")
        print(var_df.to_string(index=False, float_format="{:.3f}".format))

    if not intra_df.empty:
        print()
        print("=== Within-class pairs (mean Hausdorff, 90% bootstrap CI; fine-grained driver classes) ===")
        print(intra_df.to_string(index=False, float_format="{:.4f}".format))

    print()
    print(f"=== Recommended abstract framing: {framing['recommended']} ===")
    if framing["recommended"] != "incomplete":
        print(f"  X_nav={framing['X_nav']:.3f}m  K_nav={framing['K_nav']:.2f}x  K_bt={framing['K_bt']:.2f}x  K_het={framing['K_het']:.2f}x  K_pooled={framing['K_pooled']:.2f}x")
        print(f"  Full sentence + alternatives in {framing_path}")

    return {
        "kinematics": kin_df,
        "pairwise": pair_df,
        "self_divergence": self_div_df,
        "headline": head_df,
        "variance_decomposition": var_df,
        "framing": framing,
    }
