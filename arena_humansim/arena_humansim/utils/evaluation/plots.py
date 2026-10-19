"""Paper figures from the analysis CSVs of a divergence sweep and a closed-loop sweep.

    evaluate plots --divergence <dir> --robots <dir> --out_dir <dir>

Every figure is built from the CSVs `evaluate analyze` writes. Files written (PDF):

  k_per_scenario           K per scenario, sorted, with the bucket means as lines
  pairwise_heatmap         scenario-weighted mean Hausdorff for all driver pairs, per bucket
  kinematic_distributions  per-trial jerk, curvature and collisions per driver, per bucket
  noise_floor              across-driver vs across-seed Hausdorff (K_seed) per bucket
  variance_decomposition   share of variance by scenario, driver, seed and interactions
  sensitivity_profile      policy x driver cells on success, PSV/s and collisions/trial
  failures                 failure-cause shares per policy and bucket
  precision_curve          K and its CI as a function of the number of seeds
  launch_overhead          per-trial launch overhead as a share of wall time
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd

from arena_humansim.local_planner import LocalPlanner
from arena_humansim.utils.evaluation.buckets import DRIVER_CLASS_FINE
from arena_humansim.utils.evaluation.partitions import k_with_ci

DRIVER_ORDER = [name for name, info in LocalPlanner.info().items() if not info.robot_policy]
DRIVER_LABEL = {name: LocalPlanner.labels()[name] for name in DRIVER_ORDER}
POLICY_ORDER = [name for name, info in LocalPlanner.info().items() if info.robot_policy]
POLICY_LABEL = {name: LocalPlanner.labels()[name] for name in POLICY_ORDER}
_PAPER_COLOR = {"sfm": "#1f77b4", "hsfm": "#aec7e8", "orca": "#ff7f0e", "straight": "#7f7f7f", "nsp": "#9467bd", "socialgail": "#1b9e77"}
_EXTRA_COLORS = ["#2ca02c", "#98df8a", "#17becf", "#9edae5", "#bcbd22", "#dbdb8d", "#d62728", "#ff9896", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2"]
_UNPAINTED = [name for name in DRIVER_ORDER if name not in _PAPER_COLOR]
DRIVER_COLOR = _PAPER_COLOR | {name: _EXTRA_COLORS[i % len(_EXTRA_COLORS)] for i, name in enumerate(_UNPAINTED)}
BUCKET_ROW_LABEL = {"nav": "Pure-nav", "bt": "BT-load", "het": "Heterogeneous"}
# inches
SIZE_SENSITIVITY = (10.456, 2.757)
SIZE_KINEMATIC = (6.742, 5.760)
SIZE_LAUNCH = (4.364, 2.169)
BUCKET_LABEL = {"nav": "pure-navigation", "bt": "behavior-tree", "het": "heterogeneous", "all": "pooled"}
CAUSES = ["success", "collision_static", "collision_with_reactor", "freezing_timeout", "goal_overshoot", "timeout_no_progress", "other"]
CAUSE_LABEL = {"success": "Success", "collision_static": "Col.-S.", "collision_with_reactor": "Col.-H.", "freezing_timeout": "Freeze", "goal_overshoot": "Oversh.", "timeout_no_progress": "Timeout", "other": "Other"}


def _plt() -> types.ModuleType:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _label_drivers(names: list[str]) -> list[str]:
    return [DRIVER_LABEL.get(n, n) for n in names]


def _present(order: list[str], *columns: pd.Series) -> list[str]:
    seen = set().union(*(set(c.dropna().unique()) for c in columns))
    return [name for name in order if name in seen]


def _family_edges(drivers: list[str]) -> list[int]:
    return [k for k in range(1, len(drivers)) if DRIVER_CLASS_FINE[drivers[k]] != DRIVER_CLASS_FINE[drivers[k - 1]]]


def scenario_k_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario within/across class means and K, from the pair table."""
    rows = []
    for (bucket, scen), g in pairwise.groupby(["bucket", "scenario"]):  # pyright: ignore[reportGeneralTypeIssues]
        w = g.loc[g["same_class"], "hausdorff"].mean()
        a = g.loc[~g["same_class"], "hausdorff"].mean()
        rows.append({"bucket": bucket, "scenario": scen, "within": w, "across": a, "K": a / w if w > 0 else np.nan, "n_pairs": len(g)})
    return pd.DataFrame(rows).sort_values("K", ascending=False).reset_index(drop=True)


def plot_k_per_scenario(pairwise: pd.DataFrame, headline: pd.DataFrame, out: Path) -> pd.DataFrame:
    plt = _plt()
    table = scenario_k_table(pairwise)
    table = table[np.isfinite(table["K"])]
    colors = {"nav": "#4c78a8", "bt": "#f58518", "het": "#54a24b"}
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(table))
    ax.bar(x, table["K"], color=[colors.get(b, "#888") for b in table["bucket"]])
    ax.axhline(1.0, color="k", lw=0.8)
    for _, row in headline.iterrows():
        if row["bucket"] in colors:
            ax.axhline(row["ratio_K"], color=colors[row["bucket"]], lw=0.8, ls="--")  # pyright: ignore[reportArgumentType]
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in table["scenario"]], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("K = across-class / within-class")
    present = [b for b in colors if (table["bucket"] == b).any()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[b]) for b in present]
    ax.legend(handles, [BUCKET_LABEL[b] for b in present], loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return table  # pyright: ignore[reportReturnType]


def pair_matrix(pairwise: pd.DataFrame, bucket: str | None) -> pd.DataFrame:
    sub = pairwise if bucket is None else pairwise[pairwise["bucket"] == bucket]
    m = sub.groupby(["p1", "p2", "scenario"])["hausdorff"].mean().groupby(["p1", "p2"]).mean()  # pyright: ignore[reportAttributeAccessIssue]
    drivers = _present(DRIVER_ORDER, pairwise["p1"], pairwise["p2"])  # pyright: ignore[reportArgumentType]
    mat = pd.DataFrame(np.nan, index=drivers, columns=drivers)
    for (p1, p2), v in m.items():  # pyright: ignore[reportAttributeAccessIssue, reportGeneralTypeIssues]
        if p1 in mat.index and p2 in mat.columns:
            mat.loc[p1, p2] = v
            mat.loc[p2, p1] = v
    return mat


def plot_pairwise_heatmap(pairwise: pd.DataFrame, out: Path) -> dict[str, pd.DataFrame]:
    plt = _plt()
    buckets = [b for b in ("nav", "bt", "het") if (pairwise["bucket"] == b).any()]
    fig, axes = plt.subplots(1, len(buckets), figsize=(3.2 * len(buckets), 3.0), squeeze=False)
    mats = {}
    for ax, b in zip(axes[0], buckets, strict=True):
        mat = pair_matrix(pairwise, b)
        mats[b] = mat
        im = ax.imshow(mat.to_numpy(dtype=float), cmap="viridis")
        drivers = list(mat.index)
        ax.set_xticks(range(len(drivers)))
        ax.set_yticks(range(len(drivers)))
        ax.set_xticklabels(_label_drivers(drivers), rotation=45, ha="right")
        ax.set_yticklabels(_label_drivers(drivers))
        for i in range(len(drivers)):
            for j in range(len(drivers)):
                v = mat.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color="w" if v < np.nanmax(mat.to_numpy()) * 0.6 else "k")
        ax.set_title(f"{BUCKET_LABEL[b]} (mean Hausdorff, m)")
        for k in _family_edges(drivers):
            ax.axhline(k - 0.5, color="w", lw=1.2)
            ax.axvline(k - 0.5, color="w", lw=1.2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return mats


def plot_kinematic_distributions(kin: pd.DataFrame, out: Path, het_kin: pd.DataFrame | None = None) -> None:
    """Per-driver violins of the three per-trial scalars, one row per bucket, the het row from `het_kin`."""
    plt = _plt()
    rows = [(b, kin[kin["bucket"] == b]) for b in ("nav", "bt") if (kin["bucket"] == b).any()]
    if het_kin is not None and (het_kin["bucket"] == "het").any():
        rows.append(("het", het_kin[het_kin["bucket"] == "het"]))
    metrics = [("jerk", "Jerk (m/s$^3$)", False), ("curvature", "Curvature (1/m)", False), ("collisions", "Collisions / trial", True)]
    drivers = _present(DRIVER_ORDER, *(sub["planner"] for _, sub in rows))  # pyright: ignore[reportArgumentType]
    fig, axes = plt.subplots(len(rows), 3, figsize=SIZE_KINEMATIC, squeeze=False)
    for r, (b, sub) in enumerate(rows):
        for c, (metric, label, log) in enumerate(metrics):
            ax = axes[r][c]
            data = []
            for d in drivers:
                v = sub.loc[sub["planner"] == d, metric].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if log:
                    v = np.log10(np.maximum(v, 1.0))
                data.append(v if v.size else np.array([np.nan]))
            parts = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.8)
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(DRIVER_COLOR[drivers[i]])
                body.set_edgecolor("none")
                body.set_alpha(0.8)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                parts[key].set_color("k")
                parts[key].set_linewidth(0.7)
            ax.set_xticks(range(1, len(drivers) + 1))
            ax.set_xticklabels(drivers if r == len(rows) - 1 else [], rotation=45, ha="right")
            if log:
                lo, hi = 0, int(np.ceil(np.nanmax([np.nanmax(v) for v in data])))
                ax.set_yticks(range(lo, hi + 1))
                ax.set_yticklabels([f"$10^{{{k}}}$" for k in range(lo, hi + 1)])
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(BUCKET_ROW_LABEL[b])
            ax.grid(axis="y", alpha=0.3)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_launch_overhead(overhead: pd.DataFrame, out: Path) -> dict[str, float]:
    """Histogram of per-trial launch overhead as a share of total wall time."""
    plt = _plt()
    v = overhead["overhead_pct"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    med, p95 = float(np.median(v)), float(np.percentile(v, 95))
    fig, ax = plt.subplots(figsize=SIZE_LAUNCH)
    ax.hist(v, bins=40, color="#5b9bd5", edgecolor="w", linewidth=0.4)
    ax.axvline(med, color="k", ls="--", lw=0.9, label=f"median = {med:.1f}%")
    ax.axvline(p95, color="r", ls=":", lw=0.9, label=f"95th pct = {p95:.1f}%")
    ax.set_xlabel("Launch overhead (% of total wall time)")
    ax.set_ylabel("Trials")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {"median_pct": med, "p95_pct": p95, "n_trials": float(v.size)}


def plot_noise_floor(headline: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    rows = headline[headline["bucket"].isin(["nav", "bt", "het", "all"])]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(rows))
    w = 0.26
    ax.bar(x - w, rows["within_class_mean"], w, label="within class", color="#4c78a8")
    ax.bar(x, rows["across_class_mean"], w, label="across class", color="#e45756")
    if "within_driver_seed_mean" in rows:
        ax.bar(x + w, rows["within_driver_seed_mean"], w, label="same driver, other seed", color="#bab0ac")
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL.get(b, b) for b in rows["bucket"]])
    ax.set_ylabel("mean Hausdorff (m)")
    for xi, (_, r) in zip(x, rows.iterrows(), strict=True):
        ax.text(xi, max(r["within_class_mean"], r["across_class_mean"]) * 1.02, f"K={r['ratio_K']:.2f}", ha="center", fontsize=7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_variance_decomposition(var: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    cols = ["share_scenario", "share_driver", "share_seed", "share_scenario_x_driver", "share_scenario_x_seed", "share_driver_x_seed", "share_residual"]
    labels = ["scenario", "driver", "seed", "scenario x driver", "scenario x seed", "driver x seed", "residual"]
    sub = var[var["bucket"].isin(["nav", "bt", "all"])].copy()
    sub["row"] = sub["bucket"].map(BUCKET_LABEL) + " / " + sub["metric"]  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
    fig, ax = plt.subplots(figsize=(6.0, 0.35 * len(sub) + 1.2))
    left = np.zeros(len(sub))
    palette = ["#4c78a8", "#e45756", "#bab0ac", "#72b7b2", "#f58518", "#54a24b", "#dddddd"]
    for col, lab, colr in zip(cols, labels, palette, strict=True):
        vals = sub[col].to_numpy(dtype=float)  # pyright: ignore[reportAttributeAccessIssue]
        ax.barh(sub["row"], vals, left=left, color=colr, label=lab)
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of total sum of squares")
    ax.invert_yaxis()
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_sensitivity_profile(cells: pd.DataFrame, out: Path) -> pd.DataFrame:
    """policy x driver heat cells on three closed-loop metrics, pooled over scenarios weighted by trial count."""
    plt = _plt()
    metrics = [("success_mean", "Success rate", "viridis", ".2f"), ("psv_per_sec_mean", "PSV / sec", "magma_r", ".2f"), ("n_robot_collisions_mean", "Robot collisions", "magma_r", ".1f")]
    pooled = []
    for (pol, drv), g in cells.groupby(["robot_policy", "ped_planner"]):  # pyright: ignore[reportGeneralTypeIssues]
        row = {"robot_policy": pol, "ped_planner": drv, "n_trials": int(g["n_trials"].sum())}  # pyright: ignore[reportArgumentType]
        for m, _, _, _ in metrics:
            row[m] = float(np.average(g[m], weights=g["n_trials"]))
        pooled.append(row)
    pooled = pd.DataFrame(pooled)
    fs_cell, fs_tick, fs_title = 10.0, 10.5, 12.0
    drivers = _present(DRIVER_ORDER, pooled["ped_planner"])  # pyright: ignore[reportArgumentType]
    policies = _present(POLICY_ORDER, pooled["robot_policy"])  # pyright: ignore[reportArgumentType]
    fig, axes = plt.subplots(1, 3, figsize=SIZE_SENSITIVITY)
    for ax, (m, label, cmap, fmt) in zip(axes, metrics, strict=True):
        mat = pooled.pivot(index="robot_policy", columns="ped_planner", values=m).reindex(index=policies, columns=drivers)
        vmin, vmax = (0.0, 1.0) if m == "success_mean" else (float(np.nanmin(mat.to_numpy(dtype=float))), float(np.nanmax(mat.to_numpy(dtype=float))))
        im = ax.imshow(mat.to_numpy(dtype=float), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(drivers)))
        ax.set_xticklabels(_label_drivers(drivers), rotation=45, ha="right", rotation_mode="anchor", fontsize=fs_tick)
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels([POLICY_LABEL[p] for p in policies], fontsize=fs_tick)
        for i in range(len(policies)):
            for j in range(len(drivers)):
                v = mat.iat[i, j]
                if np.isfinite(v):
                    r, g, b, _ = im.cmap(im.norm(v))
                    dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
                    ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=fs_cell, color="w" if dark else "k")
        for k in _family_edges(drivers):
            ax.axvline(k - 0.5, color="w", lw=3.0)
        ax.set_title(label, fontsize=fs_title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=fs_tick)
    fig.tight_layout(pad=0.4, w_pad=0.4)
    fig.savefig(out)
    plt.close(fig)
    return pooled


def plot_failures(fail: pd.DataFrame, out: Path) -> pd.DataFrame:
    plt = _plt()
    buckets = [b for b in ("nav", "bt", "het") if (fail["bucket"] == b).any()]
    piv = fail.pivot_table(index=["bucket", "robot_policy"], columns="cause", values="fraction", fill_value=0.0)
    for c in CAUSES:
        if c not in piv:
            piv[c] = 0.0
    piv = piv[CAUSES]
    fig, axes = plt.subplots(1, len(buckets), figsize=(3.0 * len(buckets), 2.4), squeeze=False, sharey=True)
    palette = ["#54a24b", "#bab0ac", "#e45756", "#4c78a8", "#f58518", "#79706e", "#dddddd"]
    for ax, b in zip(axes[0], buckets, strict=True):
        sub = piv.loc[b].reindex(POLICY_ORDER).fillna(0.0)
        left = np.zeros(len(sub))
        for c, colr in zip(CAUSES, palette, strict=True):
            ax.barh([POLICY_LABEL[p] for p in sub.index], sub[c], left=left, color=colr, label=CAUSE_LABEL[c])
            left += sub[c].to_numpy()
        ax.set_xlim(0, 1)
        ax.set_title(BUCKET_LABEL[b])
        ax.invert_yaxis()
    axes[0][-1].legend(ncol=1, frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return piv.reset_index()


def precision_curve(pairwise: pd.DataFrame, n_bootstrap: int = 1000, ci_seed: int = 0) -> pd.DataFrame:
    """K and its scenario-clustered 90% CI when only the first k seeds are used."""
    rng = np.random.default_rng(ci_seed)
    seeds = sorted(pairwise["seed"].unique())
    rows = []
    for b in ("nav", "bt"):
        for k in sorted({1, 2, 3, 5, 10, len(seeds)} & set(range(1, len(seeds) + 1))):
            sub = pairwise[(pairwise["bucket"] == b) & (pairwise["seed"] <= seeds[k - 1])]
            if sub.empty:
                continue
            _, _, ratio, lo, hi, n = k_with_ci(sub, sub["same_class"].astype(bool), n_bootstrap, rng)  # pyright: ignore[reportArgumentType]
            rows.append({"bucket": b, "n_seeds": k, "K": ratio, "K_lo": lo, "K_hi": hi, "n_scenarios": n})
    return pd.DataFrame(rows)


def plot_precision_curve(curve: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for b, colr in (("nav", "#4c78a8"), ("bt", "#f58518")):
        sub = curve[curve["bucket"] == b]
        if sub.empty:
            continue
        ax.plot(sub["n_seeds"], sub["K"], "o-", color=colr, label=BUCKET_LABEL[b], ms=3)
        ax.fill_between(sub["n_seeds"], sub["K_lo"], sub["K_hi"], color=colr, alpha=0.2)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("seeds per (scenario, driver)")
    ax.set_ylabel("K with 90% scenario-clustered CI")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run_plots(divergence: Path | None, robots: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if divergence is not None:
        pairwise = pd.read_csv(divergence / "pairwise_distance.csv")
        headline = pd.read_csv(divergence / "headline.csv")
        table = plot_k_per_scenario(pairwise, headline, out_dir / "k_per_scenario.pdf")
        table.to_csv(out_dir / "k_per_scenario.csv", index=False)
        mats = plot_pairwise_heatmap(pairwise, out_dir / "pairwise_heatmap.pdf")
        for b, mat in mats.items():
            mat.to_csv(out_dir / f"pairwise_matrix_{b}.csv")
        kin_path = divergence / "kinematics_per_trial.csv"
        if kin_path.exists():
            het_path = robots / "kinematics_per_trial.csv" if robots is not None else None
            het_kin = pd.read_csv(het_path) if het_path is not None and het_path.exists() else None
            plot_kinematic_distributions(pd.read_csv(kin_path), out_dir / "kinematic_distributions.pdf", het_kin)
        plot_noise_floor(headline, out_dir / "noise_floor.pdf")
        var_path = divergence / "variance_decomposition.csv"
        if var_path.exists():
            var = pd.read_csv(var_path)
            if not var.empty:
                plot_variance_decomposition(var, out_dir / "variance_decomposition.pdf")
        curve = precision_curve(pairwise)
        curve.to_csv(out_dir / "precision_curve.csv", index=False)
        plot_precision_curve(curve, out_dir / "precision_curve.pdf")
        print(f"divergence figures written to {out_dir}")
    if robots is not None:
        cells = pd.read_csv(robots / "robots_cell.csv")
        pooled = plot_sensitivity_profile(cells, out_dir / "sensitivity_profile.pdf")
        pooled.to_csv(out_dir / "sensitivity_cells.csv", index=False)
        fail_path = robots / "failures_by_policy_bucket.csv"
        if fail_path.exists():
            piv = plot_failures(pd.read_csv(fail_path), out_dir / "failures.pdf")
            piv.to_csv(out_dir / "failures_table.csv", index=False)
        print(f"closed-loop figures written to {out_dir}")
    overhead = [pd.read_csv(d / "launch_overhead.csv") for d in (divergence, robots) if d is not None and (d / "launch_overhead.csv").exists()]
    if overhead:
        stats = plot_launch_overhead(pd.concat(overhead, ignore_index=True), out_dir / "launch_overhead.pdf")
        print(f"launch overhead: median {stats['median_pct']:.1f}%, p95 {stats['p95_pct']:.1f}% over {int(stats['n_trials'])} trials")
