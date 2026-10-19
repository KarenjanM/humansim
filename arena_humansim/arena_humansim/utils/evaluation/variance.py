import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from arena_humansim.utils.evaluation.buckets import (
    DRIVER_CLASS,
    KNOWN_BUCKETS,
)
from arena_humansim.utils.evaluation.metrics import pairwise_hausdorff


def self_divergence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Within-driver across-seed Hausdorff. Seed-axis companion to compute_pairwise_table.

    For each (scenario, agent_id, planner) cell with >= 2 seeds, computes the
    Hausdorff distance between every seed-pair of trajectories. K_seed downstream
    compares the mean of this against the mean of the across-driver pairwise table.
    """
    rows = []
    group_cols = ["scenario", "agent_id", "planner"]
    if "robot_policy" in df.columns:
        group_cols.append("robot_policy")
    grouped = df.groupby(group_cols)
    for keys, group in tqdm(grouped, total=grouped.ngroups, unit="grp", desc="self-div"):
        if not isinstance(keys, tuple):
            keys = (keys,)
        named = dict(zip(group_cols, keys, strict=False))
        scenario = named["scenario"]
        agent_id = named["agent_id"]
        planner = named["planner"]
        robot_policy = named.get("robot_policy", "")
        seeds = sorted(group["seed"].unique())
        if len(seeds) < 2:
            continue
        traj = {s: group[group["seed"] == s][["x", "y"]].values for s in seeds}  # pyright: ignore[reportAttributeAccessIssue]
        bucket = group["bucket"].iloc[0]
        cls = DRIVER_CLASS.get(planner, "unknown")  # pyright: ignore[reportArgumentType, reportCallIssue]
        for s_i, s_j in itertools.combinations(seeds, 2):
            t_i, t_j = traj[s_i], traj[s_j]
            if len(t_i) == 0 or len(t_j) == 0:
                continue
            rows.append(
                {
                    "bucket": bucket,
                    "scenario": scenario,
                    "agent_id": agent_id,
                    "planner": planner,
                    "robot_policy": robot_policy,
                    "class": cls,
                    "seed_i": s_i,
                    "seed_j": s_j,
                    "hausdorff": pairwise_hausdorff(t_i, t_j),  # pyright: ignore[reportArgumentType]
                }
            )
    columns = ["bucket", "scenario", "agent_id", "planner", "robot_policy", "class", "seed_i", "seed_j", "hausdorff"]
    return pd.DataFrame(rows, columns=columns)


def _scenario_equal_weight(slice_df: pd.DataFrame) -> float:
    if slice_df.empty:
        return float("nan")
    return float(slice_df.groupby("scenario")["hausdorff"].mean().mean())  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]


def _bootstrap_ratio(num_df: pd.DataFrame, den_df: pd.DataFrame, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    """Scenario-clustered 90% bootstrap CI on mean(num) / mean(den). Mirrors _bootstrap_K in analyze.py."""
    if num_df.empty or den_df.empty:
        return float("nan"), float("nan")

    common = sorted(set(num_df["scenario"].unique()) & set(den_df["scenario"].unique()))
    if not common:
        return float("nan"), float("nan")

    if len(common) <= 1:
        n_arr = num_df["hausdorff"].to_numpy()
        d_arr = den_df["hausdorff"].to_numpy()
        if n_arr.size == 0 or d_arr.size == 0:
            return float("nan"), float("nan")
        ks = []
        for _ in range(n_iter):
            xb = n_arr[rng.integers(0, n_arr.size, n_arr.size)].mean()
            yb = d_arr[rng.integers(0, d_arr.size, d_arr.size)].mean()
            if yb > 0:
                ks.append(xb / yb)
    else:
        n_per = num_df.groupby("scenario")["hausdorff"].mean()
        d_per = den_df.groupby("scenario")["hausdorff"].mean()
        n_arr = n_per.loc[common].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
        d_arr = d_per.loc[common].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
        n = len(common)
        ks = []
        for _ in range(n_iter):
            idx = rng.integers(0, n, n)
            xb = n_arr[idx].mean()
            yb = d_arr[idx].mean()
            if yb > 0:
                ks.append(xb / yb)

    if not ks:
        return float("nan"), float("nan")
    return float(np.percentile(ks, 5)), float(np.percentile(ks, 95))


def headline_seed_ratio(
    pairwise_df: pd.DataFrame,
    self_div_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci_seed: int = 0,
) -> pd.DataFrame:
    """Per-bucket K_seed = across-class Hausdorff / within-driver across-seed Hausdorff.

    The numerator is the across-class pair set of K. On flow-spawned scenarios an
    agent id under a new seed is a different person, which inflates the
    denominator. Does not separate
    scheduler-induced from seed-induced variance - for that, same-seed reruns are
    required.
    """
    rng = np.random.default_rng(ci_seed)
    pairwise_df = pairwise_df[~pairwise_df["same_class"].astype(bool)]  # pyright: ignore[reportAssignmentType]

    def _agg(num: pd.DataFrame, den: pd.DataFrame, label: str) -> dict:
        x = _scenario_equal_weight(num)
        y = _scenario_equal_weight(den)
        k = x / y if y and not np.isnan(y) else float("nan")
        k_lo, k_hi = _bootstrap_ratio(num, den, n_bootstrap, rng)
        return {
            "bucket": label,
            "across_class_mean": x,
            "within_driver_seed_mean": y,
            "K_seed": k,
            "K_seed_lo": k_lo,
            "K_seed_hi": k_hi,
            "n_cross_pairs": int(len(num)),
            "n_self_pairs": int(len(den)),
            "n_scenarios": int(num["scenario"].nunique()) if not num.empty else 0,  # pyright: ignore[reportArgumentType]
        }

    rows = []
    for bucket in KNOWN_BUCKETS:
        num = pairwise_df[pairwise_df["bucket"] == bucket]
        den = self_div_df[self_div_df["bucket"] == bucket]
        if num.empty and den.empty:
            continue
        rows.append(_agg(num, den, bucket))  # pyright: ignore[reportArgumentType]

    pooled_num = pairwise_df[pairwise_df["bucket"].isin(KNOWN_BUCKETS)]
    pooled_den = self_div_df[self_div_df["bucket"].isin(KNOWN_BUCKETS)]
    if not pooled_num.empty and not pooled_den.empty:
        rows.append(_agg(pooled_num, pooled_den, "all"))  # pyright: ignore[reportArgumentType]

    return pd.DataFrame(rows)


SCALAR_METRICS: tuple[str, ...] = ("jerk", "curvature", "collisions")


def _ss_main(slice_df: pd.DataFrame, factor: str, metric: str, grand: float) -> float:
    cell_mean = slice_df.groupby(factor)[metric].transform("mean")
    return float(((cell_mean - grand) ** 2).sum())  # pyright: ignore[reportOperatorIssue]


def _ss_two(slice_df: pd.DataFrame, f1: str, f2: str, metric: str, grand: float) -> float:
    m1 = slice_df.groupby(f1)[metric].transform("mean")
    m2 = slice_df.groupby(f2)[metric].transform("mean")
    m12 = slice_df.groupby([f1, f2])[metric].transform("mean")
    interaction = m12 - m1 - m2 + grand  # pyright: ignore[reportOperatorIssue]
    return float((interaction**2).sum())


def decompose_scalar_variance(kin_df: pd.DataFrame, metrics: Iterable[str] = SCALAR_METRICS) -> pd.DataFrame:
    """Three-way ANOVA-style variance decomposition on per-trial scalars.

    Factors: scenario x planner x seed, one trial per cell. Without replication the
    three-way interaction is folded into the residual. Type-I SS is approximate when
    the design is unbalanced - `n_trials` is reported so imbalance is visible.
    Shares are reported as fraction of total SS and may not sum to exactly 1.0 due
    to rounding when residual = 0.
    """
    rows = []
    bucket_slices: list[tuple[str, pd.DataFrame]] = [(b, kin_df[kin_df["bucket"] == b]) for b in KNOWN_BUCKETS]  # pyright: ignore[reportAssignmentType]
    pooled = kin_df[kin_df["bucket"].isin(KNOWN_BUCKETS)]
    if not pooled.empty:
        bucket_slices.append(("all", pooled))  # pyright: ignore[reportArgumentType]

    for label, slice_df in bucket_slices:
        if slice_df.empty or slice_df["planner"].nunique() < 2 or slice_df["seed"].nunique() < 2 or slice_df["scenario"].nunique() < 1:
            continue
        for metric in metrics:
            if metric not in slice_df.columns:
                continue
            y = slice_df[metric].to_numpy(dtype=float)
            if y.size == 0:
                continue
            mask = np.isfinite(y)
            if not mask.all():
                if not mask.any():
                    continue
                slice_clean = slice_df.iloc[mask]
                y = y[mask]
            else:
                slice_clean = slice_df
            grand = float(y.mean())
            ss_total = float(((y - grand) ** 2).sum())
            if ss_total <= 0:
                continue
            ss_s = _ss_main(slice_clean, "scenario", metric, grand)
            ss_d = _ss_main(slice_clean, "planner", metric, grand)
            ss_g = _ss_main(slice_clean, "seed", metric, grand)
            ss_sd = _ss_two(slice_clean, "scenario", "planner", metric, grand)
            ss_sg = _ss_two(slice_clean, "scenario", "seed", metric, grand)
            ss_dg = _ss_two(slice_clean, "planner", "seed", metric, grand)
            ss_residual = max(0.0, ss_total - (ss_s + ss_d + ss_g + ss_sd + ss_sg + ss_dg))
            rows.append(
                {
                    "bucket": label,
                    "metric": metric,
                    "n_trials": int(slice_clean.shape[0]),
                    "share_scenario": ss_s / ss_total,
                    "share_driver": ss_d / ss_total,
                    "share_seed": ss_g / ss_total,
                    "share_scenario_x_driver": ss_sd / ss_total,
                    "share_scenario_x_seed": ss_sg / ss_total,
                    "share_driver_x_seed": ss_dg / ss_total,
                    "share_residual": ss_residual / ss_total,
                    "ss_total": ss_total,
                }
            )

    return pd.DataFrame(rows)
