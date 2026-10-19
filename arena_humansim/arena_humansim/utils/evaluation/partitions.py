"""K under alternative driver-class partitions, from a pair table.

    evaluate partitions --recordings_dir <analyzed sweep> [--out_dir <dir>]

K per bucket for every partition below, with the same scenario-clustered 90%
bootstrap as the headline K (which uses the fine partition):

  fine            the planner families: force, geometric, no_avoidance, learned
  binary          classical (every family but learned) against learned
  binary_reactive binary partition with straight excluded
  fine_reactive   fine partition with straight excluded
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from arena_humansim.utils.evaluation.buckets import DRIVER_CLASS, DRIVER_CLASS_FINE


def _reactive(classes: dict[str, str]) -> dict[str, str]:
    return {d: c for d, c in classes.items() if d != "straight"}


PARTITIONS: dict[str, dict[str, str]] = {
    "fine": DRIVER_CLASS_FINE,
    "binary": DRIVER_CLASS,
    "binary_reactive": _reactive(DRIVER_CLASS),
    "fine_reactive": _reactive(DRIVER_CLASS_FINE),
}


def k_with_ci(sub: pd.DataFrame, same: pd.Series, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float, float, float, int]:
    w = sub.loc[same].groupby("scenario")["hausdorff"].mean()
    a = sub.loc[~same].groupby("scenario")["hausdorff"].mean()
    common = sorted(set(w.index) & set(a.index))
    if not common:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    wa, aa = w.loc[common].to_numpy(), a.loc[common].to_numpy()
    ks = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(common), len(common))
        if wa[idx].mean() > 0:
            ks.append(aa[idx].mean() / wa[idx].mean())
    return float(wa.mean()), float(aa.mean()), float(aa.mean() / wa.mean()), float(np.percentile(ks, 5)), float(np.percentile(ks, 95)), len(common)


def k_by_partition(pairwise: pd.DataFrame, n_bootstrap: int = 1000, ci_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(ci_seed)
    rows = []
    buckets = [b for b in ("nav", "bt", "het") if (pairwise["bucket"] == b).any()] + ["all"]
    for name, classes in PARTITIONS.items():
        df = pairwise[pairwise["p1"].isin(classes) & pairwise["p2"].isin(classes)]
        same = df["p1"].map(classes) == df["p2"].map(classes)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        for b in buckets:
            sub = df if b == "all" else df[df["bucket"] == b]
            if sub.empty:  # pyright: ignore[reportAttributeAccessIssue]
                continue
            within, across, k, lo, hi, n = k_with_ci(sub, same.loc[sub.index], n_bootstrap, rng)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
            rows.append({"partition": name, "bucket": b, "within_mean": within, "across_mean": across, "K": k, "K_lo": lo, "K_hi": hi, "n_scenarios": n, "n_within_pairs": int(same.loc[sub.index].sum()), "n_across_pairs": int((~same.loc[sub.index]).sum())})  # pyright: ignore[reportAttributeAccessIssue]
    return pd.DataFrame(rows)


def run_partitions(recordings_dir: Path, out_dir: Path | None = None, n_bootstrap: int = 1000, ci_seed: int = 0) -> pd.DataFrame:
    pairwise = pd.read_csv(recordings_dir / "pairwise_distance.csv")
    table = k_by_partition(pairwise, n_bootstrap=n_bootstrap, ci_seed=ci_seed)
    out_dir = out_dir or recordings_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "k_by_partition.csv", index=False)
    print(table.to_string(index=False, float_format="{:.3f}".format))
    return table
