"""Three-level pickle cache for bag extraction.

Levels (cheapest hit first):
  - multi-sweep: <commonpath>/_multi_<hash>.pkl, keyed on the sorted sweep paths
  - per-sweep:   <sweep_dir>/_sweep_extracted.pkl
  - per-trial:   <trial_dir>/_extracted.pkl

A cache is fresh iff its mtime is >= every source it depends on:
  - per-trial pkl vs the bag's metadata.yaml mtime
  - per-sweep pkl vs every per-trial pkl mtime (after they're materialized)
  - multi-sweep pkl vs every per-sweep pkl mtime, and each per-sweep pkl vs its newest bag

A trial whose bag is unreadable is cached as an empty df, so we don't keep
retrying within or across runs. Re-recording the bag updates metadata.yaml
mtime and invalidates the entry.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from arena_humansim.utils.bag_io import extract_agent_states

_TRIAL_PKL = "_extracted.pkl"
_SWEEP_PKL = "_sweep_extracted.pkl"


def _bag_mtime(trial_dir: Path) -> float:
    metadata = trial_dir / "bag" / "metadata.yaml"
    return metadata.stat().st_mtime if metadata.exists() else 0.0


def load_trial(trial_dir: Path) -> pd.DataFrame:
    bag_dir = trial_dir / "bag"
    if not bag_dir.exists():
        return pd.DataFrame()
    cache = trial_dir / _TRIAL_PKL
    bag_mt = _bag_mtime(trial_dir)
    if cache.exists() and cache.stat().st_mtime >= bag_mt:
        try:
            return pd.read_pickle(cache)  # pyright: ignore[reportReturnType]
        except Exception as exc:
            print(f"  unreadable cache {cache.name}: {type(exc).__name__}: {exc}; re-extracting")
    try:
        df = extract_agent_states(bag_dir)
    except Exception as exc:
        print(f"  unreadable bag in {trial_dir.name}: {type(exc).__name__}: {exc}")
        df = pd.DataFrame()
    cache.write_bytes(pickle.dumps(df))
    return df


def _trial_dirs(sweep_dir: Path) -> list[Path]:
    return [d for d in sorted(sweep_dir.iterdir()) if d.is_dir() and (d / "bag").exists()]


def load_sweep(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    cache = sweep_dir / _SWEEP_PKL
    trial_dirs = _trial_dirs(sweep_dir)
    newest_bag = max((_bag_mtime(d) for d in trial_dirs), default=0.0)
    if cache.exists() and cache.stat().st_mtime >= newest_bag:
        try:
            cached = pd.read_pickle(cache)
        except Exception as exc:
            print(f"  unreadable cache {cache.name}: {type(exc).__name__}: {exc}; re-aggregating")
        else:
            if set(cached.keys()) == {d.name for d in trial_dirs}:
                return cached  # pyright: ignore[reportReturnType]
    sweep: dict[str, pd.DataFrame] = {}
    for trial_dir in tqdm(trial_dirs, unit="trial", desc=f"extract {sweep_dir.name}"):
        sweep[trial_dir.name] = load_trial(trial_dir)
    cache.write_bytes(pickle.dumps(sweep))
    return sweep


def _multi_cache_path(sweep_dirs: list[Path]) -> Path:
    abs_paths = sorted(str(p.resolve()) for p in sweep_dirs)
    h = hashlib.sha256("|".join(abs_paths).encode()).hexdigest()[:16]
    if len(abs_paths) > 1:
        common = Path(os.path.commonpath(abs_paths))
        if not common.is_dir():
            common = common.parent
    else:
        common = Path(abs_paths[0]).parent
    return common / f"_multi_{h}.pkl"


def load_multi(sweep_dirs: list[Path]) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {sweep_dir.name: {trial_name: df}}, in the input order of sweep_dirs."""
    cache = _multi_cache_path(sweep_dirs)
    sweep_pkl_mts: list[float] = []
    for sd in sweep_dirs:
        sc = sd / _SWEEP_PKL
        newest_bag = max((_bag_mtime(d) for d in _trial_dirs(sd)), default=0.0)
        fresh = sc.exists() and sc.stat().st_mtime >= newest_bag
        sweep_pkl_mts.append(sc.stat().st_mtime if fresh else float("inf"))
    newest = max(sweep_pkl_mts) if sweep_pkl_mts else 0.0
    if cache.exists() and newest != float("inf") and cache.stat().st_mtime >= newest:
        try:
            nested = pd.read_pickle(cache)
        except Exception as exc:
            print(f"  unreadable cache {cache.name}: {type(exc).__name__}: {exc}; re-aggregating")
        else:
            if {sd.name for sd in sweep_dirs}.issubset(nested.keys()):
                return {sd.name: nested[sd.name] for sd in sweep_dirs}  # pyright: ignore[reportReturnType]
    nested = {sd.name: load_sweep(sd) for sd in sweep_dirs}
    cache.write_bytes(pickle.dumps(nested))
    return nested
