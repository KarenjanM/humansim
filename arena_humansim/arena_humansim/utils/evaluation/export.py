"""Export a sweep (or pooled meta-sweep) to a Croissant-compatible HF dataset tree.

Layout written under <out_dir>:

    data/
      agent_states/bucket=<b>/density=<d>/scenario=<s>/planner=<p>/part-<robot>-<seed>.parquet  # long-format trajectories, one shard per trial
      metrics/kinematics_per_trial.parquet                        # per-trial scalars
      metrics/robot_metrics.parquet                               # robots mode only
      metrics/failures.parquet                                    # robots mode only
    configs/
      scenarios/<name>.yaml                                       # snapshot per scenario
    README.md                                                     # dataset card stub
    croissant.json                                                # Croissant 1.0 metadata

The README and croissant.json are skeletons - fill in citation, license, URL,
and any task-specific recordSet fields before pushing to HuggingFace.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from arena_humansim.utils.evaluation.analyze import (
    _enumerate_unique_trials,
    _kinematics_row,
    infer_bucket,
    parse_trial_dir,
)
from arena_humansim.utils.evaluation.bag_cache import load_trial
from arena_humansim.utils.evaluation.buckets import SCENARIO_BUCKET
from arena_humansim.utils.evaluation.robots import _load_snapshot, compute_robot_metrics
from arena_humansim.utils.evaluation.robots_failures import classify_trial
from arena_humansim.utils.scenario_discovery import parse_scenario_id

CROISSANT_VERSION = "1.0"
PARQUET_ROW_GROUP = 100_000
PARQUET_COMPRESSION = "zstd"
PARTITION_COLS = ("bucket", "density", "scenario", "planner")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
        row_group_size=PARQUET_ROW_GROUP,
    )


def _shard_path(out_dir: Path, meta: dict) -> Path:
    path = out_dir
    for col in PARTITION_COLS:
        path = path / f"{col}={meta[col]}"
    return path


def _copy_scenario_snapshots(recordings_dirs: list[Path], out_dir: Path) -> list[str]:
    """Copy one scenario.yaml per unique scenario into out_dir. Returns names copied."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for recordings_dir in recordings_dirs:
        for trial_dir in sorted(recordings_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            snapshot = trial_dir / "scenario.yaml"
            if not snapshot.is_file():
                continue
            scenario = trial_dir.name.split("__", 1)[0]
            if scenario in seen:
                continue
            shutil.copy2(snapshot, out_dir / f"{scenario}.yaml")
            seen.add(scenario)
    return sorted(seen)


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_DTYPE_TO_CR = {
    "object": "sc:Text",
    "string": "sc:Text",
    "int64": "sc:Integer",
    "int32": "sc:Integer",
    "float64": "sc:Float",
    "float32": "sc:Float",
    "bool": "sc:Boolean",
}


def _fields_from_df(df: pd.DataFrame, record_id: str) -> list[dict]:
    fields = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        cr_type = _DTYPE_TO_CR.get(dtype, "sc:Text")
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"{record_id}/{col}",
                "name": col,
                "dataType": cr_type,
            }
        )
    return fields


def _build_croissant(
    dataset_name: str,
    out_dir: Path,
    agent_states_columns: list[tuple[str, str]],
    metrics: dict[str, pd.DataFrame],
) -> dict:
    distribution: list[dict] = [
        {
            "@type": "cr:FileSet",
            "@id": "agent_states_files",
            "name": "agent_states_files",
            "description": "Hive-partitioned parquet shards of long-format agent trajectories.",
            "encodingFormat": "application/vnd.apache.parquet",
            "includes": "data/agent_states/**/*.parquet",
        }
    ]
    for table_name, table_df in metrics.items():
        if table_df is None or table_df.empty:
            continue
        rel = f"data/metrics/{table_name}.parquet"
        abs_path = out_dir / rel
        if not abs_path.exists():
            continue
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": f"{table_name}_file",
                "name": f"{table_name}_file",
                "contentUrl": rel,
                "encodingFormat": "application/vnd.apache.parquet",
                "md5": _file_md5(abs_path),
            }
        )
    distribution.append(
        {
            "@type": "cr:FileSet",
            "@id": "scenario_yamls",
            "name": "scenario_yamls",
            "description": "Per-scenario YAML snapshots used at sweep time.",
            "encodingFormat": "application/yaml",
            "includes": "configs/scenarios/*.yaml",
        }
    )

    record_sets: list[dict] = [
        {
            "@type": "cr:RecordSet",
            "@id": "agent_states",
            "name": "agent_states",
            "description": "One row per (time, agent_id) step of each trial. Partition columns scenario/planner are encoded in the file path.",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": f"agent_states/{name}",
                    "name": name,
                    "dataType": cr_type,
                    "source": {
                        "fileSet": {"@id": "agent_states_files"},
                        "extract": {"column": name},
                    },
                }
                for name, cr_type in agent_states_columns
            ],
        }
    ]
    for table_name, table_df in metrics.items():
        if table_df is None or table_df.empty:
            continue
        rec_id = table_name
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": rec_id,
                "name": rec_id,
                "field": [{**f, "source": {"fileObject": {"@id": f"{table_name}_file"}, "extract": {"column": f["name"]}}} for f in _fields_from_df(table_df, rec_id)],
            }
        )

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "sc": "https://schema.org/",
            "source": "cr:source",
        },
        "@type": "sc:Dataset",
        "name": dataset_name,
        "conformsTo": f"http://mlcommons.org/croissant/{CROISSANT_VERSION}",
        "description": "TODO: dataset description",
        "license": "TODO: SPDX license id (e.g. CC-BY-4.0)",
        "citeAs": "TODO: bibtex / paper citation",
        "url": "TODO: HuggingFace dataset URL",
        "version": "1.0.0",
        "distribution": distribution,
        "recordSet": record_sets,
    }


_FIELD_DOCS: dict[str, tuple[str, str]] = {
    "time": ("s", "sim time, monotonic per trial"),
    "agent_id": ("-", "unique within trial; humans use positive ids (auto-assigned from 1), robots use negative ids (any agent_id < 0)"),
    "robot_id": ("-", "negative agent_id identifying which robot a metrics/failures row refers to; trials may have >1 robot"),
    "x": ("m", "world-frame x position"),
    "y": ("m", "world-frame y position"),
    "vx": ("m/s", "world-frame x velocity"),
    "vy": ("m/s", "world-frame y velocity"),
    "radius": ("m", "agent collision radius"),
    "source_name": ("-", "flow source that released the agent, empty for scripted agents, robots and recordings without attribution"),
    "source_seq": ("-", "release index of the agent within source_name, 0 when source_name is empty. (source_name, source_seq) pairs a released agent across runs and seeds"),
    "scenario": ("-", "canonical scenario id (Hive partition column)"),
    "planner": ("-", "pedestrian motion model (Hive partition column)"),
    "robot_policy": ("-", "robot policy name; empty string in divergence-mode trials"),
    "seed": ("-", "RNG seed for the trial"),
    "bucket": ("-", "scenario family: nav / bt / het (Hive partition column)"),
    "density": ("-", "pedestrian density regime: sparse / dense (Hive partition column)"),
    "modality": ("-", "scenario modality within (bucket, density), e.g. corridor, group_conversation"),
    "is_robot_scenario": ("bool", "true if scenario id is a robot variant of a pure-ped sibling"),
    "source_dir": ("-", "originating sweep dir name (provenance)"),
    "jerk": ("m/s^3", "mean per-agent jerk over the trial"),
    "curvature": ("1/m", "mean per-agent path curvature over the trial"),
    "collisions": ("count", "agent-agent collision events in the trial"),
    "success": ("0/1", "robot reached its goal within tolerance"),
    "time_to_goal_s": ("s", "robot wall-time to first goal arrival; NaN if no arrival"),
    "ttg_ratio": ("ratio", "time-to-goal divided by straight-line nominal time"),
    "path_efficiency": ("ratio", "robot arc-length / start-to-goal straight-line distance"),
    "n_robot_collisions": ("count", "frames in which robot overlapped any human"),
    "personal_space_violations": ("count", "frames in which a human entered the robot's personal-space disk"),
    "psv_per_sec": ("1/s", "personal_space_violations normalized by trial duration"),
    "frozen_at_end": ("0/1", "robot mean speed in last freeze-window seconds is below threshold"),
    "final_goal_dist": ("m", "robot-goal distance at trial end"),
    "cause": ("-", "failure-cause label (success / collision / timeout_no_progress / ...)"),
    "ped_planner": ("-", "pedestrian planner used; alias of `planner` in metrics tables"),
}


def _field_table(columns: list[str]) -> str:
    rows = ["| field | dtype | unit | description |", "|---|---|---|---|"]
    for col in columns:
        unit, desc = _FIELD_DOCS.get(col, ("-", "TODO"))
        rows.append(f"| `{col}` | TODO | {unit} | {desc} |")
    return "\n".join(rows)


def _write_readme(
    path: Path,
    dataset_name: str,
    *,
    n_trials: int,
    scenarios: list[str],
    planners: list[str],
    full_columns: list[str],
    robot_mode: bool,
) -> None:
    body = f"""---
license: TODO
pretty_name: {dataset_name}
task_categories:
- robotics
tags:
- pedestrian-simulation
- trajectory
- arena-rosnav
size_categories:
- 1M<n<10M
source_datasets:
- original
configs:
- config_name: agent_states
  data_files: data/agent_states/**/*.parquet
- config_name: kinematics
  data_files: data/metrics/kinematics_per_trial.parquet
---

# {dataset_name}

Pedestrian and robot trajectories from the arena_humansim simulator.

## Stats

- Trials: {n_trials}
- Scenarios: {", ".join(scenarios)}
- Pedestrian planners: {", ".join(planners)}
- Mode: {"robots (heterogeneous human+robot)" if robot_mode else "divergence (humans only)"}

## Layout

- `data/agent_states/bucket=<b>/density=<d>/scenario=<s>/planner=<p>/*.parquet` - long-format trajectories, ~{PARQUET_ROW_GROUP:_} rows per row-group, zstd compressed. Partition values for `bucket`, `density`, `scenario` and `planner` are encoded in the path; mirrors the `config/evaluation/<bucket>/<density>/<modality>.yaml` input tree.
- `data/metrics/kinematics_per_trial.parquet` - per-trial scalar metrics (jerk, curvature, collisions).
- `data/metrics/robot_metrics.parquet` - per-trial robot KPIs (success, time-to-goal, path efficiency, personal-space violations). Robots-mode only.
- `data/metrics/failures.parquet` - per-trial failure cause classification. Robots-mode only.
- `configs/scenarios/*.yaml` - scenario YAML snapshots used at sweep time.
- `croissant.json` - Croissant {CROISSANT_VERSION} machine-readable schema.
- `DATASHEET.md` - Datasheet for Datasets (Gebru et al.) covering motivation, composition, collection, uses, distribution, maintenance.

## Schema

### `agent_states` (long-format trajectories)

{_field_table(full_columns)}

### `kinematics_per_trial`

{_field_table(["bucket", "scenario", "planner", "robot_policy", "seed", "jerk", "curvature", "collisions"])}
{_robot_schema_block() if robot_mode else ""}
## Quick start

```python
from datasets import load_dataset

# trajectories - stream a single (bucket, density, scenario, planner) partition
ds = load_dataset("TODO/repo_id", "agent_states", split="train", streaming=True,
                  data_files="data/agent_states/bucket=nav/density=sparse/scenario=nav_sparse_corridor/planner=sfm/*.parquet")

# or all of one bucket (partition pushdown)
ds_nav = load_dataset("TODO/repo_id", "agent_states", split="train", streaming=True,
                      data_files="data/agent_states/bucket=nav/**/*.parquet")

# per-trial scalar metrics
import pandas as pd
kin = pd.read_parquet("hf://datasets/TODO/repo_id/data/metrics/kinematics_per_trial.parquet")
print(kin.groupby("planner")[["jerk", "curvature", "collisions"]].mean())
```

## Citation

TODO bibtex

## License

TODO - fill in `LICENSE` file at repo root and the `license` field above.
"""
    path.write_text(body)


def _robot_schema_block() -> str:
    cols = ["scenario", "bucket", "ped_planner", "robot_policy", "seed", "success", "time_to_goal_s", "ttg_ratio", "path_efficiency", "n_robot_collisions", "personal_space_violations", "psv_per_sec", "frozen_at_end", "final_goal_dist"]
    return f"\n### `robot_metrics`\n\n{_field_table(cols)}\n\n### `failures`\n\n{_field_table(['scenario', 'bucket', 'ped_planner', 'robot_policy', 'seed', 'cause'])}\n"


_DATASHEET_TEMPLATE = """# Datasheet for {dataset_name}

Following the Datasheet for Datasets template (Gebru et al., 2018).

## Motivation

- **For what purpose was the dataset created?** TODO
- **Who created the dataset and on behalf of which entity?** TODO
- **Who funded the creation of the dataset?** TODO

## Composition

- **What do the instances represent?** Per-step state of every simulated agent (pedestrian or robot) in a sweep of `arena_humansim` simulator trials.
- **How many instances are there in total?** {n_trials} trials x variable agent count x variable trial length = {n_rows} agent-state rows.
- **Does the dataset contain all possible instances or is it a sample?** A sample of the parameter sweep grid: {n_scenarios} scenarios x {n_planners} pedestrian planners x N seeds (and M robot policies in robots-mode trials).
- **What data does each instance consist of?** See README.md schema tables.
- **Is there a label or target?** No supervised labels; per-trial scalar metrics (kinematics, robot KPIs, failure cause) serve as evaluation targets.
- **Is any information missing?** `robot_policy` is the empty string in divergence-mode trials.
- **Are relationships between instances made explicit?** Yes - `(scenario, planner, robot_policy, seed)` jointly key a trial.
- **Are there recommended data splits?** No fixed splits. Hold out by `seed` for trial-level CV; hold out by `scenario` for OOD generalization.
- **Are there errors, sources of noise, or redundancies?** Trials with malformed/truncated rosbags are silently dropped during extraction (cached as empty).
- **Is the dataset self-contained?** Yes; scenario YAMLs included under `configs/`.
- **Does the dataset contain confidential, offensive, or personally identifiable data?** No. Fully simulated.

## Collection process

- **How was the data acquired?** Generated by running `arena_humansim` (ROS 2 Jazzy, Python 3.12) with sweep configurations under `configs/scenarios/`. See the `arena_humansim` repository for the exact simulator version.
- **What mechanisms or procedures were used to collect the data?** `evaluate sweep` ran each (scenario, planner, robot_policy, seed) combination; trajectories were recorded as rosbags and post-processed with `evaluate export`.
- **Over what timeframe was the data collected?** TODO
- **Were any ethical review processes conducted?** N/A - simulated data.

## Preprocessing / cleaning / labeling

- Bag messages for `agent_states` topic deserialized into long-format dataframes.
- Per-trial scalar metrics computed offline (kinematics, collisions, robot KPIs, failure cause).
- Trials with empty / unreadable bags skipped.

## Uses

- **What tasks could the dataset be used for?** Pedestrian-aware motion planning evaluation; robot navigation in crowds; trajectory prediction; planner ablations; sim-to-real transfer baselines.
- **Are there tasks for which the dataset should NOT be used?** Anything assuming real-human behavioral fidelity - these are model-driven simulants (SFM, HSFM, ORCA, NSP, SocialGAIL) with known biases.
- **Is there a repository linking to papers / systems that use this dataset?** TODO

## Distribution

- **How will the dataset be distributed?** HuggingFace Datasets; mirrored on Zenodo for DOI.
- **License?** TODO (recommend CC-BY-4.0).
- **Subject to copyright / IP restrictions?** Generated by an open-source simulator under MIT - no upstream restrictions.

## Maintenance

- **Who is supporting / hosting / maintaining?** TODO
- **How can the dataset owner be contacted?** TODO (HF discussions tab + GitHub issues)
- **Will the dataset be updated?** Versioned via `version` in `croissant.json`; changes documented in `CHANGELOG.md`.
- **Are older versions available?** Yes - via HF dataset revision history and Zenodo DOI versioning.
"""


def _write_datasheet(
    path: Path,
    dataset_name: str,
    *,
    n_trials: int,
    n_rows: int,
    n_scenarios: int,
    n_planners: int,
) -> None:
    path.write_text(
        _DATASHEET_TEMPLATE.format(
            dataset_name=dataset_name,
            n_trials=n_trials,
            n_rows=n_rows,
            n_scenarios=n_scenarios,
            n_planners=n_planners,
        )
    )


def _process_trial(recordings_dir: Path, trial_name: str, states_dir: Path, do_robots: bool) -> dict[str, Any] | None:
    """Worker entry point: load + annotate + write one trial's parquet shard, return per-trial scalars.
    Returns None for empty bags. Safe to pickle for ProcessPoolExecutor."""
    trial_dir = recordings_dir / trial_name
    scenario, planner, robot_policy, seed = parse_trial_dir(trial_name)

    df = load_trial(trial_dir)
    if df.empty:
        return None

    snapshot = trial_dir / "scenario.yaml"
    if snapshot.exists():
        try:
            bucket = infer_bucket(yaml.safe_load(snapshot.read_text()) or {}, scenario)
        except Exception:
            bucket = "unknown"
    else:
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

    meta = {
        "scenario": scenario,
        "planner": planner,
        "robot_policy": robot_policy,
        "seed": seed,
        "bucket": bucket,
        "density": density,
        "modality": modality,
        "is_robot_scenario": is_robot_scenario,
        "source_dir": recordings_dir.name,
    }

    path = _shard_path(states_dir, meta)
    path.mkdir(parents=True, exist_ok=True)
    body = df.drop(columns=list(PARTITION_COLS))
    rp = robot_policy or "none"
    _write_parquet(body, path / f"part-{rp}-{int(seed):05d}.parquet")

    result: dict[str, Any] = {
        "meta": meta,
        "n_rows": len(body),
        "full_columns": list(df.columns),
        "full_dtypes": {c: str(df[c].dtype) for c in df.columns},
        "kin_row": _kinematics_row(df, meta),
    }
    if do_robots and robot_policy:
        canonical, goals = _load_snapshot(trial_dir)
        robot_bucket = SCENARIO_BUCKET.get(canonical or scenario, "unknown")
        trial_id = {
            "scenario": scenario,
            "bucket": robot_bucket,
            "ped_planner": planner,
            "robot_policy": robot_policy,
            "seed": seed,
            "source_dir": recordings_dir.name,
        }
        result["rm_rows"] = [{**trial_id, "robot_id": rid, **metrics} for rid, metrics in compute_robot_metrics(df, goals).items()]
        result["fl_rows"] = [{**trial_id, "robot_id": rid, "cause": cause} for rid, cause in classify_trial(df, goals).items()]
    return result


def run_export(
    recordings_dirs: Path | list[Path],
    out_dir: Path,
    dataset_name: str,
    mode: str = "auto",
    workers: int | None = None,
) -> dict:
    if workers is None:
        workers = os.cpu_count() or 1
    if isinstance(recordings_dirs, Path):
        recordings_dirs = [recordings_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)

    states_dir = out_dir / "data" / "agent_states"
    metrics_dir = out_dir / "data" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    kin_rows: list[dict] = []
    rm_rows: list[dict] = []
    fl_rows: list[dict] = []
    scenarios_seen: set[str] = set()
    planners_seen: set[str] = set()
    n_trials = 0
    n_parts = 0
    n_rows_total = 0
    has_robot_policy = False
    full_columns: list[str] = []
    full_dtypes: dict[str, str] = {}

    force_robots = mode == "robots"
    do_robots = force_robots or mode == "auto"

    tasks = _enumerate_unique_trials(recordings_dirs)
    total_trials = len(tasks)
    print(f"Streaming {total_trials} trials from {len(recordings_dirs)} dir(s) with {workers} worker(s)...")
    bar = tqdm(total=total_trials, unit="trial")

    def _consume(result: dict | None) -> None:
        nonlocal n_trials, n_parts, n_rows_total, has_robot_policy, full_columns, full_dtypes
        bar.update(1)
        if result is None:
            return
        meta = result["meta"]
        if not full_columns:
            full_columns = result["full_columns"]
            full_dtypes = result["full_dtypes"]
        kin_rows.append(result["kin_row"])
        if "rm_rows" in result:
            rm_rows.extend(result["rm_rows"])
            has_robot_policy = True
        if "fl_rows" in result:
            fl_rows.extend(result["fl_rows"])
        scenarios_seen.add(meta["scenario"])
        planners_seen.add(meta["planner"])
        n_trials += 1
        n_parts += 1
        n_rows_total += result["n_rows"]
        bar.set_description_str(f"{meta['scenario']}/{meta['planner']}")

    if workers <= 1:
        for recordings_dir, trial_name in tasks:
            _consume(_process_trial(recordings_dir, trial_name, states_dir, do_robots))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_trial, rd, tn, states_dir, do_robots) for rd, tn in tasks]
            for fut in as_completed(futures):
                _consume(fut.result())

    bar.close()

    if n_trials == 0:
        raise RuntimeError(f"no usable trials under {[str(d) for d in recordings_dirs]}")

    print(f"  streamed {n_trials} trials | {n_rows_total} rows | wrote {n_parts} parquet shards under {states_dir}")

    metrics: dict[str, pd.DataFrame] = {}
    kin = pd.DataFrame(kin_rows)
    _write_parquet(kin, metrics_dir / "kinematics_per_trial.parquet")
    metrics["kinematics_per_trial"] = kin

    robot_mode = force_robots or (mode == "auto" and has_robot_policy)
    if robot_mode and rm_rows:
        rm = pd.DataFrame(rm_rows)
        _write_parquet(rm, metrics_dir / "robot_metrics.parquet")
        metrics["robot_metrics"] = rm
    if robot_mode and fl_rows:
        fl = pd.DataFrame(fl_rows)
        _write_parquet(fl, metrics_dir / "failures.parquet")
        metrics["failures"] = fl

    print("Copying scenario YAML snapshots...")
    scenario_dir = out_dir / "configs" / "scenarios"
    scenarios = _copy_scenario_snapshots(recordings_dirs, scenario_dir)
    print(f"  copied {len(scenarios)} scenario snapshot(s) to {scenario_dir}")

    print("Writing README + DATASHEET + Croissant...")
    scenarios_sorted = sorted(scenarios_seen)
    planners_sorted = sorted(planners_seen)
    _write_readme(
        out_dir / "README.md",
        dataset_name,
        n_trials=n_trials,
        scenarios=scenarios_sorted,
        planners=planners_sorted,
        full_columns=full_columns,
        robot_mode=robot_mode,
    )
    _write_datasheet(
        out_dir / "DATASHEET.md",
        dataset_name,
        n_trials=n_trials,
        n_rows=n_rows_total,
        n_scenarios=len(scenarios_sorted),
        n_planners=len(planners_sorted),
    )
    agent_state_cols = [(c, _DTYPE_TO_CR.get(full_dtypes[c], "sc:Text")) for c in full_columns if c not in PARTITION_COLS]
    croissant = _build_croissant(dataset_name, out_dir, agent_state_cols, metrics)
    (out_dir / "croissant.json").write_text(json.dumps(croissant, indent=2))

    print(f"Done -> {out_dir}")
    return {
        "out_dir": out_dir,
        "n_trials": n_trials,
        "n_parquet_shards": n_parts,
        "metrics_tables": list(metrics.keys()),
        "scenarios": scenarios,
    }
