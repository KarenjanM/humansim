from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import pandas as pd
from arena_humansim.utils.evaluation.robots import _load_snapshot, load_robots_recordings, run_robots_analysis

from tests.unit._bags import NESTED, Agent, write_bag

_CROSSING = "robot_nav_sparse_crossing__sfm__straight__1"
_SNAPSHOT_WITHOUT_GOALS = "name: robot_nav_sparse_crossing\nagents:\n  - {agent_id: -1, kind: 1}\n  - {agent_id: -9, kind: 1}\n"


def _record(trial: Path, frames: list[tuple[int, list[Agent]]], snapshot: str | None = None) -> None:
    trial.mkdir(parents=True)
    if snapshot is not None:
        (trial / "scenario.yaml").write_text(snapshot)
    write_bag(trial / "bag", NESTED, frames)


def _crossing_frames(x0: float) -> list[tuple[int, list[Agent]]]:
    return [(i * 1_000_000_000, [Agent(-1, x0 + i, 0.0, vx=1.0, radius=0.3), Agent(-9, 0.0, 30.0, radius=0.3), Agent(4, x0 + i, 0.6, radius=0.25)]) for i in range(3)]


def _sweeps(root: Path) -> tuple[Path, Path]:
    a = root / "sweep_a"
    _record(a / _CROSSING, _crossing_frames(11.0), _SNAPSHOT_WITHOUT_GOALS)
    _record(a / "het_dense_mixed_speeds__orca__2", [(i * 1_000_000_000, [Agent(-1, 0.0, 0.0), Agent(2, 5.0, 5.0)]) for i in range(2)])
    _record(a / "custom_thing__sfm__3", [(0, [Agent(-3, 1.0, 1.0)]), (1_000_000_000, [Agent(-3, 1.0, 2.0)])])
    _record(a / "nav_sparse_corridor__sfm__5", [(0, [Agent(1, 0.0, 0.0)])])
    _record(a / "not_a_trial", [(0, [Agent(-1, 0.0, 0.0)])])
    broken = a / "broken__sfm__4" / "bag"
    broken.mkdir(parents=True)
    (broken / "metadata.yaml").write_text("not a bag\n")
    b = root / "sweep_b"
    _record(b / _CROSSING, _crossing_frames(-5.0), _SNAPSHOT_WITHOUT_GOALS)
    _record(b / "custom_thing__sfm__straight__3", [(0, [Agent(-3, 0.0, 0.0)])])
    return a, b


def test_snapshot_fills_missing_robot_goal_from_eval_tree(tmp_path: Path) -> None:
    trial = tmp_path / _CROSSING
    trial.mkdir()
    (trial / "scenario.yaml").write_text(_SNAPSHOT_WITHOUT_GOALS)
    assert _load_snapshot(trial) == ("robot_nav_sparse_crossing", {-1: (13.0, 0.0), -9: None})


def test_snapshot_goal_wins_over_eval_tree(tmp_path: Path) -> None:
    (tmp_path / "scenario.yaml").write_text("name: robot_nav_sparse_crossing\nagents:\n  - {agent_id: -1, kind: 1, goal_sequence: [{x: 2.0, y: 1.0}]}\n")
    assert _load_snapshot(tmp_path) == ("robot_nav_sparse_crossing", {-1: (2.0, 1.0)})


def test_snapshot_missing_gives_no_name_and_no_goals(tmp_path: Path) -> None:
    assert _load_snapshot(tmp_path) == (None, {})


def test_load_robots_recordings_one_row_per_robot_first_sweep_wins(tmp_path: Path) -> None:
    a, b = _sweeps(tmp_path)
    df = load_robots_recordings([a, b])
    keyed = df.set_index(["source_dir", "scenario", "robot_id"])
    assert sorted(keyed.index) == [
        ("sweep_a", "custom_thing", -3),
        ("sweep_a", "het_dense_mixed_speeds", -1),
        ("sweep_a", "robot_nav_sparse_crossing", -9),
        ("sweep_a", "robot_nav_sparse_crossing", -1),
        ("sweep_b", "custom_thing", -3),
    ]
    crossing = keyed.loc[("sweep_a", "robot_nav_sparse_crossing", -1)]
    assert crossing["bucket"] == "nav"
    assert crossing["ped_planner"] == "sfm"
    assert crossing["robot_policy"] == "straight"
    assert crossing["seed"] == 1
    assert crossing["success"] == 1.0
    assert crossing["time_to_goal_s"] == pytest.approx(2.0)
    assert crossing["ttg_ratio"] == pytest.approx(1.1)
    assert crossing["path_efficiency"] == pytest.approx(1.0)
    assert crossing["n_robot_collisions"] == 0.0
    assert crossing["personal_space_violations"] == 3.0
    assert crossing["psv_per_sec"] == pytest.approx(1.5)
    assert math.isnan(keyed.loc[("sweep_a", "robot_nav_sparse_crossing", -9)]["success"])
    het = keyed.loc[("sweep_a", "het_dense_mixed_speeds", -1)]
    assert het["bucket"] == "het"
    assert het["robot_policy"] == ""
    assert math.isnan(het["success"])
    assert het["frozen_at_end"] == 1.0
    assert keyed.loc[("sweep_a", "custom_thing", -3)]["bucket"] == "unknown"
    assert keyed.loc[("sweep_b", "custom_thing", -3)]["robot_policy"] == "straight"


def test_load_robots_recordings_accepts_single_dir(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    _record(sweep / _CROSSING, _crossing_frames(11.0), _SNAPSHOT_WITHOUT_GOALS)
    df = load_robots_recordings(sweep)
    assert sorted(df["robot_id"]) == [-9, -1]
    assert set(df["source_dir"]) == {"sweep"}


def test_run_robots_analysis_writes_four_tables(tmp_path: Path) -> None:
    a, b = _sweeps(tmp_path)
    out = tmp_path / "out"
    result = run_robots_analysis([a, b], out, n_bootstrap=20)
    assert set(result) == {"per_trial", "cell", "by_policy", "by_policy_bucket"}
    for name in ("robots_per_trial.csv", "robots_cell.csv", "robots_by_policy.csv", "robots_by_policy_bucket.csv"):
        assert (out / name).is_file()
    by_policy = pd.read_csv(out / "robots_by_policy.csv", keep_default_na=False).set_index("robot_policy")
    assert by_policy.loc["straight", "n_trials"] == 3
    assert by_policy.loc["", "n_trials"] == 2
    assert float(by_policy.loc["straight", "success_mean"]) == pytest.approx(1.0)
    assert len(pd.read_csv(out / "robots_per_trial.csv")) == 5


def test_run_robots_analysis_rejects_sweep_without_robot_trials(tmp_path: Path) -> None:
    sweep = tmp_path / "humans_only"
    _record(sweep / "nav_sparse_corridor__sfm__1", [(0, [Agent(1, 0.0, 0.0)])])
    with pytest.raises(RuntimeError, match="no usable robots trials"):
        run_robots_analysis(sweep, tmp_path / "out", n_bootstrap=5)
