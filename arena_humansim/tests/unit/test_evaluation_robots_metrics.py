from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arena_humansim.utils.evaluation.robots import (
    _bootstrap_mean_ci,
    _compute_one_robot_metrics,
    _name_and_robot_goals,
    _trajectory_arc_length,
    aggregate_robots,
    assert_run_mode,
    compute_robot_metrics,
    parse_robots_trial_dir,
    robot_ids_in,
)


def _rows(agent_id: int, samples: list[tuple[float, float, float, float, float]], radius: float) -> list[dict[str, float]]:
    return [{"time": t, "agent_id": agent_id, "x": x, "y": y, "vx": vx, "vy": vy, "radius": radius} for t, x, y, vx, vy in samples]


def _encounter_trial() -> pd.DataFrame:
    robot = _rows(-1, [(0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 3.0, 4.0, 1.0, 0.0), (2.0, 6.0, 0.0, 0.0, 0.0), (3.0, 6.0, 0.0, 0.0, 0.0)], 0.3)
    human = _rows(1, [(0.0, 0.5, 0.0, 0.0, 0.0), (1.0, 3.0, 4.7, 0.0, 0.0), (2.0, 20.0, 20.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0)], 0.25)
    return pd.DataFrame(human + robot)


def _write_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_args.json").write_text(json.dumps(manifest))


def test_assert_run_mode_rejects_dir_without_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no run_args.json"):
        assert_run_mode(tmp_path, "robots")


def test_assert_run_mode_rejects_other_mode(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"mode": "humans"})
    with pytest.raises(RuntimeError, match="mode='humans', expected 'robots'"):
        assert_run_mode(tmp_path, "robots")


def test_assert_run_mode_requires_cop_flag_when_asked(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"mode": "robots"})
    assert assert_run_mode(tmp_path, "robots") == {"mode": "robots"}
    with pytest.raises(RuntimeError, match="cop=false"):
        assert_run_mode(tmp_path, "robots", require_cop=True)


def test_assert_run_mode_returns_cop_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"mode": "robots", "cop": True, "seeds": 3})
    assert assert_run_mode(tmp_path, "robots", require_cop=True) == {"mode": "robots", "cop": True, "seeds": 3}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("robot_nav_sparse_corridor__sfm__cadrl__7", ("robot_nav_sparse_corridor", "sfm", "cadrl", 7)),
        ("nav_sparse_corridor__orca__12", ("nav_sparse_corridor", "orca", "", 12)),
        ("nav_sparse_corridor__orca", None),
        ("a__b__c__d__1", None),
        ("nav_sparse_corridor__orca__seven", None),
    ],
)
def test_parse_robots_trial_dir_splits_or_rejects(name: str, expected: tuple[str, str, str, int] | None) -> None:
    assert parse_robots_trial_dir(name) == expected


def test_name_and_robot_goals_takes_last_goal_of_each_robot(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("name: robot_goals\nagents:\n  - {agent_id: 1, goal_sequence: [{x: 9.0, y: 9.0}]}\n  - {agent_id: -1, kind: 1, goal_sequence: [{x: 1.0, y: 2.0}, {x: 3.5, y: -4.0}]}\n  - {agent_id: -2, kind: 1}\n")
    name, goals = _name_and_robot_goals(path)
    assert name == "robot_goals"
    assert goals == {-1: (3.5, -4.0), -2: None}


def test_name_and_robot_goals_returns_empty_for_unloadable_yaml(tmp_path: Path) -> None:
    assert _name_and_robot_goals(tmp_path / "missing.yaml") == (None, {})


def test_name_and_robot_goals_maps_empty_name_to_none(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("name: ''\nagents: []\n")
    assert _name_and_robot_goals(path) == (None, {})


def test_trajectory_arc_length_sums_segment_lengths() -> None:
    assert _trajectory_arc_length(np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 0.0], [6.0, 0.0]])) == pytest.approx(10.0)
    assert _trajectory_arc_length(np.array([[1.0, 1.0]])) == 0.0
    assert _trajectory_arc_length(np.zeros((0, 2))) == 0.0


def test_robot_metrics_on_detour_with_one_contact_and_two_intrusions() -> None:
    m = _compute_one_robot_metrics(_encounter_trial(), -1, (6.0, 0.0))
    assert m["success"] == 1.0
    assert m["time_to_goal_s"] == pytest.approx(2.0)
    assert m["ttg_ratio"] == pytest.approx(2.0 / (6.0 / 1.1))
    assert m["path_efficiency"] == pytest.approx(10.0 / 6.0)
    assert m["n_robot_collisions"] == 1.0
    assert m["personal_space_violations"] == 2.0
    assert m["psv_per_sec"] == pytest.approx(2.0 / 3.0)
    assert m["frozen_at_end"] == 0.0
    assert m["final_goal_dist"] == 0.0


def test_robot_metrics_goal_radius_is_inclusive_at_half_a_meter() -> None:
    df = pd.DataFrame(_rows(-1, [(0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 0.0, 1.0, 0.0), (2.0, 1.5, 0.0, 1.0, 0.0)], 0.3))
    m = _compute_one_robot_metrics(df, -1, (2.0, 0.0))
    assert m["success"] == 1.0
    assert m["time_to_goal_s"] == pytest.approx(2.0)
    assert m["final_goal_dist"] == pytest.approx(0.5)


def test_robot_metrics_without_goal_keep_contact_counts() -> None:
    m = _compute_one_robot_metrics(_encounter_trial(), -1, None)
    for key in ("success", "time_to_goal_s", "ttg_ratio", "path_efficiency", "final_goal_dist"):
        assert math.isnan(m[key])
    assert m["n_robot_collisions"] == 1.0
    assert m["personal_space_violations"] == 2.0
    assert m["psv_per_sec"] == pytest.approx(2.0 / 3.0)


def test_robot_metrics_unreached_goal_reports_distance_and_nan_time() -> None:
    df = pd.DataFrame(_rows(-1, [(0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 0.0, 1.0, 0.0), (2.0, 2.0, 0.0, 1.0, 0.0)], 0.3))
    m = _compute_one_robot_metrics(df, -1, (10.0, 0.0))
    assert m["success"] == 0.0
    assert math.isnan(m["time_to_goal_s"])
    assert math.isnan(m["ttg_ratio"])
    assert m["path_efficiency"] == pytest.approx(0.2)
    assert m["final_goal_dist"] == pytest.approx(8.0)
    assert m["n_robot_collisions"] == 0.0
    assert m["personal_space_violations"] == 0.0
    assert m["psv_per_sec"] == 0.0


def test_robot_metrics_start_on_goal_has_no_ratio_or_efficiency() -> None:
    df = pd.DataFrame(_rows(-1, [(0.0, 2.0, 2.0, 0.0, 0.0), (1.0, 2.0, 2.0, 0.0, 0.0)], 0.3))
    m = _compute_one_robot_metrics(df, -1, (2.0, 2.0))
    assert m["success"] == 1.0
    assert m["time_to_goal_s"] == 0.0
    assert math.isnan(m["ttg_ratio"])
    assert math.isnan(m["path_efficiency"])
    assert m["frozen_at_end"] == 1.0


def test_robot_metrics_single_sample_has_no_rate() -> None:
    df = pd.DataFrame(_rows(-1, [(5.0, 0.0, 0.0, 0.0, 0.0)], 0.3) + _rows(2, [(5.0, 0.2, 0.0, 0.0, 0.0)], 0.25))
    m = _compute_one_robot_metrics(df, -1, (3.0, 4.0))
    assert m["n_robot_collisions"] == 1.0
    assert m["personal_space_violations"] == 1.0
    assert math.isnan(m["psv_per_sec"])
    assert m["path_efficiency"] == 0.0
    assert m["final_goal_dist"] == pytest.approx(5.0)


def test_robot_metrics_freeze_window_includes_its_left_edge() -> None:
    df = pd.DataFrame(_rows(-1, [(0.0, 0.0, 0.0, 1.0, 0.0), (5.0, 1.0, 0.0, 0.12, 0.0), (10.0, 1.0, 0.0, 0.0, 0.0)], 0.3))
    assert _compute_one_robot_metrics(df, -1, None)["frozen_at_end"] == 0.0
    df = pd.DataFrame(_rows(-1, [(0.0, 0.0, 0.0, 1.0, 0.0), (4.9, 1.0, 0.0, 0.12, 0.0), (10.0, 1.0, 0.0, 0.0, 0.0)], 0.3))
    assert _compute_one_robot_metrics(df, -1, None)["frozen_at_end"] == 1.0


def test_robot_metrics_nan_velocity_leaves_freeze_undefined() -> None:
    df = pd.DataFrame(_rows(-1, [(0.0, 0.0, 0.0, float("nan"), 0.0), (1.0, 1.0, 0.0, float("nan"), 0.0)], 0.3))
    assert math.isnan(_compute_one_robot_metrics(df, -1, None)["frozen_at_end"])


def test_robot_metrics_for_absent_robot_are_all_nan() -> None:
    m = _compute_one_robot_metrics(_encounter_trial(), -5, (1.0, 1.0))
    assert set(m) == {"success", "time_to_goal_s", "ttg_ratio", "path_efficiency", "n_robot_collisions", "personal_space_violations", "psv_per_sec", "frozen_at_end", "final_goal_dist"}
    assert all(math.isnan(v) for v in m.values())


def test_robot_ids_in_lists_negative_ids_sorted() -> None:
    df = pd.DataFrame({"agent_id": [3, -1, -4, 0, -1, 7, -2]})
    assert robot_ids_in(df) == [-4, -2, -1]


def test_compute_robot_metrics_looks_up_goal_per_robot() -> None:
    other = pd.DataFrame(_rows(-2, [(0.0, 50.0, 50.0, 0.0, 0.0), (1.0, 50.0, 50.0, 0.0, 0.0)], 0.3))
    df = pd.concat([_encounter_trial(), other], ignore_index=True)
    out = compute_robot_metrics(df, {-1: (6.0, 0.0)})
    assert sorted(out) == [-2, -1]
    assert out[-1]["success"] == 1.0
    assert math.isnan(out[-2]["success"])
    assert out[-2]["n_robot_collisions"] == 0.0
    no_goals = compute_robot_metrics(df)
    assert math.isnan(no_goals[-1]["success"])
    assert no_goals[-1]["n_robot_collisions"] == 1.0


def test_bootstrap_mean_ci_of_constant_is_degenerate() -> None:
    lo, hi = _bootstrap_mean_ci(np.array([2.0, float("nan"), 2.0]), 50, np.random.default_rng(0))
    assert (lo, hi) == (2.0, 2.0)


def test_bootstrap_mean_ci_of_all_nan_is_nan() -> None:
    lo, hi = _bootstrap_mean_ci(np.array([float("nan")]), 50, np.random.default_rng(0))
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_mean_ci_brackets_sample_mean_within_range() -> None:
    lo, hi = _bootstrap_mean_ci(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), 2000, np.random.default_rng(0))
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def _per_trial(policy: str, bucket: object, success: float, ttg: float) -> dict[str, object]:
    return {
        "robot_policy": policy,
        "bucket": bucket,
        "ped_planner": "sfm",
        "success": success,
        "ttg_ratio": ttg,
        "path_efficiency": 1.0,
        "n_robot_collisions": 0.0,
        "psv_per_sec": 0.5,
        "frozen_at_end": float("nan"),
        "final_goal_dist": 0.0,
    }


def test_aggregate_robots_means_skip_nan_and_ci_brackets_mean() -> None:
    df = pd.DataFrame(
        [
            _per_trial("cadrl", "nav", 1.0, 2.0),
            _per_trial("cadrl", "nav", 0.0, float("nan")),
            _per_trial("cadrl", "nav", 1.0, 4.0),
            _per_trial("sarl", "nav", 1.0, 1.5),
            _per_trial("sarl", float("nan"), 0.0, 3.0),
        ]
    )
    out = aggregate_robots(df, group_by=["robot_policy", "bucket"], n_bootstrap=200).set_index(["robot_policy", "bucket"])
    assert len(out) == 3
    cadrl = out.loc[("cadrl", "nav")]
    assert cadrl["n_trials"] == 3
    assert cadrl["success_mean"] == pytest.approx(2.0 / 3.0)
    assert cadrl["ttg_ratio_mean"] == pytest.approx(3.0)
    assert 2.0 <= cadrl["ttg_ratio_ci_lo"] <= 3.0 <= cadrl["ttg_ratio_ci_hi"] <= 4.0
    assert cadrl["psv_per_sec_ci_lo"] == cadrl["psv_per_sec_ci_hi"] == 0.5
    assert math.isnan(cadrl["frozen_at_end_mean"])
    assert math.isnan(cadrl["frozen_at_end_ci_lo"])
    assert out.loc[("sarl", "nav")]["ttg_ratio_mean"] == pytest.approx(1.5)
    unbucketed = out.loc[[k for k in out.index if k[0] == "sarl" and not isinstance(k[1], str)]]
    assert unbucketed["n_trials"].tolist() == [1]
    assert unbucketed["ttg_ratio_mean"].tolist() == [3.0]


def test_aggregate_robots_single_key_rows_name_the_key() -> None:
    df = pd.DataFrame([_per_trial("cadrl", "nav", 1.0, 2.0), _per_trial("sarl", "bt", 0.0, 1.0)])
    out = aggregate_robots(df, group_by=["robot_policy"], n_bootstrap=10)
    assert out["robot_policy"].tolist() == ["cadrl", "sarl"]
    assert out["success_mean"].tolist() == [1.0, 0.0]
    assert out["n_trials"].tolist() == [1, 1]


def test_aggregate_robots_ci_is_reproducible_by_seed() -> None:
    df = pd.DataFrame([_per_trial("cadrl", "nav", float(i % 2), float(i)) for i in range(10)])
    a = aggregate_robots(df, group_by=["robot_policy"], n_bootstrap=100, ci_seed=3)
    b = aggregate_robots(df, group_by=["robot_policy"], n_bootstrap=100, ci_seed=3)
    pd.testing.assert_frame_equal(a, b)
