from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arena_humansim.utils.evaluation.correspondence import (
    A_REP,
    ANISOTROPY,
    B_REP,
    DT,
    HORIZON_S,
    RADIUS,
    SCENARIOS,
    TAU,
    V_MAX,
    V_PREF,
    _circle,
    _corridor_pass,
    _crossing,
    _params,
    _perturbed,
    _pref_velocity,
    _sparse_random,
    compare,
    run_correspondence,
    run_pysocialforce,
    run_rvo2,
    run_straight_reference,
)

_STEPS = int(HORIZON_S / DT)


def test_circle_places_antipodal_goals_on_radius() -> None:
    starts, goals = _circle(4, 2.0)
    np.testing.assert_allclose(starts, [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]], atol=1e-12)
    np.testing.assert_allclose(goals, -starts)


def test_corridor_pass_swaps_sides_with_mirrored_lanes() -> None:
    starts, goals = _corridor_pass(3)
    np.testing.assert_allclose(starts, [[-6.0, -1.0], [-6.0, 0.0], [-6.0, 1.0], [6.0, 1.0], [6.0, 0.0], [6.0, -1.0]])
    np.testing.assert_allclose(goals, [[6.0, 1.0], [6.0, 0.0], [6.0, -1.0], [-6.0, -1.0], [-6.0, 0.0], [-6.0, 1.0]])


def test_crossing_goals_mirror_starts_through_origin() -> None:
    starts, goals = _crossing()
    assert starts.shape == (6, 2)
    np.testing.assert_allclose(goals, -starts)
    assert [-5.0, 0.0] in starts.tolist()


def test_sparse_random_is_seeded_and_bounded() -> None:
    starts, goals = _sparse_random(8, 3)
    again_starts, again_goals = _sparse_random(8, 3)
    np.testing.assert_array_equal(starts, again_starts)
    np.testing.assert_array_equal(goals, again_goals)
    assert starts.shape == goals.shape == (8, 2)
    assert np.all(np.abs(starts) <= 6.0)
    assert np.all(np.abs(goals) <= 6.0)
    assert not np.array_equal(starts, _sparse_random(8, 4)[0])


def test_perturbed_jitters_starts_by_at_most_five_cm_and_keeps_goals() -> None:
    build = _perturbed(lambda: _circle(6, 4.0), 11)
    starts, goals = build()
    base_starts, base_goals = _circle(6, 4.0)
    np.testing.assert_array_equal(goals, base_goals)
    delta = starts - base_starts
    assert np.all(np.abs(delta) <= 0.05)
    assert np.any(delta != 0.0)
    np.testing.assert_array_equal(starts, build()[0])


def test_scenarios_are_held_out_set_with_expected_agent_counts() -> None:
    counts = {name: len(make()[0]) for name, make in SCENARIOS.items()}
    assert counts == {"circle_6": 6, "circle_10": 10, "corridor_pass_6": 6, "crossing_6": 6, "sparse_random_10": 10, "sparse_random_16": 16}
    for make in SCENARIOS.values():
        starts, goals = make()
        assert starts.shape == goals.shape


def test_params_share_reference_constants() -> None:
    p = _params()
    assert p.desired_velocity == V_PREF
    assert p.agent_radius == RADIUS
    assert p.max_velocity == V_MAX
    assert p.reaction_time == 0.0
    assert p.local_planner_params == {"relaxation_time": TAU, "repulsion_strength": A_REP, "repulsion_range": B_REP, "anisotropy": ANISOTROPY}


def test_pref_velocity_points_at_goal_with_preferred_speed() -> None:
    np.testing.assert_allclose(_pref_velocity(np.array([0.0, 0.0]), np.array([3.0, 4.0])), [0.6 * V_PREF, 0.8 * V_PREF])
    np.testing.assert_allclose(_pref_velocity(np.array([1.0, 1.0]), np.array([1.0, 1.2])), [0.0, V_PREF])


def test_pref_velocity_is_zero_inside_goal_radius() -> None:
    np.testing.assert_array_equal(_pref_velocity(np.array([1.0, 1.0]), np.array([1.1, 1.0])), [0.0, 0.0])


def test_straight_reference_walks_at_preferred_speed_and_stops_in_goal_radius() -> None:
    out = run_straight_reference(np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([[2.0, 0.0], [5.0, 5.0]]))
    assert out.shape == (_STEPS, 2, 4)
    np.testing.assert_allclose(out[10, 0], [10 * V_PREF * DT, 0.0, V_PREF, 0.0])
    np.testing.assert_allclose(out[-1, 0], [34 * V_PREF * DT, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[33, 0, 2], V_PREF)
    np.testing.assert_allclose(out[34, 0, 2], 0.0)
    np.testing.assert_allclose(out[:, 1], np.tile([5.0, 5.0, 0.0, 0.0], (_STEPS, 1)))


def test_compare_returns_per_agent_hausdorff_and_mean_speed_error() -> None:
    ours = np.zeros((3, 2, 4))
    ours[:, 0, 0] = [0.0, 1.0, 2.0]
    ours[:, 1, 1] = [5.0, 6.0, 7.0]
    ref = ours.copy()
    ref[:, 0, 0:2] += [0.3, 0.4]
    ref[:, 0, 2:4] = [0.6, 0.8]
    hd, dv = compare(ours, ref)
    assert hd == pytest.approx([0.5, 0.0])
    assert dv == pytest.approx(0.5)


def test_compare_hausdorff_is_symmetric_over_unequal_coverage() -> None:
    ours = np.zeros((3, 1, 4))
    ours[:, 0, 0] = [0.0, 1.0, 2.0]
    ref = np.zeros((3, 1, 4))
    ref[:, 0, 0] = [0.0, 1.0, 5.0]
    hd, dv = compare(ours, ref)
    assert hd == pytest.approx([3.0])
    assert dv == 0.0


def test_rvo2_lone_agent_matches_closed_form() -> None:
    pytest.importorskip("rvo2")
    starts, goals = np.array([[0.0, 0.0]]), np.array([[2.0, 0.0]])
    out = run_rvo2(starts, goals)
    assert out.shape == (_STEPS, 1, 4)
    np.testing.assert_allclose(out[10, 0], [10 * V_PREF * DT, 0.0, V_PREF, 0.0], atol=1e-5)
    np.testing.assert_allclose(out, run_straight_reference(starts, goals), atol=1e-4)


def test_pysocialforce_lone_agent_starts_at_preferred_velocity_on_its_line() -> None:
    pytest.importorskip("pysocialforce")
    out = run_pysocialforce(np.array([[0.0, 0.0]]), np.array([[10.0, 0.0]]))
    assert out.shape == (_STEPS, 1, 4)
    np.testing.assert_allclose(out[0, 0], [0.0, 0.0, V_PREF, 0.0])
    np.testing.assert_allclose(out[:, 0, 1], 0.0, atol=1e-9)
    assert np.all(np.diff(out[:, 0, 0]) >= 0.0)
    assert out[-1, 0, 0] <= 10.0


def test_pysocialforce_reference_advances_by_shared_time_step() -> None:
    pytest.importorskip("pysocialforce")
    out = run_pysocialforce(np.array([[0.0, 0.0]]), np.array([[10.0, 0.0]]))
    assert 0.0 < out[1, 0, 0] <= V_MAX * DT
    assert np.all(np.hypot(out[:, 0, 2], out[:, 0, 3]) <= V_MAX + 1e-9)


def test_run_correspondence_rows_for_unreferenced_drivers_fail() -> None:
    table = run_correspondence(["hsfm", "made_up_driver"])
    assert table["driver"].tolist() == ["hsfm", "made_up_driver"]
    assert table["reference"].tolist() == ["no public reference", "no public reference"]
    assert table["scenario"].tolist() == ["-", "-"]
    assert table["n_agents"].tolist() == [0, 0]
    assert table["median_hausdorff_m"].isna().all()
    assert not table["passes"].any()


def test_run_correspondence_straight_passes_every_scenario_and_writes_summary(tmp_path: Path) -> None:
    out = tmp_path / "corr"
    table = run_correspondence(["straight", "nsp"], out_dir=out)
    straight = table[table["driver"] == "straight"]
    assert straight["scenario"].tolist() == list(SCENARIOS)
    assert straight["n_agents"].tolist() == [6, 10, 6, 6, 10, 16]
    assert (straight["reference"] == "closed form").all()
    assert straight["max_hausdorff_m"].max() < 1e-6
    assert straight["mean_velocity_error_mps"].max() < 1e-6
    assert straight["passes"].all()
    assert table[table["driver"] == "nsp"]["reference"].tolist() == ["no step-wise public reference"]
    pd.testing.assert_frame_equal(pd.read_csv(out / "correspondence.csv", keep_default_na=False, na_values=[""]), table, check_dtype=False)
    summary = pd.read_csv(out / "correspondence_summary.csv")
    assert summary["driver"].tolist() == ["straight"]
    assert summary["scenarios"].tolist() == [6]
    assert summary["scenarios_passing"].tolist() == [6]
