from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from arena_humansim.utils.evaluation.eth_gap import EWAP_STEP_S, SCALARS, _iqr, _per_step_scalars, _turn_rate, summarize


def _line(n: int, step_m: float) -> np.ndarray:
    return np.stack([np.arange(n) * step_m, np.zeros(n)], axis=1)


def test_turn_rate_is_zero_on_straight_line() -> None:
    turn = _turn_rate(_line(5, 0.4), np.arange(5))
    assert turn.shape == (3,)
    assert math.isnan(turn[0])
    assert turn[1] == 0.0
    assert math.isnan(turn[2])


def test_turn_rate_of_right_angle_over_chord_span() -> None:
    xy = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [0.8, 0.4], [0.8, 0.8]])
    turn = _turn_rate(xy, np.arange(5))
    assert turn[1] == pytest.approx((math.pi / 2) / (2 * EWAP_STEP_S))


def test_turn_rate_wraps_heading_change_across_pi() -> None:
    back = 0.8 * np.array([math.cos(math.radians(170.0)), math.sin(math.radians(170.0))])
    fwd = 0.8 * np.array([math.cos(math.radians(-170.0)), math.sin(math.radians(-170.0))])
    xy = np.array([[0.0, 0.0], back / 2, back, back + fwd / 2, back + fwd])
    turn = _turn_rate(xy, np.arange(5))
    assert turn[1] == pytest.approx(math.radians(20.0) / (2 * EWAP_STEP_S))


def test_turn_rate_skips_chords_spanning_a_gap() -> None:
    turn = _turn_rate(_line(5, 0.4), np.array([0, 1, 2, 4, 5]))
    assert np.isnan(turn).all()


def test_turn_rate_skips_chords_below_moving_speed() -> None:
    xy = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [0.8, 0.03], [0.8, 0.06]])
    assert np.isnan(_turn_rate(xy, np.arange(5))).all()
    xy = np.array([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [0.8, 0.05], [0.8, 0.09]])
    assert _turn_rate(xy, np.arange(5))[1] == pytest.approx((math.pi / 2) / (2 * EWAP_STEP_S))


@pytest.mark.parametrize("n", [0, 1, 2])
def test_turn_rate_of_short_track_is_empty(n: int) -> None:
    assert _turn_rate(np.zeros((n, 2)), np.arange(n)).shape == (0,)


def _walker_and_bystander() -> pd.DataFrame:
    rows = []
    for k in range(4):
        rows.append({"g": "a", "time": k * EWAP_STEP_S, "agent_id": 1, "x": 0.4 * k, "y": 0.0})
        rows.append({"g": "a", "time": k * EWAP_STEP_S, "agent_id": 2, "x": 0.0, "y": 3.0})
    rows.append({"g": "a", "time": 0.1, "agent_id": 1, "x": 99.0, "y": 99.0})
    rows.append({"g": "a", "time": 0.0, "agent_id": 3, "x": 50.0, "y": 50.0})
    rows.append({"g": "a", "time": 0.4, "agent_id": 3, "x": 50.0, "y": 50.0})
    return pd.DataFrame(rows)


def test_per_step_scalars_on_walker_passing_bystander() -> None:
    out = _per_step_scalars(_walker_and_bystander(), "g")
    assert list(out.columns) == ["group", "agent_id", *SCALARS]
    assert sorted(out["agent_id"].unique()) == [1, 2]
    walker = out[out["agent_id"] == 1]
    assert walker["speed"].tolist() == pytest.approx([1.0, 1.0])
    assert walker["clearance"].tolist() == pytest.approx([math.hypot(0.4, 3.0), math.hypot(0.8, 3.0)])
    assert walker["turn_rate"].isna().all()
    bystander = out[out["agent_id"] == 2]
    assert bystander["speed"].tolist() == [0.0, 0.0]
    assert bystander["clearance"].tolist() == pytest.approx([math.hypot(0.4, 3.0), math.hypot(0.8, 3.0)])
    assert (out["group"] == "a").all()


def test_per_step_scalars_clearance_is_nan_for_lone_agent_and_groups_stay_apart() -> None:
    a = pd.DataFrame({"g": "a", "time": np.arange(5) * EWAP_STEP_S, "agent_id": 1, "x": np.arange(5) * 0.4, "y": 0.0})
    b = pd.DataFrame({"g": "b", "time": np.arange(3) * EWAP_STEP_S, "agent_id": 1, "x": 0.0, "y": np.arange(3) * 0.8})
    out = _per_step_scalars(pd.concat([a, b], ignore_index=True), "g")
    assert out["clearance"].isna().all()
    assert out.loc[out["group"] == "a", "turn_rate"].iloc[1] == 0.0
    assert out.loc[out["group"] == "b", "speed"].tolist() == pytest.approx([2.0])


def test_per_step_scalars_snaps_samples_to_the_grid_and_spans_gaps() -> None:
    df = pd.DataFrame({"g": "a", "time": [0.0, 0.41, 0.79, 1.6], "agent_id": 7, "x": [0.0, 0.4, 0.8, 2.4], "y": 0.0})
    out = _per_step_scalars(df, "g")
    assert out["speed"].tolist() == pytest.approx([1.0, 1.6 / 0.8])


def test_per_step_scalars_without_long_tracks_is_empty_with_columns() -> None:
    df = pd.DataFrame({"g": "a", "time": [0.0, 0.4], "agent_id": 1, "x": [0.0, 0.4], "y": 0.0})
    out = _per_step_scalars(df, "g")
    assert out.empty
    assert list(out.columns) == ["group", "agent_id", *SCALARS]


def test_summarize_reports_median_mean_quartiles_ignoring_nan() -> None:
    samples = pd.DataFrame(
        {
            "source": ["sfm", "sfm", "sfm", "sfm", "sfm", "ETH-eth"],
            "speed": [1.0, 2.0, 3.0, 4.0, np.nan, 1.3],
            "clearance": [np.inf, 1.0, 1.0, 1.0, 1.0, 0.7],
            "turn_rate": [np.nan, np.nan, np.nan, np.nan, np.nan, 0.2],
        }
    )
    table = summarize(samples).set_index("source")
    assert list(table.index) == ["sfm", "ETH-eth"]
    sfm = table.loc["sfm"]
    assert sfm["n_steps"] == 5
    assert sfm["speed_median"] == pytest.approx(2.5)
    assert sfm["speed_mean"] == pytest.approx(2.5)
    assert sfm["speed_q25"] == pytest.approx(1.75)
    assert sfm["speed_q75"] == pytest.approx(3.25)
    assert sfm["clearance_mean"] == pytest.approx(1.0)
    for stat in ("median", "mean", "q25", "q75"):
        assert math.isnan(sfm[f"turn_rate_{stat}"])
    assert table.loc["ETH-eth", "turn_rate_median"] == pytest.approx(0.2)


def test_iqr_drops_non_finite_values() -> None:
    assert _iqr(np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.inf])) == pytest.approx((1.75, 2.5, 3.25))
