from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from arena_humansim.utils.evaluation.analyze import (
    _bootstrap_K,
    _bootstrap_mean,
    _scenario_equal_weight,
    compute_pairwise_table,
    headline,
    infer_bucket,
    parse_trial_dir,
    within_class_pairs,
)

_PAIR_COLUMNS = ["bucket", "scenario", "robot_policy", "seed", "agent_id", "p1", "p2", "class1", "class2", "same_class", "hausdorff"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("robot_nav_sparse_corridor__sfm__cadrl__7", ("robot_nav_sparse_corridor", "sfm", "cadrl", 7)),
        ("nav_sparse_corridor__orca__12", ("nav_sparse_corridor", "orca", "", 12)),
    ],
)
def test_parse_trial_dir_splits_three_and_four_part_names(name: str, expected: tuple[str, str, str, int]) -> None:
    assert parse_trial_dir(name) == expected


@pytest.mark.parametrize("name", ["nav_sparse_corridor__orca", "a__b__c__d__1", "nav__sfm__x"])
def test_parse_trial_dir_rejects_malformed_names(name: str) -> None:
    with pytest.raises(ValueError):
        parse_trial_dir(name)


@pytest.mark.parametrize(
    ("scenario_yaml", "scenario_id", "expected"),
    [
        ({"agent_types": {"waiter": {"mode": "behavior_tree"}}}, "het_dense_mixed_speeds", "het"),
        ({}, "het/dense/mixed_speeds", "het"),
        ({"agent_types": {"adult": {"mode": "navigate"}, "waiter": {"mode": "behavior_tree"}}}, "bt_sparse_pair", "bt"),
        ({"agent_types": {"adult": {"mode": "navigate"}, "alias": "adult"}}, "nav_sparse_corridor", "nav"),
        ({"agent_types": None}, "", "nav"),
    ],
)
def test_infer_bucket_classifies_scenarios(scenario_yaml: dict, scenario_id: str, expected: str) -> None:
    assert infer_bucket(scenario_yaml, scenario_id) == expected


def _trajectories(rows: list[tuple[str, str, int, int, str, list[tuple[float, float]]]]) -> pd.DataFrame:
    out = []
    for scenario, bucket, seed, agent_id, planner, points in rows:
        for t, (x, y) in enumerate(points):
            out.append({"scenario": scenario, "bucket": bucket, "robot_policy": "", "seed": seed, "agent_id": agent_id, "planner": planner, "time": float(t), "x": x, "y": y})
    return pd.DataFrame(out)


def _line(y: float) -> list[tuple[float, float]]:
    return [(0.0, y), (1.0, y), (2.0, y)]


def test_pairwise_table_measures_every_driver_pair_per_agent() -> None:
    df = _trajectories(
        [
            ("nav_sparse_corridor", "nav", 1, 1, "sfm", _line(0.0)),
            ("nav_sparse_corridor", "nav", 1, 1, "hsfm", _line(0.1)),
            ("nav_sparse_corridor", "nav", 1, 1, "nsp", _line(1.0)),
            ("nav_sparse_corridor", "nav", 1, 2, "sfm", _line(5.0)),
            ("nav_sparse_corridor", "nav", 2, 1, "sfm", _line(0.0)),
        ]
    )
    table = compute_pairwise_table(df)
    assert list(table.columns) == _PAIR_COLUMNS
    keyed = table.set_index(["p1", "p2"])
    assert sorted(keyed.index) == [("hsfm", "nsp"), ("hsfm", "sfm"), ("nsp", "sfm")]
    assert keyed.loc[("hsfm", "sfm"), "hausdorff"] == pytest.approx(0.1)
    assert keyed.loc[("hsfm", "nsp"), "hausdorff"] == pytest.approx(0.9)
    assert keyed.loc[("nsp", "sfm"), "hausdorff"] == pytest.approx(1.0)
    assert bool(keyed.loc[("hsfm", "sfm"), "same_class"])
    assert not bool(keyed.loc[("nsp", "sfm"), "same_class"])
    assert keyed.loc[("hsfm", "nsp"), "class1"] == "force"
    assert keyed.loc[("hsfm", "nsp"), "class2"] == "learned"
    assert set(table["agent_id"]) == {1}
    assert set(table["seed"]) == {1}
    assert set(table["bucket"]) == {"nav"}


def test_pairwise_table_marks_unregistered_driver_unknown() -> None:
    df = _trajectories([("s", "nav", 1, 1, "sfm", _line(0.0)), ("s", "nav", 1, 1, "zz_custom", _line(2.0))])
    row = compute_pairwise_table(df).iloc[0]
    assert (row["p1"], row["p2"]) == ("sfm", "zz_custom")
    assert row["class2"] == "unknown"
    assert not row["same_class"]
    assert row["hausdorff"] == pytest.approx(2.0)


def test_pairwise_table_without_pairs_is_empty_with_columns() -> None:
    table = compute_pairwise_table(_trajectories([("s", "nav", 1, 1, "sfm", _line(0.0))]))
    assert table.empty
    assert list(table.columns) == _PAIR_COLUMNS


def test_scenario_equal_weight_averages_scenario_means() -> None:
    df = pd.DataFrame({"scenario": ["a", "a", "b"], "hausdorff": [1.0, 3.0, 5.0]})
    assert _scenario_equal_weight(df) == pytest.approx(3.5)
    assert math.isnan(_scenario_equal_weight(df.iloc[0:0]))


def _pairs(rows: list[tuple[str, str, bool, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"bucket": b, "scenario": s, "same_class": same, "hausdorff": h} for b, s, same, h in rows])


def _nan_pair(pair: tuple[float, float]) -> bool:
    return math.isnan(pair[0]) and math.isnan(pair[1])


def test_bootstrap_K_is_nan_without_both_classes() -> None:
    rng = np.random.default_rng(0)
    assert _nan_pair(_bootstrap_K(_pairs([("nav", "a", True, 1.0)]), 10, rng))
    assert _nan_pair(_bootstrap_K(_pairs([("nav", "a", False, 1.0)]), 10, rng))
    assert _nan_pair(_bootstrap_K(_pairs([("nav", "a", True, 1.0)]).iloc[0:0], 10, rng))


def test_bootstrap_K_single_scenario_resamples_pairs() -> None:
    lo, hi = _bootstrap_K(_pairs([("nav", "a", True, 1.0), ("nav", "a", True, 1.0), ("nav", "a", False, 3.0)]), 50, np.random.default_rng(0))
    assert (lo, hi) == pytest.approx((3.0, 3.0))
    lo, hi = _bootstrap_K(_pairs([("nav", "a", True, 1.0), ("nav", "a", True, 3.0), ("nav", "a", False, 4.0)]), 500, np.random.default_rng(0))
    assert 4.0 / 3.0 <= lo < 2.0 < hi <= 4.0


def test_bootstrap_K_single_scenario_with_zero_within_distance_is_nan() -> None:
    assert _nan_pair(_bootstrap_K(_pairs([("nav", "a", True, 0.0), ("nav", "a", False, 3.0)]), 20, np.random.default_rng(0)))


def test_bootstrap_K_clusters_by_scenario() -> None:
    rows = [("nav", "a", True, 1.0), ("nav", "a", True, 1.0), ("nav", "a", False, 2.0), ("nav", "b", True, 2.0), ("nav", "b", False, 4.0), ("nav", "b", False, 4.0)]
    assert _bootstrap_K(_pairs(rows), 100, np.random.default_rng(0)) == pytest.approx((2.0, 2.0))


def test_bootstrap_K_without_common_scenario_is_nan() -> None:
    rows = [("nav", "a", True, 1.0), ("nav", "b", False, 2.0)]
    assert _nan_pair(_bootstrap_K(_pairs(rows), 20, np.random.default_rng(0)))


def test_bootstrap_K_multi_scenario_with_zero_within_distance_is_nan() -> None:
    rows = [("nav", "a", True, 0.0), ("nav", "a", False, 2.0), ("nav", "b", True, 0.0), ("nav", "b", False, 4.0)]
    assert _nan_pair(_bootstrap_K(_pairs(rows), 20, np.random.default_rng(0)))


def test_bootstrap_mean_of_empty_is_nan() -> None:
    assert _nan_pair(_bootstrap_mean(np.array([]), np.array([]), 10, np.random.default_rng(0)))


def test_bootstrap_mean_single_group_resamples_values() -> None:
    lo, hi = _bootstrap_mean(np.array([2.0, 2.0, 2.0]), np.array(["a", "a", "a"]), 20, np.random.default_rng(0))
    assert (lo, hi) == (2.0, 2.0)
    lo, hi = _bootstrap_mean(np.array([0.0, 1.0] * 10), np.array(["a"] * 20), 500, np.random.default_rng(0))
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_bootstrap_mean_weights_groups_equally() -> None:
    values = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
    groups = np.array(["a", "a", "a", "a", "b"])
    assert _bootstrap_mean(values, groups, 1000, np.random.default_rng(0)) == pytest.approx((1.0, 3.0))


def _labelled_pairs(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"bucket": b, "scenario": s, "p1": p1, "p2": p2, "hausdorff": h} for b, s, p1, p2, h in rows])


def test_within_class_pairs_of_empty_table_has_columns() -> None:
    out = within_class_pairs(pd.DataFrame(columns=["bucket", "scenario", "p1", "p2", "hausdorff"]))
    assert out.empty
    assert list(out.columns) == ["bucket", "class", "p1", "p2", "mean_hausdorff", "ci_lo", "ci_hi", "n_pairs", "n_scenarios"]


def test_within_class_pairs_without_intra_class_pair_is_empty() -> None:
    out = within_class_pairs(_labelled_pairs([("nav", "a", "nsp", "sfm", 1.0), ("nav", "a", "sfm", "zz_custom", 1.0), ("nav", "a", "zz_custom", "zz_other", 1.0)]))
    assert out.empty
    assert "mean_hausdorff" in out.columns


def test_within_class_pairs_scenario_weighted_per_bucket_and_pooled() -> None:
    rows = [
        ("nav", "a", "hsfm", "sfm", 1.0),
        ("nav", "a", "hsfm", "sfm", 3.0),
        ("nav", "b", "hsfm", "sfm", 5.0),
        ("bt", "c", "hsfm", "sfm", 4.0),
        ("bt", "c", "nsp", "socialgail", 6.0),
        ("nav", "a", "nsp", "sfm", 9.0),
        ("weird", "z", "hsfm", "sfm", 100.0),
    ]
    out = within_class_pairs(_labelled_pairs(rows), n_bootstrap=200)
    keyed = out.set_index(["bucket", "p1", "p2"])
    assert out[["bucket", "class", "p1", "p2"]].values.tolist() == [
        ["all", "force", "hsfm", "sfm"],
        ["all", "learned", "nsp", "socialgail"],
        ["bt", "force", "hsfm", "sfm"],
        ["bt", "learned", "nsp", "socialgail"],
        ["nav", "force", "hsfm", "sfm"],
    ]
    nav = keyed.loc[("nav", "hsfm", "sfm")]
    assert nav["mean_hausdorff"] == pytest.approx(3.5)
    assert nav["n_pairs"] == 3
    assert nav["n_scenarios"] == 2
    assert 2.0 <= nav["ci_lo"] <= nav["ci_hi"] <= 5.0
    pooled = keyed.loc[("all", "hsfm", "sfm")]
    assert pooled["mean_hausdorff"] == pytest.approx((2.0 + 5.0 + 4.0) / 3.0)
    assert pooled["n_pairs"] == 4
    assert pooled["n_scenarios"] == 3
    bt = keyed.loc[("bt", "nsp", "socialgail")]
    assert bt["mean_hausdorff"] == pytest.approx(6.0)
    assert (bt["ci_lo"], bt["ci_hi"]) == pytest.approx((6.0, 6.0))


def test_within_class_pairs_outside_known_buckets_only_is_empty() -> None:
    out = within_class_pairs(_labelled_pairs([("unknown", "z", "hsfm", "sfm", 1.0)]))
    assert out.empty
    assert list(out.columns) == ["bucket", "class", "p1", "p2", "mean_hausdorff", "ci_lo", "ci_hi", "n_pairs", "n_scenarios"]


def test_headline_ratio_is_across_over_within_per_bucket_and_pooled() -> None:
    rows = [
        ("nav", "a", True, 1.0),
        ("nav", "a", True, 1.0),
        ("nav", "a", False, 2.0),
        ("nav", "b", True, 2.0),
        ("nav", "b", False, 4.0),
        ("het", "c", True, 0.5),
        ("het", "c", False, 3.0),
        ("weird", "d", True, 10.0),
        ("weird", "d", False, 1.0),
    ]
    out = headline(_pairs(rows), n_bootstrap=100)
    assert list(out.columns) == ["bucket", "within_class_mean", "across_class_mean", "ratio_K", "K_lo", "K_hi", "n_pairs", "n_scenarios"]
    assert out["bucket"].tolist() == ["nav", "het", "all"]
    keyed = out.set_index("bucket")
    assert keyed.loc["nav", "within_class_mean"] == pytest.approx(1.5)
    assert keyed.loc["nav", "across_class_mean"] == pytest.approx(3.0)
    assert keyed.loc["nav", "ratio_K"] == pytest.approx(2.0)
    assert (keyed.loc["nav", "K_lo"], keyed.loc["nav", "K_hi"]) == pytest.approx((2.0, 2.0))
    assert keyed.loc["nav", "n_pairs"] == 5
    assert keyed.loc["nav", "n_scenarios"] == 2
    assert keyed.loc["het", "ratio_K"] == pytest.approx(6.0)
    assert keyed.loc["all", "within_class_mean"] == pytest.approx((1.0 + 2.0 + 0.5) / 3.0)
    assert keyed.loc["all", "across_class_mean"] == pytest.approx(3.0)
    assert keyed.loc["all", "n_pairs"] == 7


def test_headline_ratio_is_nan_for_zero_or_missing_within_distance() -> None:
    out = headline(_pairs([("nav", "a", True, 0.0), ("nav", "a", False, 2.0), ("bt", "b", False, 2.0)]), n_bootstrap=10).set_index("bucket")
    assert math.isnan(out.loc["nav", "ratio_K"])
    assert math.isnan(out.loc["bt", "within_class_mean"])
    assert math.isnan(out.loc["bt", "ratio_K"])
    assert math.isnan(out.loc["bt", "K_lo"])


def test_headline_of_unknown_buckets_is_empty_with_columns() -> None:
    out = headline(_pairs([("weird", "a", True, 1.0), ("weird", "a", False, 2.0)]))
    assert out.empty
    assert "ratio_K" in out.columns
