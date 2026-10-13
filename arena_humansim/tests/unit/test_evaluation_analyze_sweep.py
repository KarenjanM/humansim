from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import pandas as pd
from arena_humansim.utils.evaluation.analyze import _enumerate_unique_trials, iter_trials, run_analysis

from tests.unit._bags import NESTED, Agent, write_bag

_BT_SNAPSHOT = "agent_types:\n  waiter: {mode: behavior_tree}\n"
_NSP_OFFSET = {"nav": 1.0, "bt": 2.0, "het": 3.0}
_SCENARIO = {"nav": "nav_sparse_corridor", "bt": "bt_sparse_pair", "het": "het_dense_mixed_speeds"}


def _record(trial: Path, ys: dict[int, float], snapshot: str | None = None, accel: float = 0.0) -> None:
    trial.mkdir(parents=True)
    if snapshot is not None:
        (trial / "scenario.yaml").write_text(snapshot)
    frames = [(t * 100_000_000, [Agent(aid, 0.1 * t, y, vx=1.0 + accel * t * t) for aid, y in ys.items()]) for t in range(20)]
    write_bag(trial / "bag", NESTED, frames)


def _mixed_sweeps(root: Path) -> tuple[Path, Path]:
    a = root / "sweep_a"
    a.mkdir()
    (a / "README.txt").write_text("not a trial\n")
    (a / "notes").mkdir()
    _record(a / "nav_sparse_corridor__sfm__1", {1: 0.0}, "agents: []\n")
    _record(a / "bt_sparse_pair__orca__1", {1: 0.0, 2: 1.0}, _BT_SNAPSHOT)
    _record(a / "het_dense_mixed_speeds__sfm__cadrl__2", {1: 0.0, -1: 2.0})
    _record(a / "nav_sparse_merge__sfm__3", {1: 0.0}, "agents: [\n")
    _record(a / "nav_sparse_flow__sfm__5", {1: 0.0}, "")
    _record(a / "robot_nav_sparse_crossing__orca__1", {1: 0.0}, "agents: []\n")
    (a / "nav_sparse_bend__sfm__4").mkdir()
    b = root / "sweep_b"
    _record(b / "nav_sparse_corridor__sfm__1", {1: 9.0}, "agents: []\n")
    _record(b / "nav_sparse_corridor__sfm__9", {1: 0.0}, "agents: []\n")
    return a, b


def test_iter_trials_annotates_unique_trials_and_skips_the_rest(tmp_path: Path) -> None:
    a, b = _mixed_sweeps(tmp_path)
    trials = {meta["trial_name"]: (df, meta) for df, meta in iter_trials([a, b])}
    assert sorted(trials) == [
        "bt_sparse_pair__orca__1",
        "het_dense_mixed_speeds__sfm__cadrl__2",
        "nav_sparse_corridor__sfm__1",
        "nav_sparse_corridor__sfm__9",
        "nav_sparse_flow__sfm__5",
        "nav_sparse_merge__sfm__3",
        "robot_nav_sparse_crossing__orca__1",
    ]
    buckets = {name: meta["bucket"] for name, (_, meta) in trials.items()}
    assert buckets == {
        "bt_sparse_pair__orca__1": "bt",
        "het_dense_mixed_speeds__sfm__cadrl__2": "unknown",
        "nav_sparse_corridor__sfm__1": "nav",
        "nav_sparse_corridor__sfm__9": "nav",
        "nav_sparse_flow__sfm__5": "nav",
        "nav_sparse_merge__sfm__3": "unknown",
        "robot_nav_sparse_crossing__orca__1": "nav",
    }
    df, meta = trials["nav_sparse_corridor__sfm__1"]
    assert meta["source_dir"] == "sweep_a"
    assert meta["recordings_dir"] == a
    assert set(df["y"]) == {0.0}
    df, meta = trials["het_dense_mixed_speeds__sfm__cadrl__2"]
    assert (meta["scenario"], meta["planner"], meta["robot_policy"], meta["seed"]) == ("het_dense_mixed_speeds", "sfm", "cadrl", 2)
    assert (meta["density"], meta["modality"], meta["is_robot_scenario"]) == ("dense", "mixed_speeds", False)
    assert sorted(df["agent_id"].unique()) == [-1, 1]
    for column, value in (("scenario", "het_dense_mixed_speeds"), ("seed", 2), ("planner", "sfm"), ("robot_policy", "cadrl"), ("bucket", "unknown"), ("density", "dense"), ("modality", "mixed_speeds"), ("is_robot_scenario", False), ("source_dir", "sweep_a")):
        assert set(df[column]) == {value}
    _, meta = trials["robot_nav_sparse_crossing__orca__1"]
    assert (meta["density"], meta["modality"], meta["is_robot_scenario"]) == ("sparse", "crossing", True)
    assert trials["nav_sparse_corridor__sfm__9"][1]["source_dir"] == "sweep_b"


def test_iter_trials_accepts_single_dir(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    _record(sweep / "nav_sparse_corridor__sfm__1", {1: 0.0, 2: 1.0}, "agents: []\n")
    ((df, meta),) = list(iter_trials(sweep))
    assert meta["trial_name"] == "nav_sparse_corridor__sfm__1"
    assert len(df) == 40


def test_enumerate_unique_trials_dedups_by_parsed_key_and_skips_missing_dirs(tmp_path: Path) -> None:
    a, b = _mixed_sweeps(tmp_path)
    found = _enumerate_unique_trials([a, b, tmp_path / "absent"])
    assert [(d.name, name) for d, name in found] == [
        ("sweep_a", "bt_sparse_pair__orca__1"),
        ("sweep_a", "het_dense_mixed_speeds__sfm__cadrl__2"),
        ("sweep_a", "nav_sparse_bend__sfm__4"),
        ("sweep_a", "nav_sparse_corridor__sfm__1"),
        ("sweep_a", "nav_sparse_flow__sfm__5"),
        ("sweep_a", "nav_sparse_merge__sfm__3"),
        ("sweep_a", "robot_nav_sparse_crossing__orca__1"),
        ("sweep_b", "nav_sparse_corridor__sfm__9"),
    ]


def _three_bucket_sweep(root: Path) -> Path:
    sweep = root / "sweep"
    snapshots = {"nav": "agents: []\n", "bt": _BT_SNAPSHOT, "het": "agents: []\n"}
    for bucket, scenario in _SCENARIO.items():
        for planner, offset, accel in (("sfm", 0.0, 0.0), ("hsfm", 0.1, 0.001), ("nsp", _NSP_OFFSET[bucket], 0.002)):
            for seed in (1, 2):
                _record(sweep / f"{scenario}__{planner}__{seed}", {1: offset + 0.01 * seed}, snapshots[bucket], accel + 0.0001 * seed)
    return sweep


def test_run_analysis_headline_ratios_on_three_bucket_sweep(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run_analysis([_three_bucket_sweep(tmp_path)], out, n_bootstrap=50)

    assert len(result["kinematics"]) == 18
    assert set(result["kinematics"]["collisions"]) == {0}
    pairwise = result["pairwise"]
    assert len(pairwise) == 18
    nav_pairs = pairwise[pairwise["bucket"] == "nav"].set_index(["seed", "p1", "p2"])["hausdorff"]
    assert nav_pairs.loc[(1, "hsfm", "sfm")] == pytest.approx(0.1)
    assert nav_pairs.loc[(1, "hsfm", "nsp")] == pytest.approx(0.9)
    assert nav_pairs.loc[(2, "nsp", "sfm")] == pytest.approx(1.0)
    assert len(result["self_divergence"]) == 9
    assert result["self_divergence"]["hausdorff"].tolist() == pytest.approx([0.01] * 9)

    head = result["headline"].set_index("bucket")
    assert list(head.index) == ["nav", "bt", "het", "all"]
    for bucket, k, k_min, k_max in (("nav", 9.5, 9.0, 10.0), ("bt", 19.5, 19.0, 20.0), ("het", 29.5, 29.0, 30.0), ("all", 19.5, 9.5, 29.5)):
        assert head.loc[bucket, "within_class_mean"] == pytest.approx(0.1)
        assert head.loc[bucket, "ratio_K"] == pytest.approx(k)
        assert k_min - 1e-9 <= head.loc[bucket, "K_lo"] <= k <= head.loc[bucket, "K_hi"] <= k_max + 1e-9
    assert head.loc["nav", "within_driver_seed_mean"] == pytest.approx(0.01)
    assert head.loc["nav", "K_seed"] == pytest.approx(95.0)
    assert head.loc["het", "n_self_pairs"] == 3

    kin = result["kinematics"].set_index(["scenario", "planner", "seed"])
    assert kin.loc[("nav_sparse_corridor", "nsp", 2), "jerk"] == pytest.approx(2 * 0.0022 / 0.05**2)
    assert set(result["kinematics"]["curvature"]) == {0.0}
    variance = result["variance_decomposition"].set_index("bucket")
    assert list(variance.index) == ["nav", "bt", "het", "all"]
    assert set(variance["metric"]) == {"jerk"}
    ss_driver, ss_seed = 2 * 2 * 0.8**2, 2 * 3 * 0.04**2
    for bucket in ("nav", "bt", "het"):
        assert variance.loc[bucket, "n_trials"] == 6
        assert variance.loc[bucket, "ss_total"] == pytest.approx(ss_driver + ss_seed)
        assert variance.loc[bucket, "share_driver"] == pytest.approx(ss_driver / (ss_driver + ss_seed))
        assert variance.loc[bucket, "share_seed"] == pytest.approx(ss_seed / (ss_driver + ss_seed))
        assert variance.loc[bucket, "share_scenario"] == pytest.approx(0.0, abs=1e-12)
        assert variance.loc[bucket, "share_residual"] == pytest.approx(0.0, abs=1e-9)
    assert variance.loc["all", "share_driver"] == pytest.approx(ss_driver / (ss_driver + ss_seed))
    intra = pd.read_csv(out / "within_class_pairs.csv")
    assert intra["bucket"].tolist() == ["all", "bt", "het", "nav"]
    assert set(zip(intra["p1"], intra["p2"], strict=True)) == {("hsfm", "sfm")}
    assert intra["mean_hausdorff"].tolist() == pytest.approx([0.1] * 4)

    framing = result["framing"]
    assert framing["recommended"] == "trifurcated"
    assert framing["X_nav"] == pytest.approx(0.1)
    assert framing["K_pooled"] == pytest.approx(19.5)
    assert "**Recommended:** `trifurcated`" in (out / "framing.md").read_text()
    pd.testing.assert_frame_equal(pd.read_csv(out / "headline.csv").set_index("bucket"), head, check_dtype=False)


def test_run_analysis_single_trial_has_no_seed_ratio(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    _record(sweep / "nav_sparse_corridor__sfm__1", {1: 0.0}, "agents: []\n")
    result = run_analysis(sweep, tmp_path / "out", n_bootstrap=5)
    assert result["self_divergence"].empty
    assert result["headline"].empty
    assert "K_seed" not in result["headline"].columns
    assert result["variance_decomposition"].empty


def test_run_analysis_rejects_sweep_without_trials(tmp_path: Path) -> None:
    sweep = tmp_path / "empty"
    (sweep / "nav_sparse_corridor__sfm__1").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no usable trials"):
        run_analysis([sweep], tmp_path / "out", n_bootstrap=5)
