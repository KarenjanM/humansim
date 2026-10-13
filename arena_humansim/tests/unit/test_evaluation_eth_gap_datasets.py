from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import numpy as np
import pandas as pd
from arena_humansim.utils.evaluation.eth_gap import (
    EWAP_STEP_S,
    SCALARS,
    _ewap_dir,
    atc_samples,
    ewap_samples,
    load_atc,
    load_ewap,
    plot,
    run_eth_gap,
    sim_samples,
)

from tests.unit._bags import NESTED, Agent, write_bag

_ATC_NOON_UTC_S = 3 * 3600.0


def _write_obsmat(path: Path, rows: list[tuple[int, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{frame:.7e} {ped:.7e} {x:.7e} 0.0000000e+00 {y:.7e} 0.0 0.0 0.0\n" for frame, ped, x, y in rows))


def _ewap(root: Path) -> Path:
    eth = [(6 * k, p, 0.4 * k, float(p)) for k in range(6) for p in (1, 2)]
    hotel = [(10 * k, 5, 0.0, 0.8 * k) for k in range(4)]
    _write_obsmat(root / "seq_eth" / "obsmat.txt", eth)
    _write_obsmat(root / "seq_hotel" / "obsmat.txt", hotel)
    return root


def _write_atc(path: Path, rows: list[tuple[float, int, float, float]]) -> Path:
    path.write_text("".join(f"{t},{pid},{x},{y},0.0,1200.0,0.0,0.0\n" for t, pid, x, y in rows))
    return path


def _atc_day(tmp_path: Path) -> Path:
    walker = [(_ATC_NOON_UTC_S + 10.0 + k * EWAP_STEP_S, 5, 1000.0 + 400.0 * k, 0.0) for k in range(6)]
    bystander = [(_ATC_NOON_UTC_S + 10.0 + k * EWAP_STEP_S, 7, 0.0, 3000.0) for k in range(6)]
    late = [(_ATC_NOON_UTC_S + 3600.0 + k, 9, 0.0, 0.0) for k in range(4)]
    return _write_atc(tmp_path / "atc-20121024.csv", walker + bystander + late)


def test_ewap_dir_keeps_a_complete_local_copy(tmp_path: Path) -> None:
    root = _ewap(tmp_path / "ewap")
    assert _ewap_dir(root) == root


def test_load_ewap_converts_frames_to_seconds_per_sequence(tmp_path: Path) -> None:
    df = load_ewap(_ewap(tmp_path))
    assert set(df["sequence"]) == {"seq_eth", "seq_hotel"}
    assert len(df) == 16
    eth = df[(df["sequence"] == "seq_eth") & (df["agent_id"] == 2)].sort_values("time")
    assert eth["time"].tolist() == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    assert eth["x"].tolist() == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    assert eth["y"].tolist() == pytest.approx([2.0] * 6)
    hotel = df[df["sequence"] == "seq_hotel"].sort_values("time")
    assert hotel["time"].tolist() == pytest.approx([0.0, 0.4, 0.8, 1.2])
    assert hotel["y"].tolist() == pytest.approx([0.0, 0.8, 1.6, 2.4])


def test_ewap_samples_label_each_sequence(tmp_path: Path) -> None:
    s = ewap_samples(_ewap(tmp_path))
    assert set(s["bucket"]) == {"eth"}
    eth = s[s["source"] == "ETH-eth"]
    assert sorted(eth["agent_id"].unique()) == [1, 2]
    assert eth["speed"].tolist() == pytest.approx([1.0] * 8)
    assert eth["clearance"].tolist() == pytest.approx([1.0] * 8)
    assert eth["turn_rate"].tolist() == pytest.approx([np.nan, 0.0, 0.0, np.nan] * 2, nan_ok=True)
    hotel = s[s["source"] == "ETH-hotel"]
    assert hotel["speed"].tolist() == pytest.approx([2.0, 2.0])
    assert hotel["clearance"].isna().all()


def test_load_atc_keeps_one_local_hour_in_meters_from_zero(tmp_path: Path) -> None:
    df = load_atc(_atc_day(tmp_path))
    assert list(df.columns) == ["time", "agent_id", "x", "y", "sequence"]
    assert sorted(df["agent_id"].unique()) == [5, 7]
    assert df["time"].min() == 0.0
    walker = df[df["agent_id"] == 5]
    assert walker["time"].tolist() == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    assert walker["x"].tolist() == pytest.approx([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
    assert set(df["sequence"]) == {"atc-20121024_h12"}


def test_load_atc_caps_people_in_order_of_appearance(tmp_path: Path) -> None:
    rows = [(_ATC_NOON_UTC_S + 2.0, 11, 0.0, 0.0), (_ATC_NOON_UTC_S + 1.0, 12, 0.0, 0.0), (_ATC_NOON_UTC_S + 3.0, 13, 0.0, 0.0), (_ATC_NOON_UTC_S + 0.5, 11, 0.0, 0.0)]
    df = load_atc(_write_atc(tmp_path / "atc.csv", rows), max_people=2)
    assert sorted(df["agent_id"].unique()) == [11, 12]
    assert df["time"].min() == 0.0


def test_load_atc_selects_other_hour(tmp_path: Path) -> None:
    df = load_atc(_atc_day(tmp_path), hour=13)
    assert df["agent_id"].unique().tolist() == [9]
    assert set(df["sequence"]) == {"atc-20121024_h13"}


def test_load_atc_rejects_hour_without_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows in local hour 5"):
        load_atc(_atc_day(tmp_path), hour=5)


def test_atc_samples_measure_walker_and_bystander(tmp_path: Path) -> None:
    s = atc_samples(_atc_day(tmp_path))
    assert set(s["source"]) == {"ATC"}
    assert set(s["bucket"]) == {"atc"}
    walker = s[s["agent_id"] == 5]
    assert walker["speed"].tolist() == pytest.approx([1.0] * 4)
    assert walker["clearance"].tolist() == pytest.approx([np.hypot(x, 3.0) for x in (1.4, 1.8, 2.2, 2.6)])


def _trial(sweep: Path, name: str, agents_at: list[list[Agent]]) -> None:
    write_bag(sweep / name / "bag", NESTED, [(int(round(k * EWAP_STEP_S * 1e9)), agents) for k, agents in enumerate(agents_at)])


def _sim_sweep(root: Path) -> Path:
    sweep = root / "sweep"
    _trial(sweep, "nav_sparse_corridor__sfm__1", [[Agent(1, 0.4 * k, 0.0), Agent(2, 0.4 * k, 2.0), Agent(-1, 50.0, 50.0)] for k in range(6)])
    _trial(sweep, "nav_sparse_corridor__orca__1", [[Agent(1, 0.8 * k, 0.0), Agent(2, 0.8 * k, 1.0)] for k in range(6)])
    _trial(sweep, "robot_nav_sparse_corridor__sfm__cadrl__1", [[Agent(1, 0.4 * k, 0.0), Agent(2, 0.4 * k, 2.0)] for k in range(4)])
    _trial(sweep, "nav_sparse_corridor__straight__1", [[Agent(-1, 0.4 * k, 0.0)] for k in range(4)])
    return sweep


def test_sim_samples_label_human_steps_by_driver(tmp_path: Path) -> None:
    s = sim_samples([_sim_sweep(tmp_path)])
    assert sorted(s["source"].unique()) == ["orca", "sfm"]
    assert set(s["bucket"]) == {"unknown"}
    assert set(s["agent_id"]) == {1, 2}
    sfm = s[s["source"] == "sfm"]
    assert sfm["speed"].tolist() == pytest.approx([1.0] * 8)
    assert sfm["clearance"].tolist() == pytest.approx([2.0] * 8)
    orca = s[s["source"] == "orca"]
    assert orca["speed"].tolist() == pytest.approx([2.0] * 8)
    assert orca["clearance"].tolist() == pytest.approx([1.0] * 8)
    assert set(s["group"]) == {"nav_sparse_corridor__sfm__1", "nav_sparse_corridor__orca__1"}


def test_sim_samples_of_robot_only_sweep_is_empty(tmp_path: Path) -> None:
    sweep = tmp_path / "robots"
    _trial(sweep, "robot_nav_sparse_corridor__sfm__cadrl__1", [[Agent(1, 0.4 * k, 0.0)] for k in range(4)])
    s = sim_samples([sweep])
    assert s.empty
    assert list(s.columns) == ["source", *SCALARS]


def _samples() -> pd.DataFrame:
    rows = []
    for src, speed in (("sfm", 1.2), ("orca", 1.4), ("ETH-eth", 1.3), ("ETH-hotel", 1.1), ("ATC", 1.0)):
        for k in range(5):
            rows.append({"source": src, "speed": speed + 0.01 * k, "clearance": 1.0 + 0.1 * k, "turn_rate": 0.1 * k})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("with_atc", [True, False])
def test_plot_writes_pdf_with_and_without_atc(tmp_path: Path, with_atc: bool) -> None:
    samples = _samples()
    if not with_atc:
        samples = samples[samples["source"] != "ATC"]
    out = tmp_path / "gap.pdf"
    plot(samples, out, ["sfm", "orca", "not_a_driver"])
    assert out.read_bytes().startswith(b"%PDF")


def test_plot_tolerates_driver_without_finite_turn_rate(tmp_path: Path) -> None:
    samples = _samples()
    samples.loc[samples["source"] == "orca", "turn_rate"] = np.nan
    out = tmp_path / "gap.pdf"
    plot(samples, out, ["sfm", "orca"])
    assert out.read_bytes().startswith(b"%PDF")


def test_run_eth_gap_writes_table_samples_and_figure(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    out = tmp_path / "out"
    table = run_eth_gap([_sim_sweep(tmp_path)], out, ewap=_ewap(tmp_path / "ewap"), atc=_atc_day(tmp_path))
    assert table["source"].tolist() == ["orca", "sfm", "ETH-eth", "ETH-hotel", "ATC"]
    keyed = table.set_index("source")
    assert keyed.loc["sfm", "speed_median"] == pytest.approx(1.0)
    assert keyed.loc["orca", "speed_median"] == pytest.approx(2.0)
    assert keyed.loc["ETH-eth", "n_steps"] == 8
    assert keyed.loc["ETH-eth", "turn_rate_median"] == 0.0
    assert keyed.loc["ATC", "speed_median"] == pytest.approx(0.5)
    assert keyed.loc["ATC", "speed_q25"] == 0.0
    assert keyed.loc["ATC", "speed_q75"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(pd.read_csv(out / "eth_gap.csv"), table, check_dtype=False)
    assert len(pd.read_parquet(out / "eth_gap_samples.parquet")) == int(table["n_steps"].sum())
    assert (out / "eth_gap.pdf").read_bytes().startswith(b"%PDF")


def test_run_eth_gap_without_atc_has_no_atc_row(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    table = run_eth_gap([_sim_sweep(tmp_path)], tmp_path / "out", ewap=_ewap(tmp_path / "ewap"))
    assert "ATC" not in set(table["source"])
    assert {"ETH-eth", "ETH-hotel"} <= set(table["source"])
