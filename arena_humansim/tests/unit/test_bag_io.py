from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

from ament_index_python.packages import get_package_share_directory
from arena_humansim.utils.bag_io import AGENT_FRAME_TYPE, AGENT_META_TYPE, AgentSources, extract_agent_states, policy_names, source_attribution
from rclpy.serialization import serialize_message
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from tests.unit._bags import FLAT, LAYOUTS, NESTED, Agent, Frames, frame_msg, meta_msg, write_bag

_TRAJECTORY: Frames = [
    (0, [Agent(1, 0.0, 1.0, vx=0.1), Agent(-1, 3.0, 0.0, policy="", kind=1, radius=0.5)]),
    (50_000_000, [Agent(1, 0.5, 1.0, vx=0.1, name="alice", handedness="l"), Agent(2, 0.0, -1.0, vy=0.2, policy="orca"), Agent(-1, 3.1, 0.0, policy="", kind=1, radius=0.5)]),
    (100_000_000, [Agent(1, 1.0, 1.0, vx=0.1, policy="orca", name="alice", handedness="l"), Agent(-1, 3.2, 0.0, policy="", kind=1, radius=0.5)]),
]


def test_flat_and_nested_bags_extract_the_same_frame(tmp_path: Path) -> None:
    write_bag(tmp_path / NESTED, NESTED, _TRAJECTORY)
    write_bag(tmp_path / FLAT, FLAT, _TRAJECTORY)

    nested = extract_agent_states(tmp_path / NESTED)
    flat = extract_agent_states(tmp_path / FLAT)

    pd.testing.assert_frame_equal(nested, flat)
    assert list(flat.columns) == ["time", "agent_id", "x", "y", "vx", "vy", "radius", "planner", "source_name", "source_seq"]
    assert flat["planner"].tolist() == ["sfm", "", "sfm", "orca", "", "orca", ""]
    assert flat["agent_id"].tolist() == [1, -1, 1, 2, -1, 1, -1]
    assert flat["time"].tolist() == [0.0, 0.0, 0.05, 0.05, 0.05, 0.1, 0.1]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bag_with_only_empty_frames_extracts_empty(tmp_path: Path, layout: str) -> None:
    write_bag(tmp_path / "bag", layout, [(0, []), (50_000_000, [])])

    assert extract_agent_states(tmp_path / "bag").empty


@pytest.mark.parametrize("layout", LAYOUTS)
def test_empty_frames_around_agents_add_no_rows(tmp_path: Path, layout: str) -> None:
    write_bag(tmp_path / "bag", layout, [(0, []), (50_000_000, [Agent(4, 1.5, 2.5, vx=0.3)]), (100_000_000, [])])

    df = extract_agent_states(tmp_path / "bag")

    assert df.to_dict("records") == [{"time": 0.05, "agent_id": 4, "x": 1.5, "y": 2.5, "vx": 0.3, "vy": 0.0, "radius": 0.35, "planner": "sfm", "source_name": "", "source_seq": 0}]


def test_flat_meta_written_after_its_frames_still_names_them(tmp_path: Path) -> None:
    write_bag(tmp_path / "bag", FLAT, _TRAJECTORY, late_meta=True)

    assert extract_agent_states(tmp_path / "bag")["planner"].tolist() == ["sfm", "", "sfm", "orca", "", "orca", ""]


def test_flat_frames_without_meta_have_empty_planner(tmp_path: Path) -> None:
    write_bag(tmp_path / "bag", FLAT, _TRAJECTORY, with_meta=False)

    df = extract_agent_states(tmp_path / "bag")

    assert df["planner"].tolist() == [""] * 7
    assert df["x"].tolist() == [0.0, 3.0, 0.5, 0.0, 3.1, 1.0, 3.2]


def test_policy_names_uses_the_latest_meta_at_or_before_each_frame() -> None:
    names = policy_names(
        meta_t=np.array([20, 10]),
        meta_policies=[["sfm", "orca"], ["sfm"]],
        frame_t=np.array([5, 10, 20]),
        counts=np.array([1, 2, 2]),
        policy_idx=np.array([0, 0, 1, 1, -1]),
    )

    assert names.tolist() == ["", "sfm", "", "orca", ""]


_RELEASED: Frames = [
    (0, [Agent(1, 0.0, 1.0, source_name="source_0"), Agent(2, 4.0, 0.0, source_name="source_1", source_seq=3), Agent(-1, 3.0, 0.0, policy="", kind=1)]),
    (50_000_000, [Agent(1, 0.5, 1.0, source_name="source_0"), Agent(3, 2.0, 2.0, name="bob"), Agent(-1, 3.1, 0.0, policy="", kind=1)]),
    (100_000_000, [Agent(1, -6.0, 0.0, source_name="source_1", source_seq=4), Agent(-1, 3.2, 0.0, policy="", kind=1)]),
]


def test_flat_bag_carries_source_and_release_index_per_agent_row(tmp_path: Path) -> None:
    write_bag(tmp_path / "bag", FLAT, _RELEASED)

    df = extract_agent_states(tmp_path / "bag")

    assert df["agent_id"].tolist() == [1, 2, -1, 1, 3, -1, 1, -1]
    assert df["source_name"].tolist() == ["source_0", "source_1", "", "source_0", "", "", "source_1", ""]
    assert df["source_seq"].tolist() == [0, 3, 0, 0, 0, 0, 4, 0]
    assert df["source_seq"].dtype == np.int64


def test_flat_meta_written_after_its_frames_still_attributes_them(tmp_path: Path) -> None:
    write_bag(tmp_path / "bag", FLAT, _RELEASED, late_meta=True)

    df = extract_agent_states(tmp_path / "bag")

    assert df["source_name"].tolist() == ["source_0", "source_1", "", "source_0", "", "", "source_1", ""]
    assert df["source_seq"].tolist() == [0, 3, 0, 0, 0, 0, 4, 0]


@pytest.mark.parametrize("with_meta", [True, False])
def test_unattributed_bags_read_empty_sources(tmp_path: Path, with_meta: bool) -> None:
    write_bag(tmp_path / "nested", NESTED, _RELEASED)
    write_bag(tmp_path / "flat", FLAT, _TRAJECTORY, with_meta=with_meta)

    for df in (extract_agent_states(tmp_path / "nested"), extract_agent_states(tmp_path / "flat")):
        assert set(df["source_name"]) == {""}
        assert set(df["source_seq"]) == {0}


_AGENT_META_BEFORE_SOURCES = """std_msgs/Header header
string[] policies
int32[] agent_id
string[] name
string[] handedness
string[] agent_type
"""


def test_bag_recorded_before_source_attribution_reads_empty_sources(tmp_path: Path) -> None:
    frame_def = (Path(get_package_share_directory("arena_humansim_msgs")) / "msg" / "AgentFrame.msg").read_text()
    store = get_typestore(Stores.ROS2_JAZZY)
    store.register({**get_types_from_msg(_AGENT_META_BEFORE_SOURCES, AGENT_META_TYPE), **get_types_from_msg(frame_def, AGENT_FRAME_TYPE)})
    header = store.types["std_msgs/msg/Header"](stamp=store.types["builtin_interfaces/msg/Time"](sec=0, nanosec=0), frame_id="map")
    meta = store.types[AGENT_META_TYPE](header=header, policies=["sfm"], agent_id=np.array([1, -1], dtype=np.int32), name=["", ""], handedness=["", ""], agent_type=["adult", ""])
    agents = [Agent(1, 0.0, 1.0), Agent(-1, 3.0, 0.0, policy="", kind=1)]
    with Writer(tmp_path / "bag", version=8, storage_plugin=StoragePlugin.MCAP) as writer:
        meta_connection = writer.add_connection("/agent_meta", AGENT_META_TYPE, typestore=store)
        frame_connection = writer.add_connection("/agent_states", AGENT_FRAME_TYPE, typestore=store)
        writer.write(meta_connection, 0, store.serialize_cdr(meta, AGENT_META_TYPE))
        writer.write(frame_connection, 0, serialize_message(frame_msg(0, agents, ["sfm"])))

    df = extract_agent_states(tmp_path / "bag")

    assert df["planner"].tolist() == ["sfm", ""]
    assert df["source_name"].tolist() == ["", ""]
    assert df["source_seq"].tolist() == [0, 0]


def test_source_attribution_uses_the_latest_meta_at_or_before_each_frame_else_the_next_naming_the_agent() -> None:
    names, seqs = source_attribution(
        meta_t=np.array([20, 10]),
        meta_sources=[AgentSources([1, 2], ["source_1", "source_0"], [4, 2]), AgentSources([1], ["source_0"], [0])],
        frame_t=np.array([5, 10, 20]),
        counts=np.array([1, 2, 2]),
        agent_id=np.array([1, 1, 2, 2, 1]),
    )

    assert names.tolist() == ["source_0", "source_0", "source_0", "source_0", "source_1"]
    assert seqs.tolist() == [0, 0, 2, 2, 4]


def test_source_attribution_never_names_a_reused_id_after_its_previous_pedestrian() -> None:
    names, seqs = source_attribution(
        meta_t=np.array([0, 10, 30]),
        meta_sources=[AgentSources([5], ["west"], [1]), AgentSources([], [], []), AgentSources([5], ["east"], [0])],
        frame_t=np.array([0, 20, 30]),
        counts=np.array([1, 1, 1]),
        agent_id=np.array([5, 5, 5]),
    )

    assert names.tolist() == ["west", "east", "east"]
    assert seqs.tolist() == [1, 0, 0]


def test_source_attribution_leaves_agents_no_meta_names_empty() -> None:
    names, seqs = source_attribution(
        meta_t=np.array([0]),
        meta_sources=[AgentSources([1], ["west"], [2])],
        frame_t=np.array([0]),
        counts=np.array([2]),
        agent_id=np.array([1, 9]),
    )

    assert names.tolist() == ["west", ""]
    assert seqs.tolist() == [2, 0]


def test_spawn_frames_recorded_before_their_meta_take_its_source(tmp_path: Path) -> None:
    share = Path(get_package_share_directory("arena_humansim_msgs")) / "msg"
    store = get_typestore(Stores.ROS2_JAZZY)
    store.register({**get_types_from_msg((share / "AgentMeta.msg").read_text(), AGENT_META_TYPE), **get_types_from_msg((share / "AgentFrame.msg").read_text(), AGENT_FRAME_TYPE)})
    spawned = [Agent(7, 0.0, 1.0, source_name="west", source_seq=2)]
    with Writer(tmp_path / "bag", version=8, storage_plugin=StoragePlugin.MCAP) as writer:
        frame_connection = writer.add_connection("/agent_states", AGENT_FRAME_TYPE, typestore=store)
        meta_connection = writer.add_connection("/agent_meta", AGENT_META_TYPE, typestore=store)
        writer.write(frame_connection, 0, serialize_message(frame_msg(0, spawned, ["sfm"])))
        writer.write(meta_connection, 10, serialize_message(meta_msg(10, spawned, ["sfm"])))
        writer.write(frame_connection, 20, serialize_message(frame_msg(20, spawned, ["sfm"])))

    df = extract_agent_states(tmp_path / "bag")

    assert df["source_name"].tolist() == ["west", "west"]
    assert df["source_seq"].tolist() == [2, 2]
