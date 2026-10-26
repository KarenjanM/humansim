from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import PoseAnchor
from arena_humansim.core.formation.cluster import ClusterFormation
from arena_humansim.utils.types import Pose2D


@dataclass
class _FakeState:
    pose: Pose2D = field(default_factory=Pose2D)


@dataclass
class _FakeAgent:
    state: _FakeState


def _lookup_for(poses: dict[int, Pose2D]) -> Callable[[int], _FakeAgent | None]:
    def inner(aid: int) -> _FakeAgent | None:
        p = poses.get(aid)
        return _FakeAgent(state=_FakeState(pose=p)) if p is not None else None

    return inner


def test_cluster_assigns_explicit_slots() -> None:
    slots = [Pose2D(x=0.0, y=0.0), Pose2D(x=1.0, y=0.0), Pose2D(x=0.0, y=1.0)]
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup_for({1: Pose2D(x=0.9, y=0.1)}),
        slot_poses=slots,
    )
    f.on_join(1)
    targets = f.tick(dt=0.01)
    # Agent 1 is nearest to slot [1.0, 0.0]
    assert targets[1] == Pose2D(x=1.0, y=0.0)


def test_cluster_reassigns_freed_slot_on_leave() -> None:
    slots = [Pose2D(x=0.0, y=0.0), Pose2D(x=1.0, y=0.0)]
    poses = {1: Pose2D(x=0.0, y=0.0), 2: Pose2D(x=2.0, y=0.0)}
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup_for(poses),
        slot_poses=slots,
    )
    f.on_join(1)
    f.on_join(2)
    f.on_leave(1)
    poses[3] = Pose2D(x=0.0, y=0.1)
    f.on_join(3)
    targets = f.tick(dt=0.01)
    assert targets[3] == Pose2D(x=0.0, y=0.0)
    # Original agent 2's slot unchanged
    assert targets[2] == Pose2D(x=1.0, y=0.0)


def test_cluster_generated_slots_capacity_bounded() -> None:
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup_for({1: Pose2D(), 2: Pose2D(), 3: Pose2D(), 4: Pose2D()}),
        slot_poses=None,
        capacity=3,
    )
    for aid in (1, 2, 3, 4):
        f.on_join(aid)
    targets = f.tick(dt=0.01)
    # Only 3 agents placed; 4th has no free slot
    assert len(targets) == 3


def test_explicit_seat_arrives_from_reach_and_parks() -> None:
    seat = Pose2D(x=5.0, y=5.0, theta=-1.5)
    poses = {1: Pose2D(x=5.0, y=4.3)}
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D(x=5.0, y=5.0)),
        agent_lookup=_lookup_for(poses),
        slot_poses=[seat],
    )
    f.on_join(1)
    assert f.arrived(1)
    assert f.seat_of(1) == seat
    poses[1] = Pose2D(x=5.0, y=3.0)
    assert not f.arrived(1)
    assert f.seat_of(1) is None


def test_generated_slots_never_park() -> None:
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup_for({1: Pose2D(x=1.2, y=0.0)}),
        slot_poses=None,
        radius=1.2,
        capacity=4,
    )
    f.on_join(1)
    assert f.arrived(1)
    assert f.seat_of(1) is None


def test_seat_with_approach_parks_once_turned_at_its_standing_point() -> None:
    seat = Pose2D(x=5.0, y=5.0, theta=0.0)
    poses = {1: Pose2D(x=8.0, y=5.0, theta=math.pi)}
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D(x=5.0, y=5.0)),
        agent_lookup=_lookup_for(poses),
        slot_poses=[seat],
        slot_approach=[0.5],
    )
    f.on_join(1)
    assert f.tick(dt=0.01)[1] == Pose2D(x=5.5, y=5.0, theta=0.0)
    assert not f.faces_target(1)
    poses[1] = Pose2D(x=5.6, y=5.1, theta=math.pi)
    assert f.faces_target(1)
    assert not f.arrived(1)
    poses[1] = Pose2D(x=5.6, y=5.1, theta=0.05)
    assert f.seat_of(1) == seat
    poses[1] = seat
    assert f.arrived(1)
    f.on_leave(1)
    poses[1] = Pose2D(x=5.6, y=5.1, theta=math.pi)
    f.on_join(1)
    assert not f.arrived(1)
