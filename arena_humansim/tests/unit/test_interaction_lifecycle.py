"""Lifecycle edges (ACTIVATED / HOLD_ONSET / RELEASED / INTERRUPTED) and the contact-mode control arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import (
    CONTACT_ENABLED,
    CONTACT_LOCOMOTION_ONLY,
    InteractionManager,
    LifecycleEdge,
)
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, HighLevelCommand, Pose2D, SeekSpec

DT = 0.05


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


@dataclass
class _FakeState:
    agent_id: int = 0
    pose: Pose2D = field(default_factory=Pose2D)
    desired_velocity: float = 1.2
    kind: int = 0


@dataclass
class _FakeAgent:
    state: _FakeState
    params: _FakeParams = field(default_factory=_FakeParams)
    movement: BehaviorTreeMovement = field(default_factory=BehaviorTreeMovement)


def _pair(x_a: float = 0.0, x_b: float = 2.0) -> dict[int, _FakeAgent]:
    return {1: _FakeAgent(_FakeState(agent_id=1, pose=Pose2D(x=x_a))), 2: _FakeAgent(_FakeState(agent_id=2, pose=Pose2D(x=x_b)))}


def _mgr(agents: dict[int, _FakeAgent], mode: str = CONTACT_ENABLED, standing: float | None = None) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=WorldKnowledge(), agent_lookup=agents.get, visibility_lookup=lambda aid: set(agents) - {aid})
    mgr.set_contact_mode(mode, standing)
    return mgr


def _seek(agents: dict[int, _FakeAgent], itype: InteractionType, duration: float | None = None) -> dict[int, HighLevelCommand]:
    return {aid: HighLevelCommand(agent_id=aid, type=CommandType.SEEK, spec=SeekSpec(interaction_type=itype, duration=duration)) for aid in agents}


def _move_to_slots(mgr: InteractionManager, agents: dict[int, _FakeAgent]) -> None:
    for aid, agent in agents.items():
        target = mgr.formation_target(aid)
        assert target is not None
        agent.state.pose = Pose2D(x=target.x, y=target.y, theta=target.theta)


def _edges(mgr: InteractionManager) -> list[LifecycleEdge]:
    return [e.edge for e in mgr.drain_edges()]


def test_activation_then_hold_onset_waits_for_arrival() -> None:
    agents = _pair(0.0, 4.0)
    mgr = _mgr(agents)
    mgr.update(_seek(agents, InteractionType.HUG, duration=1.0), dt=DT)
    assert _edges(mgr) == [LifecycleEdge.ACTIVATED]
    iid = next(iter(mgr.interactions))
    mgr.update({}, dt=DT)
    assert _edges(mgr) == []
    assert not mgr.is_holding(mgr.interactions[iid])
    assert mgr.interactions[iid].contract.elapsed == 0.0

    _move_to_slots(mgr, agents)
    mgr.update({}, dt=DT)
    edges = mgr.drain_edges()
    assert [e.edge for e in edges] == [LifecycleEdge.HOLD_ONSET]
    assert edges[0].interaction_id == iid and set(edges[0].participants) == {1, 2}
    assert mgr.is_holding(mgr.interactions[iid])


def test_release_fires_after_the_hold_duration() -> None:
    agents = _pair(0.0, 4.0)
    mgr = _mgr(agents)
    mgr.update(_seek(agents, InteractionType.SHAKE_HAND, duration=0.5), dt=DT)
    _move_to_slots(mgr, agents)
    ticks_to_release = None
    seen: list[LifecycleEdge] = []
    for k in range(40):
        mgr.update({}, dt=DT)
        new = _edges(mgr)
        seen.extend(new)
        if LifecycleEdge.RELEASED in new:
            ticks_to_release = k
            break
    assert seen == [LifecycleEdge.ACTIVATED, LifecycleEdge.HOLD_ONSET, LifecycleEdge.RELEASED]
    # the onset tick counts the first dt, so the release lands duration/dt ticks after it
    assert ticks_to_release == pytest.approx(0.5 / DT - 1, abs=1)
    assert not mgr.interactions


def test_member_leaving_an_active_pair_fires_interrupted() -> None:
    agents = _pair()
    mgr = _mgr(agents)
    mgr.update(_seek(agents, InteractionType.HUG), dt=DT)
    iid = next(iter(mgr.interactions))
    mgr.drain_edges()
    mgr.stop(1, iid)
    edges = mgr.drain_edges()
    assert [e.edge for e in edges] == [LifecycleEdge.INTERRUPTED]


def test_solo_forming_interaction_emits_nothing_on_teardown() -> None:
    agents = {1: _pair()[1]}
    mgr = _mgr(agents)
    mgr.update(_seek(agents, InteractionType.HUG), dt=DT)
    iid = next(iter(mgr.interactions))
    mgr.stop(1, iid)
    assert _edges(mgr) == []


def test_reset_cancels_live_interactions() -> None:
    agents = _pair()
    mgr = _mgr(agents)
    mgr.update(_seek(agents, InteractionType.HUG), dt=DT)
    iid = next(iter(mgr.interactions))
    mgr.reset()
    edges = mgr.drain_edges()
    assert [e.edge for e in edges] == [LifecycleEdge.ACTIVATED, LifecycleEdge.CANCELED]
    assert edges[1].interaction_id == iid and set(edges[1].participants) == {1, 2}
    assert not mgr.interactions
    assert _edges(mgr) == []


@pytest.mark.parametrize("itype, contact_sep", [(InteractionType.HUG, 0.3), (InteractionType.SHAKE_HAND, 0.8)])
def test_locomotion_only_holds_at_standing_distance(itype: InteractionType, contact_sep: float) -> None:
    for mode, expected in ((CONTACT_ENABLED, contact_sep), (CONTACT_LOCOMOTION_ONLY, 1.5)):
        agents = _pair(0.0, 4.0)
        mgr = _mgr(agents, mode, standing=1.5)
        mgr.update(_seek(agents, itype), dt=DT)
        a, b = mgr.formation_target(1), mgr.formation_target(2)
        assert a is not None and b is not None
        assert abs(b.x - a.x) == pytest.approx(expected)


def test_locomotion_only_never_shows_contact_but_still_holds() -> None:
    agents = _pair(0.0, 4.0)
    mgr = _mgr(agents, CONTACT_LOCOMOTION_ONLY)
    mgr.update(_seek(agents, InteractionType.HUG, duration=1.0), dt=DT)
    iid = next(iter(mgr.interactions))
    _move_to_slots(mgr, agents)
    mgr.update({}, dt=DT)
    interaction = mgr.interactions[iid]
    assert mgr.is_holding(interaction)
    assert not mgr.shows_contact(interaction)


def test_enabled_shows_contact_only_while_holding() -> None:
    agents = _pair(0.0, 4.0)
    mgr = _mgr(agents)
    mgr.update(_seek(agents, InteractionType.HUG, duration=1.0), dt=DT)
    interaction = next(iter(mgr.interactions.values()))
    assert not mgr.shows_contact(interaction)
    _move_to_slots(mgr, agents)
    mgr.update({}, dt=DT)
    assert mgr.shows_contact(interaction)


def test_non_contact_kinds_ignore_the_contact_mode() -> None:
    agents = _pair(0.0, 4.0)
    mgr = _mgr(agents, CONTACT_LOCOMOTION_ONLY, standing=3.0)
    mgr.update(_seek(agents, InteractionType.TALK_TO), dt=DT)
    a, b = mgr.formation_target(1), mgr.formation_target(2)
    assert a is not None and b is not None
    assert abs(b.x - a.x) == pytest.approx(1.2)  # the dyad default


def test_bad_contact_mode_is_rejected() -> None:
    mgr = _mgr({})
    with pytest.raises(ValueError):
        mgr.set_contact_mode("sometimes")
    with pytest.raises(ValueError):
        mgr.set_contact_mode(CONTACT_LOCOMOTION_ONLY, 0.0)
