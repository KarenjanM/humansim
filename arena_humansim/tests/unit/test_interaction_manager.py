from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from dataclasses import dataclass
from typing import Any

from arena_humansim.core.formation import Formation
from arena_humansim.core.formation.anchor import AgentAnchor
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import CommandType, InteractionManager
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    AnchorKind,
    BehaviorTreeMovement,
    FormationSpec,
    HighLevelCommand,
    InteractionContract,
    InteractionOutcome,
    InteractionState,
    Pose2D,
    SeekSpec,
)


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


class _FakeBTAgent:
    def __init__(self, agent_id: int = 0, x: float = 0.0, y: float = 0.0, kind: AgentKind = AgentKind.HUMAN) -> None:
        self.state = AgentState(agent_id=agent_id, desired_velocity=1.0, pose=Pose2D(x=x, y=y), kind=int(kind))
        self.movement = BehaviorTreeMovement()
        self.params = _FakeParams()


def _fake_bt_agent(agent_id: int = 0, x: float = 0.0, y: float = 0.0) -> Any:
    return _FakeBTAgent(agent_id=agent_id, x=x, y=y)


def _fake_robot_agent(agent_id: int = 0, x: float = 0.0, y: float = 0.0) -> Any:
    return _FakeBTAgent(agent_id=agent_id, x=x, y=y, kind=AgentKind.ROBOT)


def _seek_cmd(
    agent_id: int,
    interaction_type: InteractionType,
    *,
    target: str | int | None = None,
    offer: bool = False,
    min_participants: int | None = None,
    max_participants: int | None = None,
    queueable: bool | None = None,
    duration: float | None = None,
) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent_id,
        type=CommandType.SEEK,
        target_pose=Pose2D(),
        desired_velocity=1.0,
        spec=SeekSpec(
            interaction_type=interaction_type,
            target=target,
            offer=offer,
            min_participants=min_participants,
            max_participants=max_participants,
            queueable=queueable,
            duration=duration,
        ),
    )


def _seed_interaction(
    mgr: InteractionManager,
    agent_id: int,
    itype: InteractionType,
    *,
    target: str | int | None = None,
    offer: bool = False,
    duration: float | None = None,
) -> int:
    spec = SeekSpec(
        interaction_type=itype,
        target=target,
        offer=offer,
        duration=duration,
    )
    interaction = mgr._create_interaction(creator_id=agent_id, spec=spec)
    if duration is not None:
        interaction.member_durations[agent_id] = duration
    return interaction.id


def _mk_mgr(
    agents: dict[int, Any] | None = None,
    *,
    world: WorldKnowledge | None = None,
    visibility: dict[int, set[int]] | None = None,
    seed: int = 0,
) -> InteractionManager:
    mgr = InteractionManager(RNG(seed))
    agents = agents or {}
    vis = visibility if visibility is not None else {aid: set(agents) - {aid} for aid in agents}
    mgr.set_context(
        world_knowledge=world or WorldKnowledge(),
        agent_lookup=lambda aid: agents.get(aid),
        visibility_lookup=lambda aid: vis.get(aid, set()),
    )
    return mgr


def test_accept_adds_participant() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)})
    iid = _seed_interaction(mgr, 1, InteractionType.GROUP_CONVERSATION)
    assert mgr.accept(2, iid) is True
    interaction = mgr.interactions[iid]
    assert 2 in interaction.participants
    assert 2 in interaction.contract.current_participants


def test_accept_when_full_queues_if_queueable() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm")
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue
    assert 2 not in mgr.interactions[iid].participants
    assert mgr.is_in_queue(2)


def test_accept_when_full_rejects_if_not_queueable() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2), 3: _fake_bt_agent(3)})
    iid = _seed_interaction(mgr, 1, InteractionType.TALK_TO)
    assert mgr.accept(2, iid) is True
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(3, iid) is False
    assert 3 not in mgr.interactions[iid].participants
    assert 3 not in mgr.interactions[iid].contract.queue


def test_block_target_agent_scoping() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 3: _fake_bt_agent(3), 5: _fake_bt_agent(5)})
    iid = _seed_interaction(mgr, 1, InteractionType.BLOCK, target=5)
    assert mgr.accept(3, iid) is False
    assert 3 not in mgr.interactions[iid].participants
    assert mgr.accept(5, iid) is True
    assert 5 in mgr.interactions[iid].participants


def test_stop_below_min_tears_down_with_interrupted() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)})
    iid = _seed_interaction(mgr, 1, InteractionType.TALK_TO)
    assert mgr.accept(2, iid) is True
    assert mgr.stop(1, iid) is None
    mgr._prune_ended_interactions()
    assert iid not in mgr.interactions


def test_stop_removes_from_queue_cleanly() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm")
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue
    mgr.stop(2, iid)
    assert 2 not in mgr.interactions[iid].contract.queue
    assert not mgr.is_in_queue(2)
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE


def test_tick_promotes_next_from_queue_when_slot_opens() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)})
    # Directly build a queued interaction state to exercise fallback promotion path.
    from arena_humansim.core.interaction_kinds import MembershipRole

    contract = InteractionContract(
        type=int(InteractionType.QUEUE_USE),
        min_participants=1,
        max_participants=2,
        queueable=True,
        current_participants=[1],
        queue=[2],
    )
    interaction = InteractionState(
        id=0,
        type=int(InteractionType.QUEUE_USE),
        contract=contract,
        participants=[1],
        outcome=InteractionOutcome.ACTIVE,
    )
    mgr.interactions[0] = interaction
    mgr.next_interaction_id = 1
    mgr._interactions_by_type.setdefault(int(InteractionType.QUEUE_USE), set()).add(0)
    mgr._add_membership(1, 0, MembershipRole.PARTICIPANT)
    mgr._add_membership(2, 0, MembershipRole.QUEUED)

    mgr.update({})

    assert 2 in interaction.participants
    assert 2 in interaction.contract.current_participants
    assert 2 not in interaction.contract.queue
    assert not mgr.is_in_queue(2)
    assert mgr.is_in_interaction(2)


def test_tick_durations_time_out_to_completed() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)})
    iid = _seed_interaction(mgr, 1, InteractionType.TALK_TO, duration=0.5)
    assert mgr.accept(2, iid) is True
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    mgr.interactions[iid].contract.formation = None

    mgr.update({}, dt=0.3)
    assert iid in mgr.interactions
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE

    mgr.update({}, dt=0.3)
    assert iid not in mgr.interactions


def test_force_stop_clears_interactions_and_queues() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2), 7: _fake_bt_agent(7)}, world=wk)
    iid_a = _seed_interaction(mgr, 1, InteractionType.GROUP_CONVERSATION)
    mgr.interactions[iid_a].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(7, iid_a) is True
    iid_q = _seed_interaction(mgr, 2, InteractionType.QUEUE_USE, target="atm")
    mgr.interactions[iid_q].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(7, iid_q) is True
    assert mgr.is_in_interaction(7)
    assert mgr.is_in_queue(7)

    mgr.force_stop(7)

    assert not mgr.is_in_interaction(7)
    assert not mgr.is_in_queue(7)
    assert 7 not in mgr.interactions[iid_a].participants
    assert 7 not in mgr.interactions[iid_q].contract.queue


def test_queue_length_for_object_sums_across_interactions() -> None:
    mgr = InteractionManager(RNG(0))
    c1 = InteractionContract(type=int(InteractionType.QUEUE_USE), min_participants=1, max_participants=1, queueable=True, current_participants=[1], queue=[10, 11])
    i1 = InteractionState(id=0, type=int(InteractionType.QUEUE_USE), contract=c1, participants=[1], object_id="bench", outcome=InteractionOutcome.ACTIVE)
    c2 = InteractionContract(type=int(InteractionType.QUEUE_USE), min_participants=1, max_participants=1, queueable=True, current_participants=[2], queue=[12])
    i2 = InteractionState(id=1, type=int(InteractionType.QUEUE_USE), contract=c2, participants=[2], object_id="bench", outcome=InteractionOutcome.ACTIVE)
    c3 = InteractionContract(type=int(InteractionType.QUEUE_USE), min_participants=1, max_participants=1, queueable=True, current_participants=[3], queue=[99])
    i3 = InteractionState(id=2, type=int(InteractionType.QUEUE_USE), contract=c3, participants=[3], object_id="other", outcome=InteractionOutcome.ACTIVE)
    mgr.interactions[0] = i1
    mgr.interactions[1] = i2
    mgr.interactions[2] = i3

    assert mgr.queue_length_for_object("bench") == 3
    assert mgr.queue_length_for_object("other") == 1
    assert mgr.queue_length_for_object("missing") == 0


def test_update_with_seeded_rng_is_deterministic() -> None:
    def run(seed: int) -> list[int]:
        agents = {i: _fake_bt_agent(i, x=float(i) * 0.5) for i in (1, 2, 3, 4, 5)}
        mgr = _mk_mgr(agents, seed=seed)
        iid1 = _seed_interaction(mgr, 1, InteractionType.GROUP_CONVERSATION)
        mgr.interactions[iid1].outcome = InteractionOutcome.ACTIVE
        iid2 = _seed_interaction(mgr, 2, InteractionType.GROUP_CONVERSATION)
        mgr.interactions[iid2].outcome = InteractionOutcome.ACTIVE
        cmds = {
            3: _seek_cmd(3, InteractionType.GROUP_CONVERSATION),
            4: _seek_cmd(4, InteractionType.GROUP_CONVERSATION),
            5: _seek_cmd(5, InteractionType.GROUP_CONVERSATION),
        }
        mgr.update(cmds)
        out: list[int] = []
        for iid in sorted(mgr.interactions.keys()):
            out.extend(mgr.interactions[iid].participants)
        return out

    assert run(123) == run(123)


def test_update_order_of_pedestrian_commands_ignores_robot_command() -> None:
    def run(seed: int, robot: bool) -> list[list[int]]:
        agents = {i: _fake_bt_agent(i, x=float(i) * 0.5) for i in (1, 2, 3, 4, 5)}
        agents[-1] = _fake_robot_agent(-1, x=50.0)
        mgr = _mk_mgr(agents, seed=seed)
        _seed_interaction(mgr, 1, InteractionType.TALK_TO)
        cmds = {aid: _seek_cmd(aid, InteractionType.TALK_TO) for aid in (3, 4, 5)}
        extra = [HighLevelCommand(agent_id=-1, type=CommandType.STOP)] if robot else None
        mgr.update(cmds, extra_commands=extra)
        return [list(mgr.interactions[i].participants) for i in sorted(mgr.interactions)]

    for seed in range(20):
        assert run(seed, robot=True) == run(seed, robot=False), f"seed {seed}"


def test_service_offer_records_provider() -> None:
    mgr = _mk_mgr({42: _fake_robot_agent(42)})
    iid = _seed_interaction(mgr, 42, InteractionType.SERVICE, target="water", offer=True)
    interaction = mgr.interactions[iid]
    assert interaction.provider == 42
    assert interaction.service_tag == "water"


def test_block_on_robot_admits_only_its_negative_id() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2), -1: _fake_robot_agent(-1)})
    iid = _seed_interaction(mgr, 1, InteractionType.BLOCK, target=-1)
    assert mgr.interactions[iid].target_agent == -1
    assert mgr.accept(2, iid) is False
    assert mgr.accept(-1, iid) is True


def test_agent_anchor_resolves_negative_robot_id() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), -1: _fake_robot_agent(-1, x=2.0)})
    iid = _seed_interaction(mgr, 1, InteractionType.GROUP_CONVERSATION)
    anchor = mgr._anchor_from_spec(FormationSpec(type="cluster", anchor_kind=AnchorKind.AGENT, anchor_ref="-1"), mgr.interactions[iid])
    assert isinstance(anchor, AgentAnchor)
    assert anchor.agent_id == -1
    assert mgr._anchor_from_spec(FormationSpec(type="cluster", anchor_kind=AnchorKind.AGENT, anchor_ref="0"), mgr.interactions[iid]) is None


def test_talk_caps_at_two_participants() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2), 3: _fake_bt_agent(3)})
    iid = _seed_interaction(mgr, 1, InteractionType.TALK_TO)
    assert mgr.accept(2, iid) is True
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(3, iid) is False
    assert len(mgr.interactions[iid].participants) == 2


def test_queue_use_gets_fifo_access_and_line_formation() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm")
    contract = mgr.interactions[iid].contract
    assert contract.access is not None
    assert contract.formation is not None


def test_group_conversation_gets_f_formation() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1)})
    iid = _seed_interaction(mgr, 1, InteractionType.GROUP_CONVERSATION)
    contract = mgr.interactions[iid].contract
    assert contract.formation is not None
    assert contract.access is None


def test_talk_to_gets_dyad_and_no_access() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1)})
    iid = _seed_interaction(mgr, 1, InteractionType.TALK_TO)
    contract = mgr.interactions[iid].contract
    assert contract.formation is not None
    assert contract.access is None


def test_sit_on_is_queueable_by_default() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="chair", type="chair", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.SIT_ON, target="chair")
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue


def test_second_seek_with_same_object_joins_existing() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    mgr.update({1: _seek_cmd(1, InteractionType.USE, target="atm")})
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    assert mgr.interactions[iid].participants == [1]

    mgr.update({2: _seek_cmd(2, InteractionType.USE, target="atm")})
    assert len(mgr.interactions) == 1
    assert 2 in mgr.interactions[iid].contract.queue
    assert 2 not in mgr.interactions[iid].participants


def test_seek_dedups_only_on_matching_object_id() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm_1", type="atm", pose=Pose2D()))
    wk.add_object(WorldObject(object_id="atm_2", type="atm", pose=Pose2D(x=10.0)))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    mgr.update({1: _seek_cmd(1, InteractionType.USE, target="atm_1")})
    mgr.update({2: _seek_cmd(2, InteractionType.USE, target="atm_2")})
    assert len(mgr.interactions) == 2


def test_seek_dedups_only_on_matching_type() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="kiosk", type="kiosk", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    mgr.update({1: _seek_cmd(1, InteractionType.USE, target="kiosk")})
    mgr.update({2: _seek_cmd(2, InteractionType.SIT_ON, target="kiosk")})
    assert len(mgr.interactions) == 2


def test_duration_expiry_promotes_queue_instead_of_teardown() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm", duration=0.5)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue

    mgr.update({}, dt=0.6)
    assert iid in mgr.interactions
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE
    assert 1 not in mgr.interactions[iid].participants
    assert 2 in mgr.interactions[iid].participants
    assert mgr.interactions[iid].contract.queue == []
    assert mgr.interactions[iid].contract.elapsed == 0.0


def test_duration_expiry_tears_down_when_queue_empty() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm", duration=0.3)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE

    mgr.update({}, dt=0.4)
    assert iid not in mgr.interactions


def test_multiple_promotions_serial_service() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2), 3: _fake_bt_agent(3)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm", duration=0.2)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid)
    assert mgr.accept(3, iid)

    mgr.update({}, dt=0.25)
    assert 2 in mgr.interactions[iid].participants
    assert 3 in mgr.interactions[iid].contract.queue

    mgr.update({}, dt=0.25)
    assert 3 in mgr.interactions[iid].participants
    assert mgr.interactions[iid].contract.queue == []

    mgr.update({}, dt=0.25)
    assert iid not in mgr.interactions


def test_release_participant_signals_completed_to_released_agent() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    agents = {1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}
    mgr = _mk_mgr(agents, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm", duration=0.5)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue

    mgr.update({}, dt=0.6)

    assert 1 not in mgr.interactions[iid].participants
    assert 2 in mgr.interactions[iid].participants
    assert agents[1].movement.last_outcome == InteractionOutcome.COMPLETED


def test_teardown_signals_interrupted_to_queued_agents() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    agents = {1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}
    mgr = _mk_mgr(agents, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.QUEUE_USE, target="atm")
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue

    mgr.stop(1, iid)

    assert mgr.interactions[iid].outcome == InteractionOutcome.INTERRUPTED
    assert agents[2].movement.last_outcome == InteractionOutcome.INTERRUPTED


def test_queue_promotion_uses_promoted_agents_own_duration() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    mgr.update({1: _seek_cmd(1, InteractionType.USE, target="atm", duration=0.5)})
    iid = next(iter(mgr.interactions))
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    mgr.update({2: _seek_cmd(2, InteractionType.USE, target="atm", duration=1.0)})
    assert 2 in mgr.interactions[iid].contract.queue
    assert mgr.interactions[iid].member_durations[1] == pytest.approx(0.5)
    assert mgr.interactions[iid].member_durations[2] == pytest.approx(1.0)

    mgr.update({}, dt=0.6)
    assert 2 in mgr.interactions[iid].participants
    assert mgr.interactions[iid].contract.duration == pytest.approx(1.0)
    assert mgr.interactions[iid].contract.elapsed == pytest.approx(0.0)
    assert 1 not in mgr.interactions[iid].member_durations

    mgr.update({}, dt=0.7)
    assert iid in mgr.interactions
    assert mgr.interactions[iid].contract.elapsed == pytest.approx(0.7)

    mgr.update({}, dt=0.4)
    assert iid not in mgr.interactions


def test_object_bound_finder_reindexes_after_teardown() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm_a", type="atm", pose=Pose2D()))
    wk.add_object(WorldObject(object_id="atm_b", type="atm", pose=Pose2D(x=10.0)))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2), 3: _fake_bt_agent(3)}, world=wk)
    mgr.update({1: _seek_cmd(1, InteractionType.USE, target="atm_a")})
    mgr.update({2: _seek_cmd(2, InteractionType.USE, target="atm_b")})
    assert len(mgr.interactions) == 2

    iid_a = mgr._interaction_by_object_type.get(("atm_a", int(InteractionType.USE)))
    assert iid_a is not None
    assert mgr.interactions[iid_a].participants == [1]

    mgr._teardown(iid_a, int(InteractionOutcome.COMPLETED))
    mgr._prune_ended_interactions()
    assert mgr._interaction_by_object_type.get(("atm_a", int(InteractionType.USE))) is None

    mgr.update({3: _seek_cmd(3, InteractionType.USE, target="atm_a")})
    iid_a2 = mgr._interaction_by_object_type.get(("atm_a", int(InteractionType.USE)))
    assert iid_a2 is not None and iid_a2 != iid_a
    assert mgr.interactions[iid_a2].participants == [3]


class _FakeFormation(Formation):
    def __init__(self) -> None:
        self._arrived: dict[int, bool] = {}

    def set_arrived(self, agent_id: int, value: bool) -> None:
        self._arrived[agent_id] = value

    def arrived(self, agent_id: int) -> bool:
        return self._arrived.get(agent_id, False)

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        self._arrived.setdefault(agent_id, False)

    def on_leave(self, agent_id: int) -> None:
        self._arrived.pop(agent_id, None)

    def tick(self, _dt: float) -> dict[int, Pose2D]:
        return {}


def test_duration_does_not_advance_until_arrived() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1)}, world=wk)
    mgr.update({1: _seek_cmd(1, InteractionType.USE, target="atm", duration=0.5)})
    interaction = next(iter(mgr.interactions.values()))
    interaction.contract.duration = 0.5
    fake = _FakeFormation()
    fake.on_join(1)
    interaction.contract.formation = fake
    interaction.outcome = InteractionOutcome.ACTIVE

    mgr.update({}, dt=0.2)
    mgr.update({}, dt=0.2)
    mgr.update({}, dt=0.2)
    assert interaction.contract.elapsed == 0.0

    fake.set_arrived(1, True)
    mgr.update({}, dt=0.2)
    assert interaction.contract.elapsed == pytest.approx(0.2)


def test_duration_expiry_rotates_fifo_queue() -> None:
    from arena_humansim.core.access.fifo_queue import FIFOQueue
    from arena_humansim.core.interaction_kinds import MembershipRole

    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2), 3: _fake_bt_agent(3)})
    contract = InteractionContract(
        type=int(InteractionType.USE),
        min_participants=1,
        max_participants=1,
        queueable=True,
        current_participants=[1],
        queue=[2, 3],
        duration=0.5,
    )
    contract.access = FIFOQueue()
    interaction = InteractionState(
        id=0,
        type=int(InteractionType.USE),
        contract=contract,
        participants=[1],
        outcome=InteractionOutcome.ACTIVE,
        object_id="atm",
    )
    mgr.interactions[0] = interaction
    mgr.next_interaction_id = 1
    mgr._interactions_by_type.setdefault(int(InteractionType.USE), set()).add(0)
    mgr._add_membership(1, 0, MembershipRole.PARTICIPANT)
    mgr._add_membership(2, 0, MembershipRole.QUEUED)
    mgr._add_membership(3, 0, MembershipRole.QUEUED)
    mgr._interaction_by_object_type[("atm", int(InteractionType.USE))] = 0

    mgr.update({}, dt=0.6)

    assert interaction.participants == [2]
    assert interaction.contract.queue == [3]
    assert mgr.is_in_interaction(2)
    assert not mgr.is_in_interaction(1)

    mgr.update({}, dt=0.6)

    assert interaction.participants == [3]
    assert interaction.contract.queue == []
    assert not mgr.is_in_interaction(2)


def test_symmetric_pairs_visible_unbound() -> None:
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek_cmd(1, InteractionType.TALK_TO), 2: _seek_cmd(2, InteractionType.TALK_TO)})
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    assert set(mgr.interactions[iid].participants) == {1, 2}


def test_symmetric_skips_invisible() -> None:
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=100.0)}
    mgr = _mk_mgr(agents, visibility={1: set(), 2: set()})
    cmd_a = _seek_cmd(1, InteractionType.TALK_TO)
    cmd_b = _seek_cmd(2, InteractionType.TALK_TO)

    mgr.update({1: cmd_a, 2: cmd_b})
    # Two separate interactions created: each agent couldn't see the other.
    assert len(mgr.interactions) == 2


def test_symmetric_joins_visible_bound_group() -> None:
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=0.5), 3: _fake_bt_agent(3, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek_cmd(1, InteractionType.GROUP_CONVERSATION), 2: _seek_cmd(2, InteractionType.GROUP_CONVERSATION)})
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))

    mgr.update({3: _seek_cmd(3, InteractionType.GROUP_CONVERSATION)})
    assert len(mgr.interactions) == 1
    assert set(mgr.interactions[iid].participants) == {1, 2, 3}


def test_forming_interaction_does_not_drive_motion() -> None:
    # Regression: centroid-anchored f_formation on a 1p FORMING chased the solo agent east.
    agents = {1: _fake_bt_agent(1, x=0.0, y=0.0)}
    mgr = _mk_mgr(agents)
    mgr.seek(1, SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION))
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    assert mgr.interactions[iid].outcome == InteractionOutcome.FORMING

    # Sanity: agent has no motion command yet.
    agents[1].movement.command = None

    for _ in range(5):
        mgr.update({}, dt=0.05)

    assert agents[1].movement.command is None, f"FORMING interaction wrote a motion command: {agents[1].movement.command}. Formations must only drive motion for ACTIVE interactions."


def test_late_arriver_with_own_forming_migrates_to_existing_active_group() -> None:
    # Regression: agent with own 1p FORMING short-circuited seek and never scanned for peers.
    visible_ids: dict[int, set[int]] = {1: set(), 2: {3}, 3: {2}}
    agents = {
        1: _fake_bt_agent(1, x=5.0, y=0.0),
        2: _fake_bt_agent(2, x=0.0, y=0.0),
        3: _fake_bt_agent(3, x=0.3, y=0.0),
    }
    mgr = InteractionManager(RNG(0))
    mgr.set_context(
        agent_lookup=lambda aid: agents.get(aid),  # type: ignore[arg-type]
        visibility_lookup=lambda aid: visible_ids.get(aid, set()),
    )

    # 2 and 3 see each other and form an ACTIVE group.
    mgr.seek(2, SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION))
    mgr.seek(3, SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION))
    active_iid = next(iid for iid, i in mgr.interactions.items() if i.outcome == InteractionOutcome.ACTIVE)

    # Agent 1 is out of visibility; seeks and creates own 1p FORMING.
    mgr.seek(1, SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION))
    own_iid = next(iid for iid, i in mgr.interactions.items() if 1 in i.participants and i.outcome == InteractionOutcome.FORMING)
    assert own_iid != active_iid

    # Now agent 1 walks closer, gains visibility of the group. Re-seek must migrate
    # agent 1 from their own FORMING into the ACTIVE group.
    visible_ids[1] = {2, 3}
    result = mgr.seek(1, SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION))
    assert result == active_iid, f"agent 1 should have migrated into ACTIVE iid={active_iid}, got {result}"
    assert 1 in mgr.interactions[active_iid].participants
    # Own FORMING emptied - its outcome marks it ENDED for pruning.
    assert own_iid not in mgr.interactions or mgr.interactions[own_iid].outcome in _ENDED
    # BT must not have received INTERRUPTED - migration is silent.
    assert agents[1].movement.last_outcome is None


_ENDED = (InteractionOutcome.INTERRUPTED, InteractionOutcome.CANCELED, InteractionOutcome.COMPLETED)


def test_symmetric_forming_survives_idle_updates_for_late_peer() -> None:
    # Regression: staggered arrivals failed to bootstrap because non-BT updates pruned the FORMING.
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=1.0)}
    mgr = _mk_mgr(agents)

    mgr.seek(1, SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION))
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    assert mgr.interactions[iid].outcome == InteractionOutcome.FORMING
    assert mgr.interactions[iid].participants == [1]

    for _ in range(20):
        mgr.update({}, dt=0.05)
    assert len(mgr.interactions) == 1, "lone 1p FORMING was torn down between seeks - late peer can never pair"
    assert mgr.interactions[iid].outcome == InteractionOutcome.FORMING

    mgr.seek(2, SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION))
    assert len(mgr.interactions) == 1
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE
    assert set(mgr.interactions[iid].participants) == {1, 2}


def test_symmetric_skips_full_interactions() -> None:
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=0.5), 3: _fake_bt_agent(3, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek_cmd(1, InteractionType.TALK_TO), 2: _seek_cmd(2, InteractionType.TALK_TO)})
    assert len(mgr.interactions) == 1

    mgr.update({3: _seek_cmd(3, InteractionType.TALK_TO)})
    # Bound TALK_TO is full -> agent 3 can't join it; creates its own lone interaction.
    assert 3 not in next(iter(mgr.interactions.values())).participants


def test_seek_idempotent_re_emit() -> None:
    agents = {1: _fake_bt_agent(1)}
    mgr = _mk_mgr(agents)
    cmd = _seek_cmd(1, InteractionType.TALK_TO)

    mgr.update({1: cmd})
    mgr.update({1: cmd})
    mgr.update({1: cmd})
    assert len(mgr.interactions) == 1
    assert mgr.interactions[next(iter(mgr.interactions))].participants == [1]


def test_anchorless_sit_on_returns_no_interaction_without_object() -> None:
    agents = {1: _fake_bt_agent(1, x=7.0, y=3.0)}
    mgr = _mk_mgr(agents)
    # OBJECT handle requires a target string + resolvable world object; with no world_knowledge, can_create is False.
    mgr.update({1: _seek_cmd(1, InteractionType.SIT_ON, target="chair")})
    assert len(mgr.interactions) == 0


def test_service_matcher_binds_visible_robot_and_human() -> None:
    agents = {1: _fake_robot_agent(1, x=0.0), 2: _fake_bt_agent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    cmd_robot = _seek_cmd(1, InteractionType.SERVICE, target="water", offer=True, max_participants=2)
    cmd_human = _seek_cmd(2, InteractionType.SERVICE, target="water")

    mgr.update({1: cmd_robot, 2: cmd_human})

    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.type == int(InteractionType.SERVICE)
    assert set(interaction.participants) == {1, 2}
    assert interaction.provider == 1
    assert interaction.contract.max_participants == 2


def test_service_fifo_queues_overflow() -> None:
    agents = {
        1: _fake_robot_agent(1, x=0.0),
        2: _fake_bt_agent(2, x=1.0),
        3: _fake_bt_agent(3, x=1.5),
    }
    mgr = _mk_mgr(agents)
    mgr.update(
        {
            1: _seek_cmd(1, InteractionType.SERVICE, target="water", offer=True, max_participants=1),
            2: _seek_cmd(2, InteractionType.SERVICE, target="water"),
            3: _seek_cmd(3, InteractionType.SERVICE, target="water"),
        }
    )

    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.contract.max_participants == 1
    participants = set(interaction.participants)
    queue = set(interaction.contract.queue)
    assert 1 in participants
    assert len(participants) + len(queue) == 3
    assert len(queue) >= 1

    from arena_humansim.core.access.fifo_queue import FIFOQueue

    assert isinstance(interaction.contract.access, FIFOQueue)


def test_service_two_robots_same_tag_parallel_interactions() -> None:
    agents = {
        1: _fake_robot_agent(1, x=0.0),
        2: _fake_robot_agent(2, x=100.0),
        3: _fake_bt_agent(3, x=1.0),
        4: _fake_bt_agent(4, x=101.0),
    }
    visibility = {1: {3}, 2: {4}, 3: {1}, 4: {2}}
    mgr = _mk_mgr(agents, visibility=visibility)
    mgr.update(
        {
            1: _seek_cmd(1, InteractionType.SERVICE, target="water", offer=True, max_participants=2),
            2: _seek_cmd(2, InteractionType.SERVICE, target="water", offer=True, max_participants=2),
            3: _seek_cmd(3, InteractionType.SERVICE, target="water"),
            4: _seek_cmd(4, InteractionType.SERVICE, target="water"),
        }
    )

    assert len(mgr.interactions) == 2
    pairs = {frozenset(i.participants) for i in mgr.interactions.values()}
    assert pairs == {frozenset({1, 3}), frozenset({2, 4})}


def test_service_max_participants_threads_from_spec() -> None:
    agents = {1: _fake_robot_agent(1, x=0.0), 2: _fake_bt_agent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update(
        {
            1: _seek_cmd(1, InteractionType.SERVICE, target="water", offer=True, max_participants=3),
            2: _seek_cmd(2, InteractionType.SERVICE, target="water"),
        }
    )
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.contract.max_participants == 3

    agents2 = {10: _fake_robot_agent(10, x=0.0), 20: _fake_bt_agent(20, x=1.0)}
    mgr2 = _mk_mgr(agents2)
    mgr2.update(
        {
            10: _seek_cmd(10, InteractionType.SERVICE, target="water", offer=True, max_participants=None),
            20: _seek_cmd(20, InteractionType.SERVICE, target="water"),
        }
    )
    assert len(mgr2.interactions) == 1
    interaction2 = next(iter(mgr2.interactions.values()))
    assert interaction2.contract.max_participants == -1


def test_reset_clears_every_index() -> None:
    # Regression: interactions.clear() alone left stale ids in the indices the scan helpers walk.
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}, world=wk)
    iid = _seed_interaction(mgr, 1, InteractionType.GROUP_CONVERSATION)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    _seed_interaction(mgr, 2, InteractionType.QUEUE_USE, target="atm")
    mgr.update({}, dt=0.05)
    assert mgr._interactions_by_type
    assert mgr._agent_membership
    assert mgr._interaction_by_object_type
    assert mgr._formation_targets

    mgr.reset()

    assert mgr.interactions == {}
    assert mgr._interactions_by_type == {}
    assert mgr._agent_membership == {}
    assert mgr._interaction_by_object_type == {}
    assert mgr._formation_targets == {}
    assert mgr._current_departed == set()


def test_reset_keeps_interaction_ids_monotonic() -> None:
    mgr = _mk_mgr({1: _fake_bt_agent(1)})
    _seed_interaction(mgr, 1, InteractionType.GROUP_CONVERSATION)
    seen = mgr.next_interaction_id

    mgr.reset()

    assert mgr.next_interaction_id == seen
