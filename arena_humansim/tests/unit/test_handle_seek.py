"""Per-strategy _handle_seek dispatch coverage (NONE / TAG / AGENT / OBJECT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    BehaviorTreeMovement,
    CommandType,
    HighLevelCommand,
    Pose2D,
    SeekSpec,
)


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


class _FakeAgent:
    def __init__(self, agent_id: int, x: float = 0.0, kind: AgentKind = AgentKind.HUMAN) -> None:
        self.state = AgentState(agent_id=agent_id, pose=Pose2D(x=x), kind=int(kind))
        self.params = _FakeParams()
        self.movement = BehaviorTreeMovement()


def _mk_mgr(agents: dict[int, Any], *, world: WorldKnowledge | None = None, visibility: dict[int, set[int]] | None = None) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    vis = visibility if visibility is not None else {aid: set(agents) - {aid} for aid in agents}
    mgr.set_context(
        world_knowledge=world or WorldKnowledge(),
        agent_lookup=lambda aid: agents.get(aid),
        visibility_lookup=lambda aid: vis.get(aid, set()),
    )
    return mgr


def _seek(agent_id: int, itype: InteractionType, *, target: str | int | None = None, offer: bool = False) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent_id,
        type=CommandType.SEEK,
        spec=SeekSpec(interaction_type=itype, target=target, offer=offer),
    )


def test_none_strategy_creates_symmetric_interaction_with_no_peer() -> None:
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO)})
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.participants == [1]
    assert interaction.type == int(InteractionType.TALK_TO)


def test_none_strategy_joins_visible_existing_interaction() -> None:
    agents = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.GROUP_CONVERSATION), 2: _seek(2, InteractionType.GROUP_CONVERSATION)})
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert set(interaction.participants) == {1, 2}


def test_tag_strategy_requires_offer_to_create() -> None:
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents)
    # Seeker (offer=False) without matching provider -> nothing created.
    mgr.update({1: _seek(1, InteractionType.SERVICE, target="water")})
    assert len(mgr.interactions) == 0


def test_tag_strategy_provider_creates_and_seeker_joins() -> None:
    agents = {1: _FakeAgent(1, x=0.0, kind=AgentKind.ROBOT), 2: _FakeAgent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update(
        {
            1: _seek(1, InteractionType.SERVICE, target="water", offer=True),
            2: _seek(2, InteractionType.SERVICE, target="water"),
        }
    )
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.provider == 1
    assert interaction.service_tag == "water"
    assert set(interaction.participants) == {1, 2}


def test_tag_strategy_ignores_mismatched_tag() -> None:
    agents = {1: _FakeAgent(1, x=0.0, kind=AgentKind.ROBOT), 2: _FakeAgent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update(
        {
            1: _seek(1, InteractionType.SERVICE, target="water", offer=True),
            2: _seek(2, InteractionType.SERVICE, target="coffee"),
        }
    )
    # Provider's interaction is "water"; seeker wanted "coffee" -> can't match, can't create (no offer).
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.participants == [1]


def test_agent_strategy_requires_integer_target() -> None:
    agents = {1: _FakeAgent(1), 5: _FakeAgent(5)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.BLOCK, target=5)})
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.target_agent == 5
    assert interaction.participants == [1]


def test_agent_strategy_zero_target_rejected() -> None:
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.BLOCK, target=0)})
    assert len(mgr.interactions) == 0


def test_object_strategy_requires_world_resolution() -> None:
    wk = WorldKnowledge()
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents, world=wk)
    # No matching object in world -> can_create False, no interaction.
    mgr.update({1: _seek(1, InteractionType.USE, target="atm")})
    assert len(mgr.interactions) == 0


def test_object_strategy_resolves_by_type_to_concrete_id() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm_1", type="atm", pose=Pose2D()))
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents, world=wk)
    mgr.update({1: _seek(1, InteractionType.USE, target="atm")})
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    # spec.target was "atm" (type); stored object_id is the concrete resolved id.
    assert interaction.object_id == "atm_1"


def test_object_strategy_dedups_same_object_same_type() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm_1", type="atm", pose=Pose2D()))
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents, world=wk)
    mgr.update({1: _seek(1, InteractionType.USE, target="atm_1")})
    mgr.update({2: _seek(2, InteractionType.USE, target="atm_1")})
    assert len(mgr.interactions) == 1


def test_object_strategy_separate_interactions_for_different_types_same_object() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="kiosk", type="kiosk", pose=Pose2D()))
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents, world=wk)
    mgr.update({1: _seek(1, InteractionType.USE, target="kiosk")})
    mgr.update({2: _seek(2, InteractionType.SIT_ON, target="kiosk")})
    assert len(mgr.interactions) == 2


def test_seek_is_noop_when_agent_already_in_matching_interaction() -> None:
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO)})
    assert len(mgr.interactions) == 1
    mgr.update({1: _seek(1, InteractionType.TALK_TO)})
    assert len(mgr.interactions) == 1


def test_navigate_commands_are_not_consumed_by_interaction_manager() -> None:
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents)
    nav = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=5.0))
    mgr.update({1: nav})
    assert len(mgr.interactions) == 0


def test_two_providers_same_tag_do_not_merge() -> None:
    # Providers must never join another provider's offer, even if visible and same tag.
    # Otherwise the second provider becomes a participant of the first's service.
    agents = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update(
        {
            1: _seek(1, InteractionType.SERVICE, target="escort", offer=True),
            2: _seek(2, InteractionType.SERVICE, target="escort", offer=True),
        }
    )
    assert len(mgr.interactions) == 2
    providers = {i.provider for i in mgr.interactions.values()}
    assert providers == {1, 2}


def test_offers_process_before_seekers_in_same_tick() -> None:
    # Even when shuffled, SEEK with offer=True must be dispatched before plain seekers,
    # otherwise same-tick seekers miss the interaction the provider is about to create.
    agents = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update(
        {
            2: _seek(2, InteractionType.SERVICE, target="water"),
            1: _seek(1, InteractionType.SERVICE, target="water", offer=True),
        }
    )
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert set(interaction.participants) == {1, 2}


def test_symmetric_seek_merges_via_perception_proximity() -> None:
    # End-to-end: two agents facing apart with tight FOV sit within proximity_sense.
    # The perception layer unions proximity into the neighbour CSR, so visibility_lookup
    # reports the peer and both SEEKs converge on one GROUP_CONVERSATION interaction.
    from arena_humansim.core.agents.base import BaseAgent
    from arena_humansim.core.agents.types import SampledParams, SampledPerception
    from arena_humansim.core.pool import AgentPool
    from arena_humansim.perception.default import DefaultPerception

    def _sampled(prox: float) -> SampledParams:
        return SampledParams(
            name="adult",
            desired_velocity=1.1,
            agent_radius=0.25,
            max_velocity=1.5,
            max_acceleration=1.5,
            max_deceleration=2.5,
            min_turning_radius=0.3,
            pivot_angular_velocity=2.0,
            reaction_time=0.4,
            personal_space_min=0.6,
            perception=SampledPerception(vision_range=5.0, vision_fov=1.0, proximity_sense=prox),
            local_planner_params={
                "relaxation_time": 0.5,
                "repulsion_strength": 2.1,
                "repulsion_range": 0.3,
                "anisotropy": 0.5,
            },
        )

    def _real_agent(aid: int, x: float, y: float, theta: float, prox: float) -> BaseAgent:
        st = AgentState(agent_id=aid, pose=Pose2D(x=x, y=y, theta=theta), velocity=(0.0, 0.0), desired_velocity=1.1)
        return BaseAgent(
            state=st,
            params=_sampled(prox),
            global_planner=cast(Any, None),
            local_planner=cast(Any, None),
            animation=cast(Any, None),
        )

    # Both agents face +y with a pinhole FOV; the peer on the x-axis is perpendicular
    # to their heading so the FOV cone excludes it. Proximity sense is the only path in.
    agents = {
        1: _real_agent(1, 0.0, 0.0, theta=1.5707963, prox=1.2),
        2: _real_agent(2, 0.8, 0.0, theta=1.5707963, prox=1.2),
    }
    pool = AgentPool(capacity=4)
    for ag in agents.values():
        pool.add_agent(ag)
    perception = DefaultPerception()
    perception.compute_pool(pool)

    mgr = InteractionManager(RNG(0))
    mgr.set_context(
        world_knowledge=WorldKnowledge(),
        agent_lookup=lambda aid: agents.get(aid),
        visibility_lookup=pool.visible_agent_ids,
    )
    mgr.update({1: _seek(1, InteractionType.GROUP_CONVERSATION), 2: _seek(2, InteractionType.GROUP_CONVERSATION)})
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert set(interaction.participants) == {1, 2}


def test_handle_seek_threads_duration_on_join() -> None:
    # Joining an existing interaction with a duration must record member_durations for the joiner,
    # so queue promotion (or contract refresh) can honour each member's individual timer.
    agents = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=1.0)}
    world = WorldKnowledge()
    world.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    mgr = _mk_mgr(agents, world=world)
    mgr.update({1: HighLevelCommand(agent_id=1, type=CommandType.SEEK, spec=SeekSpec(interaction_type=InteractionType.USE, target="atm", duration=0.5))})
    iid = next(iter(mgr.interactions))
    mgr.interactions[iid].outcome = 1  # ACTIVE
    mgr.update({2: HighLevelCommand(agent_id=2, type=CommandType.SEEK, spec=SeekSpec(interaction_type=InteractionType.USE, target="atm", duration=1.25))})
    assert mgr.interactions[iid].member_durations[2] == pytest.approx(1.25)
