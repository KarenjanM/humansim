from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("rclpy")

import py_trees
from arena_humansim.core.agents import Tunes
from arena_humansim.core.agents.types import NeedCondition, SequenceDef, TransitionDef
from arena_humansim.core.behavior.nodes import SequenceStateMachine
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    BehaviorTreeMovement,
    InteractionOutcome,
    NeedsState,
    NeedState,
    Pose2D,
    SeekSpec,
)


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


class _FakeAgent:
    def __init__(self, agent_id: int, x: float = 0.0, y: float = 0.0) -> None:
        self.state = AgentState(agent_id=agent_id, pose=Pose2D(x=x, y=y), kind=int(AgentKind.HUMAN))
        self.params = _FakeParams()
        self.movement = BehaviorTreeMovement()
        self.needs: NeedsState | None = None
        self.tunes = Tunes()


def _mk_mgr(agents: dict[int, Any]) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(
        agent_lookup=lambda aid: agents.get(aid),  # type: ignore[arg-type]
        visibility_lookup=lambda aid: set(agents) - {aid},
    )
    return mgr


def _seed_active(mgr: InteractionManager, participants: list[int], itype: InteractionType) -> int:
    spec = SeekSpec(interaction_type=itype)
    interaction = mgr._create_interaction(creator_id=participants[0], spec=spec)
    for pid in participants[1:]:
        mgr.accept(pid, interaction.id)
    interaction.outcome = InteractionOutcome.ACTIVE
    interaction.contract.formation = None
    return interaction.id


def _mk_sm(agent: _FakeAgent, sequences: dict[str, py_trees.behaviour.Behaviour], sequence_defs: dict[str, SequenceDef], initial: str, im: InteractionManager | None = None) -> SequenceStateMachine:
    return SequenceStateMachine(
        name="sm",
        sequences=sequences,
        sequence_defs=sequence_defs,
        initial=initial,
        agent=agent,  # type: ignore[arg-type]
        im=im,
    )


def test_goto_writes_force_stop_command() -> None:
    agent_id = 7
    other_id = 8
    agents: dict[int, Any] = {
        agent_id: _FakeAgent(agent_id),
        other_id: _FakeAgent(other_id, x=0.2),
    }
    mgr = _mk_mgr(agents)
    _seed_active(mgr, [agent_id, other_id], InteractionType.GROUP_CONVERSATION)
    assert mgr.is_bound(agent_id)

    agent: _FakeAgent = agents[agent_id]
    sequences = {
        "seq_a": py_trees.behaviours.Running(name="seq_a"),
        "seq_b": py_trees.behaviours.Running(name="seq_b"),
    }
    sequence_defs = {
        "seq_a": SequenceDef(steps={}),
        "seq_b": SequenceDef(steps={}),
    }
    sm = _mk_sm(agent, sequences, sequence_defs, initial="seq_a", im=mgr)

    sm._goto("seq_b")

    assert not mgr.is_bound(agent_id)
    assert agent.movement.last_outcome == InteractionOutcome.INTERRUPTED


def test_need_transition_evicts_agent_from_existing_interaction() -> None:
    agent_id = 1
    other_id = 2
    agents: dict[int, Any] = {
        agent_id: _FakeAgent(agent_id, x=0.0),
        other_id: _FakeAgent(other_id, x=0.2),
    }
    mgr = _mk_mgr(agents)
    _seed_active(mgr, [agent_id, other_id], InteractionType.GROUP_CONVERSATION)

    assert mgr.is_bound(agent_id)

    agent: _FakeAgent = agents[agent_id]
    agent.needs = NeedsState(needs={"thirst": NeedState(value=50.0, decay_rate=0.0)})

    chat_body = py_trees.behaviours.Running(name="chat_body")
    drink_body = py_trees.behaviours.Running(name="drink_body")
    sequences = {"chat": chat_body, "drink": drink_body}
    sequence_defs = {
        "chat": SequenceDef(
            steps={},
            transitions=(TransitionDef(when={"thirst": NeedCondition(below=30.0)}, goto="drink"),),
        ),
        "drink": SequenceDef(steps={}),
    }
    sm = _mk_sm(agent, sequences, sequence_defs, initial="chat", im=mgr)
    sm.initialise()

    # Drop thirst below the transition threshold, then tick.
    agent.needs.needs["thirst"].value = 25.0
    sm.update()

    # _goto called im.force_stop directly - agent is immediately evicted.
    assert not mgr.is_bound(agent_id)
    assert agent_id not in mgr._agent_membership or not mgr._agent_membership[agent_id]


def test_self_loop_via_then_does_not_evict() -> None:
    agent_id = 5
    other_id = 6
    agents: dict[int, Any] = {
        agent_id: _FakeAgent(agent_id, x=0.0),
        other_id: _FakeAgent(other_id, x=0.2),
    }
    mgr = _mk_mgr(agents)
    _seed_active(mgr, [agent_id, other_id], InteractionType.GROUP_CONVERSATION)

    assert mgr.is_bound(agent_id)

    agent: _FakeAgent = agents[agent_id]

    sequences = {"chat": py_trees.behaviours.Success(name="chat")}
    sequence_defs = {"chat": SequenceDef(steps={}, then="chat")}
    sm = _mk_sm(agent, sequences, sequence_defs, initial="chat", im=mgr)
    sm.initialise()

    sm.update()

    assert mgr.is_bound(agent_id), "self-loop must keep the agent in the group"


def test_success_transition_via_then_also_evicts() -> None:
    agent_id = 3
    other_id = 4
    agents: dict[int, Any] = {
        agent_id: _FakeAgent(agent_id, x=0.0),
        other_id: _FakeAgent(other_id, x=0.2),
    }
    mgr = _mk_mgr(agents)
    _seed_active(mgr, [agent_id, other_id], InteractionType.GROUP_CONVERSATION)

    assert mgr.is_bound(agent_id)

    agent: _FakeAgent = agents[agent_id]

    # A behaviour that returns SUCCESS immediately triggers the `then:` path.
    sequences = {
        "first": py_trees.behaviours.Success(name="first"),
        "second": py_trees.behaviours.Running(name="second"),
    }
    sequence_defs = {
        "first": SequenceDef(steps={}, then="second"),
        "second": SequenceDef(steps={}),
    }
    sm = _mk_sm(agent, sequences, sequence_defs, initial="first", im=mgr)
    sm.initialise()

    sm.update()

    # _goto called im.force_stop directly - agent is immediately evicted.
    assert not mgr.is_bound(agent_id)
    assert agent.movement.last_outcome == InteractionOutcome.INTERRUPTED
