from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import AgentType, NeedCondition, NeedDist, ParamDist, SequenceDef, StepDef, TransitionDef
from arena_humansim.core.behavior.compiler import compile_agent_behavior
from arena_humansim.core.behavior.nodes import SequenceStateMachine
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import BehaviorTreeMovement, NeedsState, NeedState

if TYPE_CHECKING:
    from arena_humansim.core.agent_manager import AgentManager
    from arena_humansim_msgs.srv import NotifyStimulus


def _alarm_needs(value: float = 0.0) -> NeedsState:
    return NeedsState(needs={"alarm": NeedState(value=value, decay_rate=0.0)})


def test_needs_state_set_clamps_and_ignores_unknown() -> None:
    needs = _alarm_needs(10.0)
    needs.set("alarm", 250.0)
    assert needs.needs["alarm"].value == 100.0
    needs.set("alarm", -5.0)
    assert needs.needs["alarm"].value == 0.0
    needs.set("alarm", 42.5)
    assert needs.needs["alarm"].value == 42.5
    needs.set("hunger", 50.0)
    assert set(needs.needs) == {"alarm"}


def _find_state_machine(root: py_trees.behaviour.Behaviour) -> SequenceStateMachine:
    if isinstance(root, SequenceStateMachine):
        return root
    for child in root.children:
        if isinstance(child, SequenceStateMachine):
            return child
    raise AssertionError("no SequenceStateMachine under root")


def test_alarm_need_preempts_default_into_evacuate(agent_factory: Callable[..., BaseAgent]) -> None:
    agent_type = AgentType(
        name="evacuee",
        mode="behavior_tree",
        needs={"alarm": NeedDist(initial=ParamDist(0.0), decay_rate=ParamDist(0.0))},
        sequences={
            "default": SequenceDef(
                steps={"idle": StepDef(duration=ParamDist(1.0))},
                then="default",
                transitions=(TransitionDef(when={"alarm": NeedCondition(above=50.0)}, goto="evacuate"),),
            ),
            "evacuate": SequenceDef(steps={"idle": StepDef(duration=ParamDist(1.0))}),
        },
    )
    agent = agent_factory(agent_id=1)
    agent.needs = _alarm_needs()
    agent.movement = BehaviorTreeMovement()

    bt = compile_agent_behavior(agent_type, agent, WorldKnowledge(), EventBus(), np.random.default_rng(0), 0.05)
    assert bt is not None
    sm = _find_state_machine(bt.root)

    for _ in range(3):
        bt.tick()
    assert sm._current_name == "default"

    agent.needs.set("alarm", 100.0)
    bt.tick()
    assert sm._current_name == "evacuate"


def _scenario(bt_tick_interval: int = 5) -> ScenarioConfig:
    return ScenarioConfig(
        name="stimulus",
        simulation=SimulationParams(seed=42, dt=0.05, bt_tick_interval=bt_tick_interval, max_ticks=10),
        modules=ModuleConfig(),
    )


@pytest.fixture
def manager_factory(rclpy_context: object) -> Iterator[Callable[..., AgentManager]]:
    try:
        from arena_humansim_msgs.srv import NotifyStimulus
    except ImportError:
        pytest.skip("arena_humansim_msgs built without NotifyStimulus")
    from tests.integration._helpers import build_manager

    created: list[AgentManager] = []

    def make(scenario: ScenarioConfig, node_name: str) -> AgentManager:
        mgr = build_manager(scenario, node_name=node_name)
        created.append(mgr)
        return mgr

    yield make

    for mgr in created:
        mgr.destroy_node()


def _spawn(mgr: AgentManager, n: int) -> list[int]:
    from arena_humansim_msgs.msg import AgentState as AgentStateMsg
    from arena_humansim_msgs.srv import SpawnAgents
    from geometry_msgs.msg import Pose2D as RosPose2D

    req = SpawnAgents.Request()
    for i in range(n):
        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = RosPose2D(x=float(i), y=0.0, theta=0.0)
        msg.desired_velocity = 1.3
        msg.agent_type = "adult"
        req.agents.append(msg)
    resp = mgr._spawn_agents_callback(req, SpawnAgents.Response())
    return list(resp.spawned_ids)


def _notify(mgr: AgentManager, agent_id: int, stimulus: str = "alarm", intensity: float = 1.0) -> NotifyStimulus.Response:
    from arena_humansim_msgs.srv import NotifyStimulus

    req = NotifyStimulus.Request()
    req.agent_id = agent_id
    req.stimulus = stimulus
    req.intensity = intensity
    return mgr._notify_stimulus_callback(req, NotifyStimulus.Response())


def _due_ticks(mgr: AgentManager, aid: int) -> int:
    return int(round(mgr._agents[aid].params.reaction_time / mgr._dt))


def test_notify_applies_after_reaction_time(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario(), node_name="test_notify_delay")
    (aid,) = _spawn(mgr, 1)
    agent = mgr._agents[aid]
    agent.needs = _alarm_needs()

    resp = _notify(mgr, aid, intensity=0.8)
    assert resp.success
    assert resp.message == "queued alarm for 1 agent(s)"
    assert agent.needs.needs["alarm"].value == 0.0

    due = _due_ticks(mgr, aid)
    assert due > 0
    for _ in range(due):
        mgr.tick()
        assert agent.needs.needs["alarm"].value == 0.0
    mgr.tick()
    assert agent.needs.needs["alarm"].value == pytest.approx(80.0)
    assert not mgr._pending_stimuli


def test_notify_without_needs_still_fires_event(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario(bt_tick_interval=1000), node_name="test_notify_no_needs")
    (aid,) = _spawn(mgr, 1)
    mgr._agents[aid].needs = None

    assert _notify(mgr, aid).success
    for _ in range(_due_ticks(mgr, aid) + 1):
        mgr.tick()
    assert mgr._event_bus.has("alarm", aid)


def test_notify_broadcast_targets_every_agent(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario(), node_name="test_notify_broadcast")
    ids = _spawn(mgr, 3)
    for aid in ids:
        mgr._agents[aid].needs = _alarm_needs()

    resp = _notify(mgr, -1, intensity=0.5)
    assert resp.success
    assert resp.message == "queued alarm for 3 agent(s)"
    assert sorted(entry[1] for entry in mgr._pending_stimuli) == sorted(ids)

    for _ in range(max(_due_ticks(mgr, aid) for aid in ids) + 1):
        mgr.tick()
    for aid in ids:
        assert mgr._agents[aid].needs.needs["alarm"].value == pytest.approx(50.0)


def test_notify_unknown_agent_fails(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario(), node_name="test_notify_unknown")
    _spawn(mgr, 1)
    resp = _notify(mgr, 999)
    assert not resp.success
    assert "999" in resp.message
    assert not mgr._pending_stimuli


def test_notify_skips_agent_removed_before_due(manager_factory: Callable[..., AgentManager]) -> None:
    from arena_humansim_msgs.srv import RemoveAgents

    mgr = manager_factory(_scenario(), node_name="test_notify_removed")
    removed, survivor = _spawn(mgr, 2)
    mgr._agents[survivor].needs = _alarm_needs()
    assert _notify(mgr, -1).success
    last_due = max(_due_ticks(mgr, removed), _due_ticks(mgr, survivor))
    req = RemoveAgents.Request()
    req.agent_ids = [removed]
    mgr._remove_agents_callback(req, RemoveAgents.Response())
    for _ in range(last_due + 1):
        mgr.tick()
    assert not mgr._pending_stimuli
    assert mgr._agents[survivor].needs.needs["alarm"].value == pytest.approx(100.0)


def test_reset_clears_pending_stimuli(manager_factory: Callable[..., AgentManager]) -> None:
    from arena_humansim_msgs.srv import ResetSimulation

    mgr = manager_factory(_scenario(), node_name="test_notify_reset")
    (aid,) = _spawn(mgr, 1)
    assert _notify(mgr, aid).success
    assert mgr._pending_stimuli
    mgr._reset_callback(ResetSimulation.Request(), ResetSimulation.Response())
    assert not mgr._pending_stimuli


def test_removing_every_agent_drops_stimuli_queued_for_them(manager_factory: Callable[..., AgentManager]) -> None:
    from arena_humansim_msgs.srv import RemoveAgents

    mgr = manager_factory(_scenario(), node_name="test_notify_remove_all")
    (old,) = _spawn(mgr, 1)
    assert _notify(mgr, old).success
    due = _due_ticks(mgr, old)
    mgr._remove_agents_callback(RemoveAgents.Request(), RemoveAgents.Response())
    assert not mgr._pending_stimuli
    (new,) = _spawn(mgr, 1)
    assert new == old
    mgr._agents[new].needs = _alarm_needs()
    for _ in range(due + 1):
        mgr.tick()
    assert mgr._agents[new].needs.needs["alarm"].value == 0.0
