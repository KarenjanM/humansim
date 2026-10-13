from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.behavior.nodes import SequenceStateMachine
from arena_humansim.core.behavior.nodes.primitives import PatienceWatchdogNode
from arena_humansim.utils.scenario import load_scenario

_SCENARIO = """
name: sequence_tune
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 1, max_ticks: 1000}
agent_types:
  runner:
    extends: adult
    mode: behavior_tree
    desired_velocity: {mean: 1.0, std: 0.0}
    needs:
      alarm: {initial: 0.0, decay_rate: 0.0}
    initial_sequence: calm
    sequences:
      calm:
        steps:
          walk: {kind: go_to, target_pose: {x: 40.0, y: 0.0}}
        then: calm
        transitions:
          - when: {alarm: {above: 50.0}}
            goto: rush
      rush:
        tune:
          speed_scale: 1.5
          max_velocity: 2.6
          perception: {vision_range: 2.0}
          local_planner_params: {repulsion_strength: 0.5}
        steps:
          walk: {kind: go_to, target_pose: {x: 40.0, y: 0.0}, patience: {mean: 0.5}}
        then: rush
        on_failure: rush
        transitions:
          - when: {alarm: {below: 50.0}}
            goto: calm
agents:
  - {agent_id: 1, agent_type: runner, spawn_pose: {x: 0.0, y: 0.0, theta: 0.0}}
"""


def _machine(mgr: AgentManager) -> SequenceStateMachine:
    return next(n for n in mgr._behavior_trees[1].root.iterate() if isinstance(n, SequenceStateMachine))


def _rows(mgr: AgentManager) -> tuple[float, ...]:
    pool = mgr._pool
    i = pool.idx(1)
    planner = mgr._policies[int(pool.policy_idx[i])]
    return (float(pool.desired_vel[i]), float(pool.max_velocity[i]), float(pool.vision_range[i]), float(planner._repulsion_strength[i]))


def test_a_sequence_tune_holds_through_its_restarts_and_leaves_with_it(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    path = tmp_path / "sequence_tune.yaml"
    path.write_text(_SCENARIO)
    mgr = manager_factory(load_scenario(str(path)), node_name="test_sequence_tune")
    for _ in range(5):
        mgr.tick()
    spawn = _rows(mgr)
    params = mgr._agents[1].params
    alarm = mgr._agents[1].needs.needs["alarm"]

    alarm.value = 100.0
    mgr.tick()
    machine = _machine(mgr)
    assert machine._current_name == "rush"
    watchdog = next(n for n in machine._current_node.iterate() if isinstance(n, PatienceWatchdogNode))
    restarts = 0
    elapsed = watchdog._elapsed
    for _ in range(100):
        mgr.tick()
        assert _rows(mgr) == pytest.approx((spawn[0] * 1.5, 2.6, 2.0, 0.5))
        restarts += watchdog._elapsed < elapsed
        elapsed = watchdog._elapsed
    assert restarts >= 5

    alarm.value = 0.0
    mgr.tick()
    assert _rows(mgr) == pytest.approx(spawn)
    assert mgr._agents[1].params == params


_TWIN = """
name: sequence_tune_twin
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 1, max_ticks: 1000}
agent_types:
  waiter:
    extends: adult
    mode: behavior_tree
    desired_velocity: {mean: 1.0, std: 0.0}
    needs:
      dark: {initial: 0.0, decay_rate: 0.0}
    initial_sequence: wait
    sequences:
      wait:
        steps:
          hold: {duration: {mean: 10.0}}
        then: wait
        transitions:
          - when: {dark: {above: 50.0}}
            goto: wait__dark
      wait__dark:
        tune: {perception: {vision_range: 2.0}}
        steps:
          hold: {duration: {mean: 10.0}}
        then: wait__dark
agents:
  - {agent_id: 1, agent_type: waiter, spawn_pose: {x: 0.0, y: 0.0, theta: 0.0}}
"""


def test_a_transition_that_only_retunes_keeps_the_running_step(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    path = tmp_path / "sequence_tune_twin.yaml"
    path.write_text(_TWIN)
    mgr = manager_factory(load_scenario(str(path)), node_name="test_sequence_tune_twin")
    for _ in range(60):
        mgr.tick()
    machine = _machine(mgr)
    hold = machine._current_node
    mgr._agents[1].needs.needs["dark"].value = 100.0
    for _ in range(60):
        mgr.tick()
    assert machine._current_name == "wait__dark" and machine._current_node is hold
    assert mgr._pool.vision_range[mgr._pool.idx(1)] == pytest.approx(2.0)
