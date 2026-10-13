from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import py_trees
import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.agents import spawn_params
from arena_humansim.core.behavior.nodes import SequenceStateMachine
from arena_humansim.core.behavior.nodes.primitives import PatienceWatchdogNode
from arena_humansim.utils.scenario import load_scenario
from rclpy.parameter import Parameter

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
          desired_velocity: {mul: 1.5}
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


def _load(manager_factory: Callable[..., AgentManager], tmp_path: Path, text: str, name: str) -> AgentManager:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return manager_factory(load_scenario(str(path)), node_name=f"test_{name}")


def test_a_sequence_tune_holds_through_its_restarts_and_leaves_with_it(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = _load(manager_factory, tmp_path, _SCENARIO, "sequence_tune")
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


_CONDITIONAL = """
name: conditional_tune
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 1, max_ticks: 1000}
agent_types:
  waiter:
    extends: adult
    mode: behavior_tree
    desired_velocity: {mean: 1.0, std: 0.0}
    needs:
      dark: {initial: 0.0, decay_rate: 0.0}
    tunes:
      - when: {dark: {above: 50.0}}
        tune: {perception: {vision_range: 2.0}, desired_velocity: {mul: 0.5}}
    initial_sequence: wait
    sequences:
      wait:
        steps:
          hold: {duration: {mean: 10.0}}
        then: wait
agents:
  - {agent_id: 1, agent_type: waiter, spawn_pose: {x: 0.0, y: 0.0, theta: 0.0}}
"""


def test_a_conditional_tune_retunes_the_running_step_in_place(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = _load(manager_factory, tmp_path, _CONDITIONAL, "conditional_tune")
    for _ in range(60):
        mgr.tick()
    spawn = _rows(mgr)
    machine = _machine(mgr)
    hold = machine._current_node
    dark = mgr._agents[1].needs.needs["dark"]
    dark.value = 100.0
    for _ in range(60):
        mgr.tick()
        assert _rows(mgr)[2] == pytest.approx(2.0) and _rows(mgr)[0] == pytest.approx(spawn[0] * 0.5)
    assert machine._current_name == "wait" and machine._current_node is hold and hold.status == py_trees.common.Status.RUNNING
    dark.value = 0.0
    mgr.tick()
    assert _rows(mgr) == pytest.approx(spawn)


_STEP = """
name: step_tune
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 1, max_ticks: 1000}
agent_types:
  walker:
    extends: adult
    mode: behavior_tree
    desired_velocity: {mean: 1.0, std: 0.0}
    initial_sequence: walk
    sequences:
      walk:
        steps:
          near: {kind: go_to, target_pose: {x: 1.0, y: 0.0}, tune: {perception: {vision_range: 2.0}}}
          far: {kind: go_to, target_pose: {x: 40.0, y: 0.0}}
        then: walk
agents:
  - {agent_id: 1, agent_type: walker, spawn_pose: {x: 0.0, y: 0.0, theta: 0.0}}
"""


def test_a_step_tune_ends_with_its_step(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = _load(manager_factory, tmp_path, _STEP, "step_tune")
    mgr.tick()
    spawn_range = mgr._agents[1].tunes.base.perception.vision_range
    for _ in range(5):
        mgr.tick()
        assert _rows(mgr)[2] == pytest.approx(2.0)
    ticks_tuned = 5
    for _ in range(200):
        mgr.tick()
        if _rows(mgr)[2] != pytest.approx(2.0):
            break
        ticks_tuned += 1
    else:
        pytest.fail("the near step never ended")
    assert _rows(mgr)[2] == pytest.approx(spawn_range) and mgr._agents[1].tunes.base is None
    for _ in range(40):
        mgr.tick()
        assert _rows(mgr)[2] == pytest.approx(spawn_range)
    assert ticks_tuned < 100


_ENDS = """
name: tree_end
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 1, max_ticks: 1000}
agent_types:
  walker:
    extends: adult
    mode: behavior_tree
    desired_velocity: {mean: 1.0, std: 0.0}
    initial_sequence: walk
    sequences:
      walk:
        tune: {max_velocity: 2.6}
        steps:
          near: {kind: go_to, target_pose: {x: 1.0, y: 0.0}, tune: {perception: {vision_range: 2.0}}}
        then: nowhere
agents:
  - {agent_id: 1, agent_type: walker, spawn_pose: {x: 0.0, y: 0.0, theta: 0.0}}
"""


def test_a_tree_that_ends_releases_every_key(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = _load(manager_factory, tmp_path, _ENDS, "tree_end")
    mgr.tick()
    spawn = mgr._agents[1].tunes.base
    assert spawn is not None and set(mgr._agents[1].tunes.active) == {"sequence", "walk/near/tune"}
    for _ in range(200):
        mgr.tick()
        if mgr._behavior_trees[1].root.status == py_trees.common.Status.FAILURE:
            break
    else:
        pytest.fail("the tree never ended")
    assert mgr._agents[1].tunes.base is None and mgr._agents[1].params == spawn
    assert _rows(mgr)[1:3] == pytest.approx((spawn.max_velocity, spawn.perception.vision_range))


def test_a_planner_mean_change_moves_the_spawn_params_under_a_live_tune(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = _load(manager_factory, tmp_path, _SCENARIO, "reseat_tune")
    agent = mgr._agents[1]
    for _ in range(5):
        mgr.tick()
    spawn = agent.params.local_planner_params["repulsion_strength"]
    agent.needs.needs["alarm"].value = 100.0
    mgr.tick()
    assert _rows(mgr)[3] == pytest.approx(0.5)
    mean = mgr._lp_dist(agent, "repulsion_strength").mean
    (result,) = mgr.set_parameters([Parameter("local_planner.repulsion_strength", Parameter.Type.DOUBLE, mean + 1.0)])
    assert result.successful
    mgr._apply_parameters()
    assert spawn_params(agent).local_planner_params["repulsion_strength"] == pytest.approx(spawn + 1.0)
    mgr.tick()
    assert _rows(mgr)[3] == pytest.approx(0.5)
    agent.needs.needs["alarm"].value = 0.0
    mgr.tick()
    assert agent.tunes.base is None and agent.params.local_planner_params["repulsion_strength"] == pytest.approx(spawn + 1.0)
    assert _rows(mgr)[3] == pytest.approx(spawn + 1.0)


_SAMPLED_SPEED = """
name: sampled_speed
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 1, max_ticks: 1000}
agent_types:
  slow:
    extends: adult
    desired_velocity: {mean: 0.7, std: 0.0}
agents:
  - {agent_id: 1, agent_type: slow, spawn_pose: {x: 0.0, y: 0.0, theta: 0.0}, desired_velocity: 0.0}
"""


def test_a_spawn_without_a_speed_walks_at_the_sampled_one(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = _load(manager_factory, tmp_path, _SAMPLED_SPEED, "sampled_speed")
    mgr.tick()
    agent = mgr._agents[1]
    assert agent.state.desired_velocity == agent.params.desired_velocity == pytest.approx(0.7)
    assert _rows(mgr)[0] == pytest.approx(0.7)
