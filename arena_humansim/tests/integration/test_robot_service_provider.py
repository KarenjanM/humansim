from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import load_scenario

SCENARIO = """
name: robot_service_provider
simulation: {seed: 3, dt: 0.05, bt_tick_interval: 5, max_ticks: 400}
agent_types:
  seeker:
    extends: adult
    mode: behavior_tree
    initial_sequence: hail
    sequences:
      hail:
        steps:
          ask:
            interaction: SERVICE
            target: water
            wait_for_outcome: true
        then: hail
        on_failure: hail
agents:
  - {agent_id: 1, agent_type: seeker, spawn_pose: {x: -3.0, y: 0.0, theta: 0.0}}
  - agent_id: 0
    kind: 1
    agent_type: adult
    spawn_pose: {x: 0.0, y: 0.0, theta: 3.14159}
    services: [{tag: water}]
"""


def test_pedestrian_reaches_service_of_negative_id_robot(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO)
    mgr = manager_factory(load_scenario(str(path)), node_name="robot_service_provider")
    robot = next(aid for aid, agent in mgr._agents.items() if agent.state.kind == 1)
    assert robot < 0
    for _ in range(300):
        mgr.tick()
    service = [i for i in mgr._interaction_manager.interactions.values() if i.type == int(InteractionType.SERVICE)]
    assert [i.provider for i in service] == [robot]
    assert service[0].contract.formation is not None
    assert 1 in service[0].participants
    seeker = mgr._pool.pos[mgr._pool.idx(1)]
    provider = mgr._pool.pos[mgr._pool.idx(robot)]
    assert math.hypot(*(seeker - provider)) < 2.0
