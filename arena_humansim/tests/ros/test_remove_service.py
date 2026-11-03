from __future__ import annotations

import pytest

from tests.ros._helpers import (
    RemoveAgents,
    RosTestSystem,
    SpawnAgents,
    make_remove_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    return ros_system


def test_remove_subset_of_agents(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    specs = [{"x": float(i), "y": 0.0} for i in range(4)]
    spawn_resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    spawned = list(spawn_resp.spawned_ids)
    to_remove = spawned[:2]
    remaining = set(spawned[2:])

    remove_resp = system.call(RemoveAgents, "remove_agents", make_remove_request(to_remove))
    assert remove_resp.success is True

    system.subscribe_agent_states()
    msg = system.wait_for_agent_states(timeout=20.0, tick=True)

    published_ids = set(msg.agent_id)
    assert remaining.issubset(published_ids)
    for rid in to_remove:
        assert rid not in published_ids


def test_remove_all_with_empty_list(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    specs = [{"x": float(i), "y": 0.0} for i in range(3)]
    system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))

    resp = system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    assert resp.success is True

    system.subscribe_agent_states()
    msg = system.wait_for_agent_states(timeout=20.0, tick=True)
    assert len(msg.agent_id) == 0
