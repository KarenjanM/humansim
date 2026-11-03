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


def test_spawn_agents_round_trip(system: RosTestSystem) -> None:
    specs = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": -2.0, "y": 3.0},
    ]
    req = make_spawn_request(specs)
    resp = system.call(SpawnAgents, "spawn_agents", req)

    assert resp is not None
    assert resp.success is True
    assert len(resp.spawned_ids) == len(specs)
    for aid in resp.spawned_ids:
        assert aid > 0


def test_spawned_ids_appear_in_agent_states_topic(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    specs = [{"x": float(i), "y": 0.0} for i in range(3)]
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    expected = set(resp.spawned_ids)

    system.subscribe_agent_states()
    msg = system.wait_for_agent_states(timeout=20.0, tick=True)

    published_ids = set(msg.agent_id)
    assert expected.issubset(published_ids), f"expected {expected} subset of {published_ids}"


def test_spawn_accepts_explicit_agent_id(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    specs = [{"agent_id": 42, "x": 0.0, "y": 0.0}]
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    assert resp.success is True
    assert 42 in resp.spawned_ids


def test_spawn_robot_kind_and_policy(system: RosTestSystem) -> None:
    from arena_humansim_msgs.msg import AgentState as AgentStateMsg
    from arena_humansim_msgs.msg import Waypoints
    from arena_humansim_msgs.srv import SpawnAgents as SpawnAgentsSrv
    from geometry_msgs.msg import Pose2D as Pose2DMsg
    from geometry_msgs.msg import Vector3

    system.call(RemoveAgents, "remove_agents", make_remove_request([]))

    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = Pose2DMsg(x=0.0, y=0.0, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.0
    msg.radius = 0.3
    msg.agent_type = "robot_alpha"
    msg.kind = AgentStateMsg.KIND_ROBOT
    msg.policy = ""
    msg.waypoints = Waypoints()

    req = SpawnAgentsSrv.Request()
    req.agents.append(msg)
    resp = system.call(SpawnAgentsSrv, "spawn_agents", req)
    assert resp.success is True
    assert len(resp.spawned_ids) == 1
    aid = resp.spawned_ids[0]

    pool = system.manager._pool
    idx = pool._id_to_idx[aid]
    assert int(pool.kind[idx]) == 1
    assert int(pool.policy_idx[idx]) == -1
    assert system.manager._agent_name_to_id.get("robot_alpha") == aid
