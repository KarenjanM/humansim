from __future__ import annotations

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.viz import _STATIC_NS
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from visualization_msgs.msg import MarkerArray

from ._capture import settle, spin_until, subscribe
from ._helpers import build_manager


def _spawn_pair(mgr: AgentManager) -> None:
    req = SpawnAgents.Request()
    for i in range(2):
        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = RosPose2D(x=float(i) * 2.0, y=0.0, theta=0.0)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = 1.3
        msg.radius = 0.0
        msg.agent_type = "adult"
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=float(i) * 2.0 + 10.0, y=0.0, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
        req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    assert len(resp.spawned_ids) == 2


def test_tick_markers_full_detail_ticks_cleanly(rclpy_context: object, minimal_scenario: ScenarioConfig) -> None:  # noqa: ARG001
    extra = [Parameter("publish_markers", Parameter.Type.INTEGER, 2)]
    mgr = build_manager(minimal_scenario, node_name="test_tick_markers_full", extra_params=extra)
    try:
        received = subscribe(mgr, MarkerArray, "viz", 100)
        executor = SingleThreadedExecutor()
        executor.add_node(mgr)
        spin_until(executor, lambda: mgr._marker_pub.watched)
        _spawn_pair(mgr)

        for _ in range(20):
            mgr.tick()

        assert mgr._tick_count == 20
        assert mgr._publish_markers == 2
        spin_until(executor, lambda: len(received) > 0)
        assert any(ns not in _STATIC_NS for ns, _mid in mgr._marker_pub._scene)
        executor.remove_node(mgr)
    finally:
        mgr.destroy_node()


def test_tick_markers_skips_module_markers_without_a_subscriber(rclpy_context: object, minimal_scenario: ScenarioConfig) -> None:  # noqa: ARG001
    extra = [Parameter("publish_markers", Parameter.Type.INTEGER, 2)]
    mgr = build_manager(minimal_scenario, node_name="test_tick_markers_unwatched", extra_params=extra)
    try:
        executor = SingleThreadedExecutor()
        executor.add_node(mgr)
        settle(executor)
        _spawn_pair(mgr)
        start = mgr._tick_count

        for _ in range(20):
            mgr.tick()

        assert mgr._tick_count - start == 20
        assert not mgr._marker_pub.watched
        assert all(ns in _STATIC_NS for ns, _mid in mgr._marker_pub._scene)
        executor.remove_node(mgr)
    finally:
        mgr.destroy_node()
