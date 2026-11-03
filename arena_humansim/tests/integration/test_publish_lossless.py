from __future__ import annotations

import time

import pytest

pytest.importorskip("rclpy")

from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy

from tests.integration._capture import spin_until
from tests.integration._helpers import build_manager


def _spawn_agents(mgr, count: int, gx: float = 10.0) -> None:
    req = SpawnAgents.Request()
    for i in range(count):
        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = RosPose2D(x=float(i) * 0.5, y=0.0, theta=0.0)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = 1.3
        msg.radius = 0.3
        msg.agent_type = "adult"
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=gx, y=float(i) * 0.5, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
        req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)


def test_master_mode_no_drops_uniform_spacing() -> None:
    """Every master-mode tick produces exactly one /agent_states with header.stamp = k*dt.

    Drives _master_timer_callback() manually (timer cancelled to avoid double-ticking)
    and subscribes directly with a large queue so the test isolates the publish path
    from recorder file I/O and discovery timing.
    """
    n_ticks = 200
    n_agents = 30
    dt = 0.05

    scenario = ScenarioConfig(
        name="stress_master",
        simulation=SimulationParams(seed=1, dt=dt, max_ticks=0),
        modules=ModuleConfig(),
    )
    mgr = build_manager(
        scenario,
        node_name="test_stress_master",
        extra_params=[Parameter("mode", Parameter.Type.STRING, "master")],
    )

    if mgr._timer is not None:
        mgr._timer.cancel()

    _spawn_agents(mgr, n_agents)

    received: list[AgentFrameMsg] = []
    qos = QoSProfile(depth=n_ticks * 2, reliability=ReliabilityPolicy.RELIABLE)
    mgr.create_subscription(AgentFrameMsg, "agent_states", lambda m: received.append(m), qos)

    executor = SingleThreadedExecutor()
    executor.add_node(mgr)

    try:
        spin_until(executor, lambda: mgr.count_subscribers("agent_states") >= 1, timeout=20.0)

        for _ in range(n_ticks):
            mgr._master_timer_callback()
            executor.spin_once(timeout_sec=0.01)

        deadline = time.monotonic() + 20.0
        while len(received) < n_ticks and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.remove_node(mgr)
        mgr.destroy_node()

    assert len(received) == n_ticks, f"expected {n_ticks} /agent_states, got {len(received)}"

    dt_ns = int(dt * 1e9)
    for i, msg in enumerate(received):
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        assert stamp_ns == i * dt_ns, f"frame {i} header.stamp={stamp_ns}ns, expected {i * dt_ns}ns (k*dt) - missing or reordered tick"
