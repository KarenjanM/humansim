from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import rclpy
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint, Waypoints
from arena_humansim_msgs.srv import (
    AddObstacles,
    AddWalls,
    GetProfile,
    RemoveAgents,
    RemoveWalls,
    ResetSimulation,
    SpawnAgents,
)
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg
from geometry_msgs.msg import Vector3
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


class RosTestSystem:
    def __init__(self, manager: Any, client_node: Node, executor: SingleThreadedExecutor) -> None:
        self.manager = manager
        self.client_node = client_node
        self.executor = executor
        self._clients: dict[str, Any] = {}
        self._sub: Any = None
        self._received: list[AgentFrameMsg] = []

    def client(self, srv_type: Any, name: str) -> Any:
        key = f"{srv_type.__name__}:{name}"
        if key not in self._clients:
            self._clients[key] = self.client_node.create_client(srv_type, name)
        return self._clients[key]

    def call(self, srv_type: Any, name: str, request: Any, timeout: float = 5.0) -> Any:
        cli = self.client(srv_type, name)
        if not cli.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"service {name} not available within {timeout}s")
        future = cli.call_async(request)
        self.executor.spin_until_future_complete(future, timeout_sec=timeout)
        if not future.done():
            future.cancel()
            raise TimeoutError(f"service {name} call did not complete within {timeout}s")
        return future.result()

    def subscribe_agent_states(self, topic: str = "agent_states") -> None:
        if self._sub is not None:
            return
        self._received = []
        self._sub = self.client_node.create_subscription(
            AgentFrameMsg,
            topic,
            lambda msg: self._received.append(msg),
            10,
        )

    def wait_for_agent_states(self, timeout: float = 5.0, tick: bool = False) -> AgentFrameMsg:
        self.subscribe_agent_states()
        deadline = time.monotonic() + timeout
        start_len = len(self._received)
        while time.monotonic() < deadline:
            if tick:
                self.manager.tick()
            self.executor.spin_once(timeout_sec=0.05)
            if len(self._received) > start_len:
                return self._received[-1]
        raise TimeoutError(f"no AgentFrame received within {timeout}s")

    def drain(self, duration: float = 0.2) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def tick_manager(self, n: int = 1) -> None:
        for _ in range(n):
            self.manager.tick()

    def shutdown(self) -> None:
        for cli in self._clients.values():
            try:
                self.client_node.destroy_client(cli)
            except Exception:
                pass
        if self._sub is not None:
            try:
                self.client_node.destroy_subscription(self._sub)
            except Exception:
                pass
        try:
            self.executor.remove_node(self.manager)
        except Exception:
            pass
        try:
            self.executor.remove_node(self.client_node)
        except Exception:
            pass
        try:
            self.manager.destroy_node()
        except Exception:
            pass
        try:
            self.client_node.destroy_node()
        except Exception:
            pass


def make_system(client_node_name: str = "ros_test_client", **node_kwargs: Any) -> RosTestSystem:
    from arena_humansim.core.agent_manager import AgentManager

    manager = AgentManager(**node_kwargs)
    if manager._timer is not None:
        manager._timer.cancel()
    client_node = Node(client_node_name)
    executor = SingleThreadedExecutor()
    executor.add_node(manager)
    executor.add_node(client_node)
    return RosTestSystem(manager=manager, client_node=client_node, executor=executor)


def make_agent_msg(
    agent_id: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    desired_velocity: float = 1.3,
    radius: float = 0.3,
    agent_type: str = "adult",
    waypoints: list[tuple[float, float]] | None = None,
    mode: int = 0,
) -> AgentStateMsg:
    msg = AgentStateMsg()
    msg.agent_id = agent_id
    msg.pose = Pose2DMsg(x=x, y=y, theta=theta)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = desired_velocity
    msg.radius = radius
    msg.agent_type = agent_type
    wps = Waypoints()
    wps.mode = mode
    wp_list: list[tuple[float, float]] = waypoints if waypoints is not None else [(x + 5.0, y)]
    for wx, wy in wp_list:
        w = Waypoint()
        w.pose = Pose2DMsg(x=float(wx), y=float(wy), theta=0.0)
        w.radius = 0.0
        wps.points.append(w)
    msg.waypoints = wps
    return msg


def make_spawn_request(specs: list[dict[str, Any]]) -> SpawnAgents.Request:
    req = SpawnAgents.Request()
    for s in specs:
        req.agents.append(make_agent_msg(**s))
    return req


def make_remove_request(agent_ids: list[int]) -> RemoveAgents.Request:
    req = RemoveAgents.Request()
    req.agent_ids = list(agent_ids)
    return req


def make_reset_request() -> ResetSimulation.Request:
    return ResetSimulation.Request()


def make_add_walls_request(walls: list[tuple[str, tuple[float, float], tuple[float, float]]]) -> AddWalls.Request:
    req = AddWalls.Request()
    for name, start, end in walls:
        req.names.append(name)
        req.starts.append(Point32(x=float(start[0]), y=float(start[1]), z=0.0))
        req.ends.append(Point32(x=float(end[0]), y=float(end[1]), z=0.0))
    return req


def make_remove_walls_request(names: list[str] | None = None) -> RemoveWalls.Request:
    req = RemoveWalls.Request()
    if names is not None:
        req.names = list(names)
    return req


def make_get_profile_request(reset: bool = False) -> GetProfile.Request:
    req = GetProfile.Request()
    req.reset = reset
    return req


def iter_agent_ids(msg: AgentFrameMsg) -> Iterator[int]:
    yield from msg.agent_id


# convenience re-exports
__all__ = [
    "AddObstacles",
    "AddWalls",
    "GetProfile",
    "RemoveAgents",
    "RemoveWalls",
    "ResetSimulation",
    "RosTestSystem",
    "SpawnAgents",
    "iter_agent_ids",
    "make_add_walls_request",
    "make_agent_msg",
    "make_get_profile_request",
    "make_remove_request",
    "make_remove_walls_request",
    "make_reset_request",
    "make_spawn_request",
    "make_system",
]
