from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import (
    SampledParams,
    SampledPerception,
)
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.scenario import (
    ModuleConfig,
    ScenarioConfig,
    SimulationParams,
)
from arena_humansim.utils.types import AgentState, Pose2D, Segments

_ROS_SKIP_REASON = "ROS2 not discoverable - source install/setup.bash to enable"


def _ros_available() -> bool:
    try:
        import rclpy
        import arena_humansim_msgs.srv
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _ros_available():
        return
    skip = pytest.mark.skip(reason=_ROS_SKIP_REASON)
    for item in items:
        path = str(item.path)
        if "/tests/integration/" in path or "/tests/ros/" in path:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def rclpy_context():
    try:
        import rclpy
        import rclpy.node
    except ImportError:
        yield None
        return
    from arena_humansim.utils.loggable import Loggable

    # xdist workers share one DDS domain, so each subscription sees every worker's publishers
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is not None:
        base = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        os.environ["ROS_DOMAIN_ID"] = str(base + 1 + int(worker.removeprefix("gw")))
    rclpy.init()
    node = rclpy.node.Node("pytest_host")
    Loggable.init_logging(node)
    try:
        yield rclpy
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def rng() -> RNG:
    return RNG(42)


@pytest.fixture
def rng_pair() -> tuple[RNG, RNG]:
    return RNG(42), RNG(42)


@pytest.fixture
def walls_empty() -> Segments:
    return []


@pytest.fixture
def walls_simple() -> Segments:
    return [
        ((-5.0, -1.0), (5.0, -1.0)),
        ((-5.0, 1.0), (5.0, 1.0)),
    ]


def _make_sampled_params(name: str = "adult") -> SampledParams:
    return SampledParams(
        name=name,
        desired_velocity=1.1,
        agent_radius=0.25,
        max_velocity=1.5,
        max_acceleration=1.5,
        max_deceleration=2.5,
        min_turning_radius=0.3,
        pivot_angular_velocity=2.0,
        reaction_time=0.4,
        personal_space_min=0.6,
        perception=SampledPerception(vision_range=5.0, vision_fov=180.0),
        local_planner_params={
            "relaxation_time": 0.5,
            "repulsion_strength": 2.1,
            "repulsion_range": 0.3,
            "anisotropy": 0.5,
            "lateral_gain": 0.3,
            "lateral_damping": 1.5,
            "angular_gain": 4.0,
            "angular_damping": 4.0,
        },
        perception_stack=("default",),
        local_planner="sfm",
        global_planner="dijkstra",
        animation="noop",
    )


def _make_agent(agent_id: int, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> BaseAgent:
    state = AgentState(
        agent_id=agent_id,
        pose=Pose2D(x=x, y=y, theta=theta),
        velocity=(0.0, 0.0),
        desired_velocity=1.3,
    )
    params = _make_sampled_params()
    return BaseAgent(
        state=state,
        params=params,
        global_planner=cast(Any, None),
        local_planner=cast(Any, None),
        animation=cast(Any, None),
    )


@pytest.fixture
def agent_factory() -> Callable[..., BaseAgent]:
    def make(agent_id: int, x: float = 0.0, y: float = 0.0) -> BaseAgent:
        return _make_agent(agent_id, x=x, y=y)

    return make


@pytest.fixture
def pool_empty() -> Callable[..., AgentPool]:
    def make(capacity: int = 8) -> AgentPool:
        return AgentPool(capacity=capacity)

    return make


@pytest.fixture
def pool_with_agents() -> Callable[..., AgentPool]:
    def make(n: int = 4) -> AgentPool:
        pool = AgentPool(capacity=max(8, n))
        for i in range(n):
            pool.add_agent(_make_agent(agent_id=i + 1, x=float(i), y=0.0))
        return pool

    return make


@pytest.fixture
def commands_factory() -> Callable[..., dict[int, Any]]:
    def make(agent_ids: list[int] | None = None, target: tuple[float, float] = (5.0, 0.0)) -> dict[int, Any]:
        from arena_humansim.utils.types import HighLevelCommand

        from arena_humansim.utils.types import CommandType

        ids = agent_ids if agent_ids is not None else [1]
        tx, ty = target
        out: dict[int, Any] = {}
        for aid in ids:
            out[int(aid)] = HighLevelCommand(
                agent_id=int(aid),
                type=CommandType.NAVIGATE,
                target_pose=Pose2D(x=float(tx), y=float(ty), theta=0.0),
                desired_velocity=1.3,
                interaction_target=-1,
            )
        return out

    return make


@pytest.fixture
def minimal_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="minimal",
        simulation=SimulationParams(seed=42, dt=0.05, max_ticks=10),
        modules=ModuleConfig(),
    )
