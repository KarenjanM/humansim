from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.local_planner import LocalPlanner, _registry
from arena_humansim.local_planner.robot.base import RobotPolicy
from arena_humansim.utils.types import Pose2D, Segments, WallAware

from ._geometry import corridor_walls, single_agent, vertical_wall
from tests.contracts._util import registry_ids


def _is_robot_or_optional(name: str) -> bool:
    try:
        return issubclass(_registry.get(name), RobotPolicy)
    except ImportError:
        return True


_IMPL_IDS = [k for k in registry_ids(_registry) if k != "socialgail" and not _is_robot_or_optional(k)]


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def planner_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def planner(planner_name: str, walls_empty: Segments) -> LocalPlanner:
    p = LocalPlanner.create(planner_name)
    p.set_walls(walls_empty)
    return p


def _is_wall_aware(obj: object) -> bool:
    return isinstance(obj, WallAware) and type(obj).set_walls is not WallAware.set_walls


def test_free_field_goal_seeking(planner: LocalPlanner, single_agent: list[BaseAgent]) -> None:
    goal = Pose2D(x=5.0, y=0.0)
    goals = {single_agent[0].state.agent_id: goal}
    velocities = planner.compute(single_agent, goals, dt=0.1)
    vx, vy = velocities[single_agent[0].state.agent_id]
    agent = single_agent[0]
    dx = goal.x - agent.state.pose.x
    dy = goal.y - agent.state.pose.y
    dot = vx * dx + vy * dy
    assert dot > 0.0, f"velocity ({vx}, {vy}) not directed toward goal; dot={dot}"


def test_monotonic_progress_single_agent(planner_name: str, planner: LocalPlanner, single_agent: list[BaseAgent]) -> None:
    goal = Pose2D(x=5.0, y=0.0)
    agent = single_agent[0]
    aid = agent.state.agent_id
    goals = {aid: goal}

    dt = 0.1
    prev_d = math.hypot(goal.x - agent.state.pose.x, goal.y - agent.state.pose.y)
    for _ in range(5):
        velocities = planner.compute(single_agent, goals, dt=dt)
        vx, vy = velocities[aid]
        if planner_name == "sfm":
            assert math.hypot(vx, vy) >= agent.params.desired_velocity * 0.1, f"sfm speed too low: ({vx}, {vy})"
        agent.state.pose.x += vx * dt
        agent.state.pose.y += vy * dt
        agent.state.velocity = (vx, vy)
        d = math.hypot(goal.x - agent.state.pose.x, goal.y - agent.state.pose.y)
        assert d < prev_d - 1e-4, f"distance did not shrink: prev={prev_d}, cur={d}"
        prev_d = d


def test_zero_velocity_at_goal(planner: LocalPlanner, agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    goal = Pose2D(x=0.0, y=0.0)
    velocities = planner.compute([agent], {agent.state.agent_id: goal}, dt=0.1)
    vx, vy = velocities[agent.state.agent_id]
    mag = math.hypot(vx, vy)
    assert mag < agent.params.desired_velocity * 0.02, f"velocity magnitude {mag} too large at goal"


def test_wall_repulsion(planner_name: str, agent_factory: Callable[..., BaseAgent]) -> None:
    p_free = LocalPlanner.create(planner_name)
    p_free.set_walls([])
    if not _is_wall_aware(p_free):
        pytest.skip(f"{planner_name} is not WallAware")

    p_wall = LocalPlanner.create(planner_name)
    p_wall.set_walls([((0.0, 0.0), (5.0, 0.0))])

    a_free = agent_factory(agent_id=1, x=0.0, y=0.5)
    a_wall = agent_factory(agent_id=1, x=0.0, y=0.5)
    goal = Pose2D(x=5.0, y=-0.5)

    for _ in range(30):
        for planner, agent in ((p_free, a_free), (p_wall, a_wall)):
            vx, vy = planner.compute([agent], {agent.state.agent_id: goal}, dt=0.1)[agent.state.agent_id]
            agent.state.velocity = (vx, vy)
            agent.state.pose.x += vx * 0.1
            agent.state.pose.y += vy * 0.1

    assert a_wall.state.pose.y > a_free.state.pose.y + 1e-3, f"no wall awareness: free={a_free.state.pose}, wall={a_wall.state.pose}"
