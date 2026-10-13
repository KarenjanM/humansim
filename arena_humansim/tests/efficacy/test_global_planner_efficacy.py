from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner import GlobalPlanner, _registry
from arena_humansim.utils.types import Pose2D, Segments

from ._geometry import vertical_wall
from tests.contracts._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def planner_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def planner(planner_name: str, walls_empty: Segments) -> GlobalPlanner:
    p = GlobalPlanner.create(planner_name)
    p.set_walls(walls_empty)
    return p


def _segments_intersect(a1: Pose2D, a2: Pose2D, b1: tuple[float, float], b2: tuple[float, float]) -> bool:
    def _cross(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = _cross(b1[0], b1[1], b2[0], b2[1], a1.x, a1.y)
    d2 = _cross(b1[0], b1[1], b2[0], b2[1], a2.x, a2.y)
    d3 = _cross(a1.x, a1.y, a2.x, a2.y, b1[0], b1[1])
    d4 = _cross(a1.x, a1.y, a2.x, a2.y, b2[0], b2[1])
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def test_goal_approaches_target(planner: GlobalPlanner, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    cmds = commands_factory(agent_ids=[agent.state.agent_id], target=(10.0, 0.0))
    goals = planner.compute([agent], cmds)
    goal = goals[agent.state.agent_id]
    target_dist = math.hypot(10.0 - goal.x, 0.0 - goal.y)
    origin_dist = math.hypot(10.0 - 0.0, 0.0 - 0.0)
    assert target_dist <= origin_dist + 1e-6, f"goal {goal} farther from target than origin ({target_dist} > {origin_dist})"


def test_cached_path_terminates_near_target(planner_name: str, vertical_wall: Segments, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    p = GlobalPlanner.create(planner_name)
    p.set_walls(vertical_wall)
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    target = (10.0, 0.0)
    cmds = commands_factory(agent_ids=[agent.state.agent_id], target=target)
    p.compute([agent], cmds)
    paths = p.get_cached_paths()
    if not paths or agent.state.agent_id not in paths or not paths[agent.state.agent_id]:
        pytest.skip(f"{planner_name} did not expose a cached path for this input")
    wps = paths[agent.state.agent_id]
    last = wps[-1]
    d = math.hypot(last.x - target[0], last.y - target[1])
    grid_resolution = getattr(p, "_resolution", 0.2)
    assert d <= grid_resolution + 1e-6, f"path end {last} not near target {target}: d={d}"


def test_path_avoids_walls(planner_name: str, vertical_wall: Segments, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    p = GlobalPlanner.create(planner_name)
    p.set_walls(vertical_wall)
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    cmds = commands_factory(agent_ids=[agent.state.agent_id], target=(10.0, 0.0))
    p.compute([agent], cmds)
    paths = p.get_cached_paths()
    if not paths or agent.state.agent_id not in paths or len(paths[agent.state.agent_id]) < 2:
        pytest.skip(f"{planner_name} did not expose a usable cached path")
    wps = paths[agent.state.agent_id][:10]
    wall = vertical_wall[0]
    for i in range(len(wps) - 1):
        crosses = _segments_intersect(wps[i], wps[i + 1], wall[0], wall[1])
        assert not crosses, f"path segment {wps[i]} -> {wps[i + 1]} crosses wall {wall}"


def test_responsiveness_to_target_change(planner_name: str, walls_empty: Segments, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    p = GlobalPlanner.create(planner_name)
    p.set_walls(walls_empty)
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    aid = agent.state.agent_id

    cmds_a = commands_factory(agent_ids=[aid], target=(5.0, 0.0))
    goal_a = p.compute([agent], cmds_a)[aid]

    cmds_b = commands_factory(agent_ids=[aid], target=(-5.0, 0.0))
    goal_b = p.compute([agent], cmds_b)[aid]

    assert (goal_a.x, goal_a.y) != (goal_b.x, goal_b.y), f"planner ignored target change: {goal_a} == {goal_b}"


def test_dijkstra_astar_agree_on_simple_grid(vertical_wall: Segments, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    if "dijkstra" not in _IMPL_IDS or "astar" not in _IMPL_IDS:
        pytest.skip("both dijkstra and astar must be registered for cross-impl check")

    agent_d = agent_factory(agent_id=1, x=0.0, y=0.0)
    agent_a = agent_factory(agent_id=1, x=0.0, y=0.0)
    cmds = commands_factory(agent_ids=[1], target=(10.0, 0.0))

    pd = GlobalPlanner.create("dijkstra")
    pd.set_walls(vertical_wall)
    pd.compute([agent_d], cmds)
    path_d = pd.get_cached_paths().get(1, [])

    pa = GlobalPlanner.create("astar")
    pa.set_walls(vertical_wall)
    pa.compute([agent_a], cmds)
    path_a = pa.get_cached_paths().get(1, [])

    if not path_d or not path_a:
        pytest.skip("one of the planners did not return a cached path")

    def _length(wps: list[Pose2D]) -> float:
        total = 0.0
        for i in range(len(wps) - 1):
            total += math.hypot(wps[i + 1].x - wps[i].x, wps[i + 1].y - wps[i].y)
        return total

    len_d = _length(path_d)
    len_a = _length(path_a)
    longer = max(len_d, len_a)
    grid_cell_size = getattr(pd, "_resolution", 0.2)
    assert abs(len_d - len_a) <= max(longer * 0.10, 2 * grid_cell_size), f"path lengths diverge: dijkstra={len_d}, astar={len_a}"
