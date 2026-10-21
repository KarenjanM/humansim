from __future__ import annotations

from collections.abc import Callable

import pytest

from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.global_planner.astar import AStarPlanner
from arena_humansim.global_planner.dijkstra import DijkstraPlanner
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import Pose2D, Segments


def _room(x0: float, y0: float, size: float) -> Segments:
    x1, y1 = x0 + size, y0 + size
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


TWO_ROOMS = _room(0.0, 0.0, 6.0) + _room(20.0, 0.0, 6.0)


@pytest.mark.parametrize(
    "make",
    [NavMeshPlanner, lambda: DijkstraPlanner(resolution=0.1), lambda: AStarPlanner(resolution=0.1)],
    ids=["navmesh", "dijkstra", "astar"],
)
def test_walled_rooms_are_separate_components_and_a_room_is_one(make: Callable[[], GlobalPlanner]) -> None:
    planner = make()
    planner.set_walls(TWO_ROOMS)
    a1, a2, b1, outside = planner.components([Pose2D(x=1.0, y=1.0), Pose2D(x=5.0, y=5.0), Pose2D(x=21.0, y=1.0), Pose2D(x=12.0, y=3.0)])
    assert a1 and b1 and outside
    assert a1 & a2
    assert not a1 & b1
    assert not a1 & outside
    assert not b1 & outside


@pytest.mark.parametrize("make", [NavMeshPlanner, lambda: DijkstraPlanner(resolution=0.1)], ids=["navmesh", "dijkstra"])
def test_components_follow_wall_changes(make: Callable[[], GlobalPlanner]) -> None:
    planner = make()
    planner.set_walls(TWO_ROOMS)
    assert not planner.components([Pose2D(x=1.0, y=1.0)])[0] & planner.components([Pose2D(x=21.0, y=1.0)])[0]
    planner.set_walls([w for i, w in enumerate(TWO_ROOMS) if i not in (2, 6)])
    a, b = planner.components([Pose2D(x=1.0, y=1.0), Pose2D(x=21.0, y=1.0)])
    assert a & b


def test_coarse_grid_and_wall_free_planners_cannot_tell_components() -> None:
    coarse = DijkstraPlanner(resolution=0.2)
    coarse.set_walls(TWO_ROOMS)
    assert coarse.components([Pose2D(x=1.0, y=1.0)]) == [frozenset()]
    assert NavMeshPlanner().components([Pose2D(x=1.0, y=1.0)]) == [frozenset()]
