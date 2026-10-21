from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import PoolAware
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segment, WallAware
from arena_humansim.utils.wall_grid import WallGrid, cast_rays

from ._grid import min_distances_to_paths, next_waypoint

if TYPE_CHECKING:
    from arena_humansim.core.viz import MarkerPublisher

_registry: ModuleRegistry[GlobalPlanner] = ModuleRegistry()

PlanRequest = tuple[int, Pose2D, Pose2D]

DIRECT_RANGE = 3.0


def simplify_path(
    waypoints: Sequence[Pose2D],
    min_area: float = 0.01,
) -> list[Pose2D]:
    n = len(waypoints)
    if n <= 2:
        return list(waypoints)

    xs = [w.x for w in waypoints]
    ys = [w.y for w in waypoints]
    prev_idx = list(range(-1, n - 1))
    next_idx = list(range(1, n + 1))

    _inf = math.inf
    _heappush = heapq.heappush
    _heappop = heapq.heappop

    def _area(i: int) -> float:
        p, nx = prev_idx[i], next_idx[i]
        return 0.5 * abs((xs[i] - xs[p]) * (ys[nx] - ys[p]) - (xs[nx] - xs[p]) * (ys[i] - ys[p]))

    areas = [_inf] + [_area(i) for i in range(1, n - 1)] + [_inf]
    heap = [(areas[i], i) for i in range(1, n - 1)]
    heapq.heapify(heap)
    removed = bytearray(n)

    while heap:
        area, i = _heappop(heap)
        if removed[i] or areas[i] != area:
            continue
        if area >= min_area:
            break

        removed[i] = 1
        p, nx = prev_idx[i], next_idx[i]
        next_idx[p] = nx
        prev_idx[nx] = p

        if p > 0:
            a = max(_area(p), area)
            areas[p] = a
            _heappush(heap, (a, p))
        if nx < n - 1:
            a = max(_area(nx), area)
            areas[nx] = a
            _heappush(heap, (a, nx))

    return [waypoints[i] for i in range(n) if not removed[i]]


class GlobalPlanner(PoolAware, WallAware, Loggable, ABC):
    def __init__(self, replan_distance: float = 1.0) -> None:
        self._replan_distance = replan_distance
        self._path_cache: dict[int, tuple[tuple[float, float], list[Pose2D], np.ndarray, int]] = {}
        self._cached_results: dict[int, Pose2D] = {}
        self._wall_segments: list[Segment] = []
        self._wall_grid: tuple[list[Segment], WallGrid] | None = None
        self._warmup()

    def _warmup(self) -> None:
        cast_rays(WallGrid([((0.0, 1.0), (1.0, 1.0))]), np.zeros((1, 2)), np.ones((1, 2)), 1.0)

    @abstractmethod
    def _has_map(self) -> bool: ...

    @abstractmethod
    def _plan(self, requests: list[PlanRequest]) -> dict[int, list[Pose2D] | None]: ...

    def compute(
        self,
        agents: Iterable[BaseAgent],
        high_level_commands: dict[int, HighLevelCommand],
    ) -> dict[int, Pose2D]:
        agent_positions: dict[int, Pose2D] = {agent.state.agent_id: agent.state.pose for agent in agents}
        goals: dict[int, Pose2D] = {}
        requests: dict[int, tuple[Pose2D, Pose2D]] = {}
        navigate = {agent_id: cmd.target_pose for agent_id, cmd in high_level_commands.items() if isinstance(cmd, HighLevelCommand) and cmd.type == CommandType.NAVIGATE}
        has_map = self._has_map()

        deviations: dict[int, float] = {}
        if has_map:
            tracked = [(agent_id, agent_positions[agent_id], entry[2]) for agent_id, target in navigate.items() if agent_id in agent_positions and (entry := self._path_cache.get(agent_id)) is not None and math.hypot(entry[0][0] - target.x, entry[0][1] - target.y) <= self._replan_distance and entry[1]]
            if tracked:
                positions = np.array([(pos.x, pos.y) for _, pos, _ in tracked], dtype=np.float64)
                distances = min_distances_to_paths(positions, [points for _, _, points in tracked])
                deviations = dict(zip([agent_id for agent_id, _, _ in tracked], distances.tolist(), strict=True))
        unchanged = {aid for aid, d in deviations.items() if d <= self._replan_distance and self._path_cache[aid][0] == (round(navigate[aid].x, 3), round(navigate[aid].y, 3))}
        straight = [aid for aid, target in navigate.items() if has_map and aid not in unchanged and (pos := agent_positions.get(aid)) is not None and (math.hypot(target.x - pos.x, target.y - pos.y) <= DIRECT_RANGE or (aid in deviations and self._path_cache[aid][3] >= len(self._path_cache[aid][1]) - 2))]
        direct = set(straight)
        moved = [aid for aid, d in deviations.items() if d <= self._replan_distance and aid not in unchanged and aid not in direct]
        legs = straight + moved
        ends = dict(zip(legs, self.snap_terminals([navigate[aid] for aid in legs]), strict=True))
        starts = [agent_positions[aid] for aid in straight] + [waypoints[-2] if len(waypoints := self._path_cache[aid][1]) > 1 else agent_positions[aid] for aid in moved]
        lines = np.array([((a.x, a.y), (ends[aid].x, ends[aid].y)) for aid, a in zip(legs, starts, strict=True)], dtype=np.float64).reshape(-1, 2, 2)
        clear = dict(zip(legs, self._sees(lines[:, 0], lines[:, 1]).tolist() if legs else [], strict=True))
        line_of = dict(zip(straight, lines, strict=False))

        for agent_id, target in navigate.items():
            agent_pos = agent_positions.get(agent_id)

            if agent_pos is None or not has_map:
                goals[agent_id] = target
                continue

            if clear.get(agent_id):
                end = ends[agent_id]
                if agent_id in direct:
                    waypoints, idx, points = [Pose2D(x=agent_pos.x, y=agent_pos.y), end], 0, line_of[agent_id]
                else:
                    cached_goal, waypoints, points, idx = self._path_cache[agent_id]
                    waypoints = [*waypoints[:-1], end]
                    points = np.vstack([points[:-1], (end.x, end.y)])
                idx = self.advance_along_path(agent_pos, waypoints, idx)
                self._path_cache[agent_id] = ((round(target.x, 3), round(target.y, 3)), waypoints, points, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)
                continue

            if agent_id in unchanged:
                cached_goal, waypoints, points, idx = self._path_cache[agent_id]
                idx = self.advance_along_path(agent_pos, waypoints, idx)
                self._path_cache[agent_id] = (cached_goal, waypoints, points, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)
                continue

            requests[agent_id] = (agent_pos, target)

        if requests:
            plans = self._plan([(agent_id, agent_pos, target) for agent_id, (agent_pos, target) in requests.items()])
            detours = [(agent_id, agent_pos, reachable) for agent_id, (agent_pos, target) in requests.items() if plans[agent_id] is None and (reachable := self._nearest_reachable(agent_pos, target)) is not None]
            if detours:
                plans.update(self._plan(detours))

            for agent_id, (agent_pos, target) in requests.items():
                waypoints = plans[agent_id]

                if waypoints is None:
                    self._logger.debug(f"No path for agent {agent_id} ({agent_pos} -> {target}), using direct goal")
                    goals[agent_id] = self.snap_terminal(target)
                    self._path_cache.pop(agent_id, None)
                    continue

                goal_key = (round(target.x, 3), round(target.y, 3))
                idx = self.advance_along_path(agent_pos, waypoints, 0)
                self._path_cache[agent_id] = (goal_key, waypoints, np.array([(w.x, w.y) for w in waypoints], dtype=np.float64).reshape(-1, 2), idx)
                goals[agent_id] = next_waypoint(waypoints, idx)

        self._cached_results = goals
        return goals

    def _sees(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Whether each straight leg is walkable as a whole path."""
        if self._wall_grid is None or self._wall_grid[0] is not self._wall_segments:
            self._wall_grid = (self._wall_segments, WallGrid(self._wall_segments))
        return cast_rays(self._wall_grid[1], starts, ends - starts, 1.0) >= 1.0

    def get_cached_goals(self) -> dict[int, Pose2D]:
        return dict(self._cached_results)

    def get_cached_paths(self) -> dict[int, list[Pose2D]]:
        return {aid: wps for aid, (_, wps, _, _) in self._path_cache.items()}

    def _forget_paths(self) -> None:
        self._path_cache.clear()

    def configure(self, *, inflation_radius: float, resolution: float, comfort_radius: float) -> None:
        pass

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        return pose

    def snap_terminals(self, poses: Sequence[Pose2D]) -> list[Pose2D]:
        return [self.snap_terminal(p) for p in poses]

    def _nearest_reachable(self, start: Pose2D, target: Pose2D) -> Pose2D | None:
        """Point closest to an unreachable target that start can still reach, None to walk straight at the target."""
        return None

    def components(self, poses: Sequence[Pose2D]) -> list[frozenset[int]]:
        """Free-space components each pose can walk into, empty where the planner cannot tell."""
        return [frozenset() for _ in poses]

    def publish_markers(self, pub: MarkerPublisher) -> None:
        pass

    @staticmethod
    def advance_along_path(
        agent_pos: Pose2D,
        waypoints: Sequence[Pose2D],
        current_idx: int,
    ) -> int:
        idx = current_idx
        while idx < len(waypoints) - 1:
            wp = waypoints[idx]
            nxt = waypoints[idx + 1]
            dx, dy = nxt.x - wp.x, nxt.y - wp.y
            tx, ty = agent_pos.x - wp.x, agent_pos.y - wp.y
            if dx * tx + dy * ty >= dx * dx + dy * dy:
                idx += 1
            else:
                break
        return idx

    @classmethod
    def register(cls, name: str, label: str | None = None) -> Callable[[Callable[[], type[GlobalPlanner]]], Callable[[], type[GlobalPlanner]]]:
        return _registry.register(name, label)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> GlobalPlanner:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def labels(cls) -> dict[str, str]:
        return _registry.labels()


def _load_dijkstra() -> type[GlobalPlanner]:
    from .dijkstra import DijkstraPlanner

    return DijkstraPlanner


def _load_astar() -> type[GlobalPlanner]:
    from .astar import AStarPlanner

    return AStarPlanner


def _load_navmesh() -> type[GlobalPlanner]:
    from .navmesh import NavMeshPlanner

    return NavMeshPlanner


_registry.register("navmesh", "NavMesh")(_load_navmesh)
_registry.register("astar", "A*")(_load_astar)
_registry.register("dijkstra", "Dijkstra")(_load_dijkstra)
