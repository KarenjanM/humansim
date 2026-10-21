from __future__ import annotations

import enum
import math
from collections import deque
from collections.abc import Sequence

import attrs

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

WAIT_RADIUS = 1.0
BOARD_RADIUS = 0.3
SLOT_SPACING = 0.7
SLOT_MARGIN = 0.35


def _inside(pts: tuple[tuple[float, float], ...], x: float, y: float) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(pts, pts[-1:] + pts[:-1], strict=True):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def _slots(cabin: Cabin) -> tuple[Pose2D, ...]:
    """Rider spots on a grid inside the footprint, the one farthest from the doorway first, the boarding pose when none fits."""
    if not cabin.footprint:
        return (cabin.boarding,)
    xs, ys = [x for x, _ in cabin.footprint], [y for _, y in cabin.footprint]
    axes = []
    for lo, hi in ((min(xs) + SLOT_MARGIN, max(xs) - SLOT_MARGIN), (min(ys) + SLOT_MARGIN, max(ys) - SLOT_MARGIN)):
        n = max(1, math.floor(max(hi - lo, 0.0) / SLOT_SPACING) + 1)
        mid = (lo + hi) / 2.0
        axes.append([mid + (i - (n - 1) / 2.0) * SLOT_SPACING for i in range(n)])
    spots = [Pose2D(x=x, y=y, theta=cabin.boarding.theta) for x in axes[0] for y in axes[1] if _inside(cabin.footprint, x, y)]
    spots.sort(key=lambda p: -math.hypot(p.x - cabin.doorway.x, p.y - cabin.doorway.y))
    return tuple(spots) or (cabin.boarding,)


@attrs.frozen
class Cabin:
    name: str
    destination: str
    boarding: Pose2D
    doorway: Pose2D
    queue: Pose2D
    footprint: tuple[tuple[float, float], ...] = ()
    slots: tuple[Pose2D, ...] = attrs.field(init=False, default=attrs.Factory(_slots, takes_self=True))

    def holds(self, x: float, y: float) -> bool:
        """Whether a rider at x, y stands inside the cabin."""
        if math.hypot(x - self.boarding.x, y - self.boarding.y) <= BOARD_RADIUS:
            return True
        return _inside(self.footprint, x, y)


class Phase(enum.IntEnum):
    WALK = 0
    WAIT = 1
    BOARD = 2
    HOLD = 3


@attrs.define
class Ride:
    goal: tuple[float, float]
    cabin: str
    phase: Phase = Phase.WALK
    since: int = 0
    slot: int = -1


@attrs.frozen
class _Graph:
    zone: dict[int, int]
    edges: dict[int, list[tuple[str, int]]]


class Elevators:
    """Elevator cabins, their doors and the pedestrians riding them."""

    def __init__(self) -> None:
        self.cabins: dict[str, Cabin] = {}
        self.open: dict[str, bool] = {}
        self.rides: dict[int, Ride] = {}
        self._banned: dict[int, tuple[tuple[float, float], set[str]]] = {}

    def set_cabins(self, cabins: Sequence[Cabin], door_open: Sequence[bool]) -> None:
        self.cabins = {c.name: c for c in cabins}
        self.open = {c.name: bool(o) for c, o in zip(cabins, door_open, strict=True)}
        self.clear_rides()

    def clear_rides(self) -> None:
        self.rides.clear()
        self._banned.clear()

    def forget(self, aid: int) -> None:
        self.rides.pop(aid, None)
        self._banned.pop(aid, None)

    def steer(self, agents: Sequence[BaseAgent], cmds: dict[int, HighLevelCommand]) -> dict[int, HighLevelCommand]:
        """Commands with each rider's target swapped for its cabin's queue pose or its spot inside."""
        if not self.cabins:
            return cmds
        out = dict(cmds)
        groups: dict[GlobalPlanner, list[tuple[BaseAgent, HighLevelCommand]]] = {}
        for agent in agents:
            cmd = cmds.get(agent.state.agent_id)
            if cmd is not None and cmd.type == CommandType.NAVIGATE and agent.state.kind != KIND_ROBOT:
                groups.setdefault(agent.global_planner, []).append((agent, cmd))
        for planner, group in groups.items():
            graph = self._graph(planner)
            comps = planner.components([agent.state.pose for agent, _ in group] + [cmd.target_pose for _, cmd in group])
            for k, (agent, cmd) in enumerate(group):
                aid = agent.state.agent_id
                goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
                ride = self.rides.get(aid)
                if ride is not None and ride.goal != goal:
                    ride = None
                    del self.rides[aid]
                banned = self._banned.get(aid)
                if banned is not None and banned[0] != goal:
                    banned = None
                    del self._banned[aid]
                if ride is None or ride.phase == Phase.WALK:
                    cabin = self._route(graph, comps[k], comps[len(group) + k], set() if banned is None else banned[1])
                    if cabin is None:
                        self.rides.pop(aid, None)
                        continue
                    if ride is None or ride.cabin != cabin:
                        ride = self.rides[aid] = Ride(goal=goal, cabin=cabin)
                c = self.cabins[ride.cabin]
                out[aid] = attrs.evolve(cmd, target_pose=c.queue if ride.phase in (Phase.WALK, Phase.WAIT) else c.slots[ride.slot])
        return out

    def hold(self, pool: AgentPool, tick: int, timeout_ticks: int) -> None:
        """Advance every rider's phase, slow the waiting ones to desired speed 0 and stand the holding ones still."""
        for aid in list(self.rides):
            ride = self.rides[aid]
            idx = pool._id_to_idx.get(aid)
            if idx is None:
                self.forget(aid)
                continue
            cabin = self.cabins[ride.cabin]
            is_open = self.open[ride.cabin]
            x, y = float(pool.pos[idx, 0]), float(pool.pos[idx, 1])
            if ride.phase == Phase.WALK and math.hypot(x - cabin.queue.x, y - cabin.queue.y) <= WAIT_RADIUS:
                ride.phase, ride.since = Phase.WAIT, tick
            if ride.phase == Phase.WAIT and is_open:
                slot = self._free_slot(cabin)
                if slot is not None:
                    ride.phase, ride.slot, ride.since = Phase.BOARD, slot, tick
            elif ride.phase == Phase.BOARD and self._at_slot(cabin, ride, x, y):
                ride.phase = Phase.HOLD
            elif ride.phase == Phase.BOARD and not is_open and not cabin.holds(x, y):
                ride.phase, ride.slot = Phase.WAIT, -1
            elif ride.phase == Phase.HOLD and is_open and not cabin.holds(x, y):
                ride.phase = Phase.BOARD
            if ride.phase != Phase.WALK and tick - ride.since >= timeout_ticks:
                del self.rides[aid]
                self._banned.setdefault(aid, (ride.goal, set()))[1].add(ride.cabin)
                continue
            if ride.phase in (Phase.WAIT, Phase.HOLD):
                pool.desired_vel[idx] = 0.0
            if ride.phase == Phase.HOLD:
                pool.vel[idx] = 0.0
                pool.prev_vel[idx] = 0.0

    def _free_slot(self, cabin: Cabin) -> int | None:
        taken = {r.slot for r in self.rides.values() if r.cabin == cabin.name and r.phase in (Phase.BOARD, Phase.HOLD)}
        return next((i for i in range(len(cabin.slots)) if i not in taken), None)

    @staticmethod
    def _at_slot(cabin: Cabin, ride: Ride, x: float, y: float) -> bool:
        spot = cabin.slots[ride.slot]
        return math.hypot(x - spot.x, y - spot.y) <= BOARD_RADIUS

    def _graph(self, planner: GlobalPlanner) -> _Graph:
        """Cabins as ride edges between zones, a zone being the components one cabin's doorway and boarding pose join."""
        names = sorted(self.cabins)
        comps = planner.components([p for name in names for p in (self.cabins[name].doorway, self.cabins[name].boarding)])
        parent: dict[int, int] = {}

        def find(c: int) -> int:
            while parent.setdefault(c, c) != c:
                c = parent[c]
            return c

        sides = {name: comps[2 * i] | comps[2 * i + 1] for i, name in enumerate(names)}
        for side in sides.values():
            roots = sorted({find(c) for c in side})
            for r in roots[1:]:
                parent[r] = roots[0]
        zone = {c: find(c) for c in parent}
        edges: dict[int, list[tuple[str, int]]] = {}
        for name in names:
            dest = self.cabins[name].destination
            if not sides[name] or dest not in sides or not sides[dest]:
                continue
            a, b = zone[min(sides[name])], zone[min(sides[dest])]
            if a != b:
                edges.setdefault(a, []).append((name, b))
        return _Graph(zone=zone, edges=edges)

    @staticmethod
    def _route(graph: _Graph, start: frozenset[int], goal: frozenset[int], banned: set[str]) -> str | None:
        """First cabin of the fewest-rides route from start to goal, None when start reaches goal on foot or no route exists."""
        starts = {graph.zone.get(c, c) for c in start}
        goals = {graph.zone.get(c, c) for c in goal}
        if not starts or not goals or starts & goals:
            return None
        first: dict[int, str | None] = dict.fromkeys(starts)
        queue = deque(sorted(starts))
        while queue:
            z = queue.popleft()
            for name, to in graph.edges.get(z, ()):
                if name in banned or to in first:
                    continue
                first[to] = first[z] or name
                if to in goals:
                    return first[to]
                queue.append(to)
        return None
