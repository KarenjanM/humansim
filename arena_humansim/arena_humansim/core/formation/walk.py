from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

import numpy as np

from arena_humansim.utils.types import Pose2D

from . import AgentLookup, Formation
from .anchor import AgentAnchor, Anchor

if TYPE_CHECKING:
    from .clearance import Clearance

_STATIONS = np.array([0.0, 1.0, 2.0])
_TRAIL_STEP = 0.05
_HEADING_SPAN = 1.0
_HEADING_CHORD = 0.5

_Step = tuple[Pose2D, float, float, list[int]]


class WalkFormation(Formation):
    """Followers abreast on the leader's trail, in a V or single file as crowds and walls narrow the way, in a circle when the leader stops."""

    def __init__(
        self,
        anchor: Anchor,
        agent_lookup: AgentLookup,
        spacing: float = 0.75,
        row_gap: float = 0.9,
        lead_time: float = 1.0,
        gain: float = 1.0,
        body_margin: float = 0.35,
        lag_tolerance: float = 0.6,
        lag_span: float = 2.0,
        min_pace: float = 0.4,
        stop_speed: float = 0.15,
        stop_time: float = 1.0,
        reopen_rate: float = 0.5,
        hysteresis: float = 0.25,
        vee_depth: float = 0.5,
        crowd_radius: float = 3.0,
        crowd_low: float = 0.05,
        crowd_high: float = 0.25,
        crowd_file: float = 0.5,
        crowd_tau: float = 1.0,
        circle_on_stop: bool = True,
        sense_period: float = 0.1,
        base_radius: float = 0.7,
        radius_per_member: float = 0.12,
        formation_scale: float = 1.0,
    ) -> None:
        if not isinstance(anchor, AgentAnchor):
            raise TypeError("walk formation needs an agent anchor")
        self.anchor = anchor
        self.agent_lookup = agent_lookup
        self.leader = anchor.agent_id
        self.spacing = spacing * formation_scale
        self.row_gap = row_gap * formation_scale
        self.lead_time = lead_time
        self.gain = gain
        self.body_margin = body_margin
        self.lag_tolerance = lag_tolerance
        self.lag_span = lag_span
        self.min_pace = min_pace
        self.stop_speed = stop_speed
        self.stop_time = stop_time
        self.reopen_rate = reopen_rate
        self.hysteresis = hysteresis
        self.vee_depth = vee_depth * formation_scale
        self.crowd_radius = crowd_radius
        self.crowd_low = crowd_low
        self.crowd_high = crowd_high
        self.crowd_file = crowd_file
        self.crowd_tau = crowd_tau
        self.circle_on_stop = circle_on_stop
        self.sense_period = sense_period
        self.base_radius = base_radius * formation_scale
        self.radius_per_member = radius_per_member * formation_scale
        self._members: list[int] = []
        self._side: dict[int, float] = {}
        self._heading = anchor.pose().theta
        self._tx: list[float] = []
        self._ty: list[float] = []
        self._ts: list[float] = []
        self._room: tuple[float, float] | None = None
        self._prefer_left = True
        self._front: int | None = None
        self._density = 0.0
        self._filing = False
        self._vee = False
        self._vee_row = False
        self._vee_left = True
        self._layout: list[list[tuple[int, float]]] | None = None
        self._sig: tuple[object, ...] | None = None
        self._depth = 0.0
        self._since_sense = 0.0
        self._still = 0.0
        self._circle: tuple[float, float, list[int], list[float]] | None = None
        self._slots: dict[int, Pose2D] = {}
        self._speeds: dict[int, float] = {}

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        if agent_id in self._members:
            return
        self._members.append(agent_id)
        if agent_id == self.leader:
            return
        lead = self.agent_lookup(self.leader)
        agent = self.agent_lookup(agent_id)
        if lead is None or agent is None:
            self._side[agent_id] = 0.0
            return
        lp, ap = lead.state.pose, agent.state.pose
        self._side[agent_id] = -(ap.x - lp.x) * math.sin(self._heading) + (ap.y - lp.y) * math.cos(self._heading)
        self._circle = None
        self._layout = None
        self._sig = None

    def on_leave(self, agent_id: int) -> None:
        if agent_id in self._members:
            self._members.remove(agent_id)
        self._side.pop(agent_id, None)
        self._slots.pop(agent_id, None)
        self._circle = None
        self._layout = None
        self._sig = None

    def speeds(self) -> dict[int, float]:
        return self._speeds

    def slot_of(self, agent_id: int) -> Pose2D | None:
        return self._slots.get(agent_id)

    def tick(self, dt: float) -> dict[int, Pose2D]:
        return self.tick_all([self], dt)[0]

    @classmethod
    def tick_all(cls, formations: Sequence[Self], dt: float) -> list[dict[int, Pose2D]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        steps = [f._advance(dt) for f in formations]
        walking = [(f, step) for f, step in zip(formations, steps, strict=True) if step is not None and not f._stopped()]
        cls._sense([(f, step) for f, step in walking if f._due(dt)])
        return [{} if step is None else f._close(step[0], step[3]) if f._stopped() else f._walk(*step) for f, step in zip(formations, steps, strict=True)]

    def _advance(self, dt: float) -> _Step | None:
        self._slots = {}
        self._speeds = {}
        lead = self.agent_lookup(self.leader)
        followers = [m for m in self._members if m != self.leader]
        if lead is None or self.leader not in self._members or not followers:
            return None
        pose = lead.state.pose
        vx, vy = lead.state.velocity
        speed = math.hypot(vx, vy)
        self._still = 0.0 if speed > self.stop_speed else self._still + dt
        self._record(pose.x, pose.y)
        if self._ts[-1] - self._ts[0] >= _HEADING_SPAN:
            bx, by, _ = self._on_trail(pose, _HEADING_SPAN, 0.0)
            if math.hypot(pose.x - bx, pose.y - by) >= _HEADING_CHORD:
                self._heading = math.atan2(pose.y - by, pose.x - bx)
        if not self._stopped():
            self._circle = None
        return pose, speed, lead.state.desired_velocity, followers

    def _stopped(self) -> bool:
        return self.circle_on_stop and self._still >= self.stop_time

    def _due(self, dt: float) -> bool:
        self._since_sense += dt
        return self._layout is None or self._since_sense >= self.sense_period

    @staticmethod
    def _sense(due: list[tuple[WalkFormation, _Step]]) -> None:
        by_clearance: dict[Clearance | None, list[tuple[WalkFormation, _Step]]] = {}
        for f, step in due:
            by_clearance.setdefault(f.clearance, []).append((f, step))
        for clearance, group in by_clearance.items():
            poses = np.array([(step[0].x, step[0].y, f._heading, len(step[3]) * f.spacing + f.body_margin, f.crowd_radius) for f, step in group])
            if clearance is None:
                free = poses[:, [3, 3]]
                counts = np.zeros(len(group), dtype=np.int64)
            else:
                free = clearance.sides(poses, _STATIONS)
                ahead = poses[:, 4] / 2.0
                centers = np.stack([poses[:, 0] + ahead * np.cos(poses[:, 2]), poses[:, 1] + ahead * np.sin(poses[:, 2])], axis=-1)
                counts = clearance.crowds([f.leader for f, _ in group], centers, poses[:, 4], [f._members for f, _ in group])
            for (f, step), (left, right), count in zip(group, free.tolist(), counts.tolist(), strict=True):
                f._sensed(step[0], left, right, count, step[3])

    def _sensed(self, pose: Pose2D, left: float, right: float, count: int, followers: list[int]) -> None:
        left, right = self._room_after(left, right, self._since_sense)
        self._depth = self.vee_depth * self._crowding_after(pose, count, self._since_sense) / self.spacing
        if self._filing:
            left = right = min(left, right, self.body_margin + self.spacing / 2.0)
        sig = (tuple(followers), self._filing, self._vee, self._vee_left, self._prefer_left, *self._fits(left, right, len(followers)))
        if sig != self._sig:
            self._layout = self._rows(followers, left, right)
            self._sig = sig
        self._since_sense = 0.0

    def _fits(self, left: float, right: float, count: int) -> list[int]:
        out: list[int] = []
        for room_left, room_right in ((left, right), (left - self.hysteresis, right - self.hysteresis)):
            out.append(min(max(math.floor((room_left - self.body_margin) / self.spacing), -count - 1), count + 1))
            out.append(min(max(math.ceil(-(room_right - self.body_margin) / self.spacing), -count - 1), count + 1))
        return out

    def _record(self, x: float, y: float) -> None:
        ts = self._ts
        if not ts:
            self._tx.append(x)
            self._ty.append(y)
            ts.append(0.0)
            return
        step = math.hypot(x - self._tx[-1], y - self._ty[-1])
        if step < _TRAIL_STEP:
            return
        self._tx.append(x)
        self._ty.append(y)
        ts.append(ts[-1] + step)
        drop = min(bisect_left(ts, ts[-1] - (len(self._members) * self.row_gap + 2.0), 1) - 1, len(ts) - 2)
        if drop > 0:
            del self._tx[:drop], self._ty[:drop], ts[:drop]

    def _on_trail(self, pose: Pose2D, back: float, lat: float) -> tuple[float, float, float]:
        """Point `back` along the trail behind the pose and `lat` to its left, with the trail heading there, for back > 0 on a trail of two points or more."""
        ts = self._ts
        last = len(ts) - 1
        target = ts[last] + math.hypot(pose.x - self._tx[last], pose.y - self._ty[last]) - back
        i = max(bisect_right(ts, target) - 1, 0)
        ax, ay, a_s = self._tx[i], self._ty[i], ts[i]
        bx, by, b_s = (self._tx[i + 1], self._ty[i + 1], ts[i + 1]) if i < last else (pose.x, pose.y, target + back)
        seg = b_s - a_s
        tx, ty = (bx - ax, by - ay) if seg > 1e-9 else (math.cos(self._heading), math.sin(self._heading))
        norm = math.hypot(tx, ty) or 1.0
        tx, ty = tx / norm, ty / norm
        f = (target - a_s) / seg if seg > 1e-9 else 0.0
        x, y = ax + f * (bx - ax), ay + f * (by - ay)
        return x - lat * ty, y + lat * tx, math.atan2(ty, tx)

    def _room_after(self, left: float, right: float, dt: float) -> tuple[float, float]:
        if self._room is not None:
            grow = self.reopen_rate * dt
            left = min(left, self._room[0] + grow)
            right = min(right, self._room[1] + grow)
        self._room = (left, right)
        if self._prefer_left and right > left + self.hysteresis:
            self._prefer_left = False
        elif not self._prefer_left and left > right + self.hysteresis:
            self._prefer_left = True
        return left, right

    def _crowding_after(self, pose: Pose2D, count: int, dt: float) -> float:
        density = count / (math.pi * self.crowd_radius**2)
        self._density += (density - self._density) * min(1.0, dt / self.crowd_tau)
        self._filing = self._density >= (0.75 * self.crowd_file if self._filing else self.crowd_file)
        level = min(max((self._density - self.crowd_low) / (self.crowd_high - self.crowd_low), 0.0), 1.0)
        vee = level >= (0.25 if self._vee else 0.5)
        if vee and not self._vee and self.clearance is not None:
            ahead = self.crowd_radius / 2.0
            x, y = pose.x + ahead * math.cos(self._heading), pose.y + ahead * math.sin(self._heading)
            left, right = self.clearance.oncoming(self.leader, x, y, self.crowd_radius, self._heading, self._members)
            self._vee_left = left < right or (left == right and self._prefer_left)
        self._vee = vee
        return level

    def _vee_offsets(self, left: float, right: float, count: int) -> list[float] | None:
        lo, hi = -(right - self.body_margin), left - self.body_margin
        for side in (1, -1) if self._vee_left else (-1, 1):
            offsets = [side * j * self.spacing for j in range(1, count + 1)]
            if all(lo <= o <= hi for o in offsets):
                return offsets
        return None

    def _offsets(self, left: float, right: float, count: int, reserve_center: bool) -> list[float]:
        lo, hi = -(right - self.body_margin), left - self.body_margin
        order = [] if reserve_center else [0]
        first = 1 if self._prefer_left else -1
        for j in range(1, count + 1):
            order.extend((first * j, -first * j))
        return [o * self.spacing for o in order if lo <= o * self.spacing <= hi][:count]

    def _front_count(self, left: float, right: float, count: int) -> int:
        fits = len(self._offsets(left, right, count, reserve_center=True))
        roomy = len(self._offsets(left - self.hysteresis, right - self.hysteresis, count, reserve_center=True))
        if self._front is None or fits < self._front:
            self._front = fits
        elif roomy > self._front:
            self._front = roomy
        return self._front

    def _rows(self, followers: list[int], left: float, right: float) -> list[list[tuple[int, float]]]:
        pending = list(followers)
        rows: list[list[tuple[int, float]]] = []
        self._vee_row = False
        while pending:
            vee = None if rows or self._filing or not self._vee or len(pending) < 2 else self._vee_offsets(left, right, len(pending))
            if rows:
                offsets = self._offsets(left, right, len(pending), reserve_center=False) or [0.0]
            elif vee is not None:
                offsets = vee
                self._vee_row = True
            else:
                offsets = [] if self._filing else self._offsets(left, right, len(pending), reserve_center=True)[: self._front_count(left, right, len(pending))]
            take: list[int] = []
            used: list[float] = []
            for offset in offsets:
                fits = [f for f in pending if rows or vee is not None or self._side.get(f, 0.0) * offset >= 0.0]
                if not fits:
                    continue
                best = min(fits, key=lambda f: (abs(self._side.get(f, 0.0) - offset), f))
                pending.remove(best)
                take.append(best)
                used.append(offset)
            take.sort(key=lambda f: self._side.get(f, 0.0))
            rows.append(list(zip(take, sorted(used), strict=True)))
        return rows

    def _walk(self, pose: Pose2D, speed: float, own_speed: float, followers: list[int]) -> dict[int, Pose2D]:
        depth = self._depth
        out: dict[int, Pose2D] = {}
        worst = 0.0
        lead = speed * self.lead_time
        hx, hy = math.cos(self._heading), math.sin(self._heading)
        theta = math.atan2(hy, hx)
        straight = len(self._ts) < 2
        for row_idx, row in enumerate(self._layout or ()):
            center = sum(lat for _, lat in row) / (len(row) + 1)
            for aid, lat in row:
                agent = self.agent_lookup(aid)
                if agent is None:
                    continue
                if row_idx:
                    back = row_idx * self.row_gap
                else:
                    back = depth * (abs(center) - abs(lat - center)) if self._vee_row else 0.0
                if back <= 0.0 or straight:
                    sx, sy, heading, tx, ty = pose.x - back * hx - lat * hy, pose.y - back * hy + lat * hx, theta, hx, hy
                else:
                    sx, sy, heading = self._on_trail(pose, back, lat)
                    tx, ty = math.cos(heading), math.sin(heading)
                if back <= lead or straight:
                    cx, cy = pose.x - (back - lead) * hx - lat * hy, pose.y - (back - lead) * hy + lat * hx
                else:
                    cx, cy, _ = self._on_trail(pose, back - lead, lat)
                ap = agent.state.pose
                along = (sx - ap.x) * tx + (sy - ap.y) * ty
                self._slots[aid] = Pose2D(x=sx, y=sy, theta=heading)
                self._speeds[aid] = min(max(speed + self.gain * along, 0.0), agent.params.max_velocity)
                out[aid] = Pose2D(x=cx, y=cy, theta=heading)
                worst = max(worst, along)
        pace = min(max(1.0 - (worst - self.lag_tolerance) / self.lag_span, self.min_pace), 1.0)
        self._speeds[self.leader] = own_speed * pace
        return out

    def _close(self, pose: Pose2D, followers: list[int]) -> dict[int, Pose2D]:
        if self._circle is None:
            positions = [a.state.pose for f in followers if (a := self.agent_lookup(f)) is not None]
            mx = sum(p.x for p in positions) / max(len(positions), 1) - pose.x
            my = sum(p.y for p in positions) / max(len(positions), 1) - pose.y
            norm = math.hypot(mx, my)
            ux, uy = (mx / norm, my / norm) if norm > 1e-6 else (math.cos(self._heading), math.sin(self._heading))
            radius = self.base_radius + self.radius_per_member * len(followers)
            cx, cy = pose.x + radius * ux, pose.y + radius * uy
            a0 = math.atan2(pose.y - cy, pose.x - cx)

            def bearing(f: int) -> float:
                agent = self.agent_lookup(f)
                if agent is None:
                    return 0.0
                return (math.atan2(agent.state.pose.y - cy, agent.state.pose.x - cx) - a0) % (2.0 * math.pi)

            n = len(followers) + 1
            angles = [a0 + 2.0 * math.pi * k / n for k in range(1, n)]
            radii = [radius] * len(angles)
            if self.clearance is not None:
                dirs = np.array([(math.cos(a), math.sin(a)) for a in angles])
                free = self.clearance.free(np.full((len(angles), 2), (cx, cy)), dirs, radius + self.body_margin)
                radii = [float(min(radius, max(f - self.body_margin, 0.0))) for f in free]
            self._circle = (cx, cy, sorted(followers, key=bearing), radii)
        cx, cy, order, radii = self._circle
        a0 = math.atan2(pose.y - cy, pose.x - cx)
        n = len(order) + 1
        out: dict[int, Pose2D] = {}
        for k, (aid, r) in enumerate(zip(order, radii, strict=True), start=1):
            a = a0 + 2.0 * math.pi * k / n
            slot = Pose2D(x=cx + r * math.cos(a), y=cy + r * math.sin(a), theta=a + math.pi)
            self._slots[aid] = slot
            out[aid] = slot
        return out
