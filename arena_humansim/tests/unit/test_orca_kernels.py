from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.spatial import cKDTree

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.orca import (
    ORCAPlanner,
    _agent_line,
    _det,
    _dist_sq_point_segment,
    _linear_program1,
    _linear_program2,
    _linear_program3,
    _obstacle_line,
    _orca_kernel,
    _push_line,
)
from arena_humansim.utils.types import Pose2D

_DT = 0.05


def _lines(*rows: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def _random_lines(rng: np.random.Generator, n: int) -> np.ndarray:
    ang = rng.uniform(-math.pi, math.pi, n)
    return np.column_stack([rng.uniform(-2.0, 2.0, (n, 2)), np.cos(ang), np.sin(ang)])


def _violation(lines: np.ndarray, vx: float | np.ndarray, vy: float | np.ndarray) -> np.ndarray:
    return lines[:, 2, None] * (lines[:, 1, None] - vy) - lines[:, 3, None] * (lines[:, 0, None] - vx)


def _seg_seg_dist(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    def point_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        t = 0.0 if ab @ ab == 0.0 else min(max((p - a) @ ab / (ab @ ab), 0.0), 1.0)
        return float(np.linalg.norm(p - a - t * ab))

    d1, d2 = p1 - p0, q1 - q0
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if denom != 0.0:
        w = q0 - p0
        s = (w[0] * d2[1] - w[1] * d2[0]) / denom
        u = (w[0] * d1[1] - w[1] * d1[0]) / denom
        if 0.0 <= s <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(point_seg(p0, q0, q1), point_seg(p1, q0, q1), point_seg(q0, p0, p1), point_seg(q1, p0, p1))


def _edge(ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    length = math.hypot(bx - ax, by - ay)
    return np.array([ax, ay, bx, by, (bx - ax) / length, (by - ay) / length])


def _pool(agent_factory: Callable[..., BaseAgent], specs: list[tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float]]]) -> AgentPool:
    pool = AgentPool(capacity=max(len(specs), 1))
    for i, (pos, goal, vel) in enumerate(specs):
        agent = agent_factory(agent_id=i + 1, x=pos[0], y=pos[1])
        agent.state.velocity = vel
        idx = pool.add_agent(agent)
        if goal is not None:
            pool.goal_pos[idx] = goal
            pool.has_goal[idx] = True
    return pool


def _kernel_both(planner: ORCAPlanner, pool: AgentPool, nbr_idx: np.ndarray | None = None) -> np.ndarray:
    n = pool.n
    pos = pool.pos[:n]
    if nbr_idx is None:
        k = min(planner.max_neighbors + 1, n)
        _, idx = cKDTree(pos).query(pos, k=k, distance_upper_bound=planner.neighbor_dist)
        nbr_idx = np.ascontiguousarray(idx.reshape(n, k), dtype=np.int64)
    grid = planner._grid
    args = (
        pos.copy(),
        pool.prev_vel[:n].copy(),
        pool.agent_radius[:n].copy(),
        pool.max_velocity[:n].copy(),
        pool.desired_vel[:n].copy(),
        pool.goal_pos[:n].copy(),
        pool.has_goal[:n].copy(),
        nbr_idx,
        planner.max_neighbors,
        planner._edges,
        grid.cell_start,
        grid.cell_walls,
        grid.wall_cx0,
        grid.wall_cy0,
        grid.origin_x,
        grid.origin_y,
        grid.cell,
        grid.nx,
        grid.ny,
        planner.goal_radius,
        planner.time_horizon,
        planner.time_horizon_obst,
        planner.wall_clearance,
        1.0 / _DT,
    )
    py_out = np.full((n, 2), np.nan)
    jit_out = np.full((n, 2), np.nan)
    _orca_kernel.py_func(*args, py_out)
    _orca_kernel(*args, jit_out)
    assert py_out == pytest.approx(jit_out, rel=1e-12, abs=1e-12)
    return py_out


def test_det_and_push_line_primitives() -> None:
    assert _det.py_func(1.0, 0.0, 0.0, 1.0) == 1.0
    assert _det.py_func(0.0, 1.0, 1.0, 0.0) == -1.0
    assert _det.py_func(2.0, 4.0, 1.0, 2.0) == 0.0
    lines = np.zeros((2, 4))
    assert _push_line.py_func(lines, 0, 1.0, 2.0, 0.6, 0.8) == 1
    assert _push_line.py_func(lines, 1, -1.0, 0.0, 0.0, -1.0) == 2
    assert lines.tolist() == [[1.0, 2.0, 0.6, 0.8], [-1.0, 0.0, 0.0, -1.0]]


@pytest.mark.parametrize(("point", "expected"), [((1.0, 2.0), 4.0), ((-3.0, 4.0), 25.0), ((5.0, -1.0), 10.0)], ids=["interior", "before_start", "past_end"])
def test_dist_sq_point_segment_known_cases(point: tuple[float, float], expected: float) -> None:
    assert _dist_sq_point_segment.py_func(0.0, 0.0, 2.0, 0.0, *point) == pytest.approx(expected)


def test_linear_program1_python_matches_compiled_on_documented_cases() -> None:
    cases = [
        (_lines((0.0, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, -1.0)), 1, 1.0, 0.6, 0.0, False, (False, 0.0, 0.0)),
        (_lines((0.0, 10.0, 0.0, -1.0), (0.5, 0.0, 0.0, -1.0)), 1, 2.0, 0.5, 0.5, False, (True, 0.5, 0.5)),
        (_lines((0.0, 0.2, 1.0, 0.0), (0.0, 0.8, -1.0, 0.0), (0.0, 0.0, 0.0, -1.0)), 2, 1.0, 0.0, 0.95, False, (True, 0.0, 0.8)),
        (_lines((0.0, 0.2, 1.0, 0.0), (0.0, 0.8, -1.0, 0.0), (0.0, 0.0, 0.0, -1.0)), 2, 1.0, 0.0, 0.0, False, (True, 0.0, 0.2)),
        (_lines((5.0, 0.0, 0.0, 1.0)), 0, 1.0, 0.0, 0.0, False, (False, 0.0, 0.0)),
        (_lines((0.0, 0.0, 0.0, -1.0)), 0, 2.0, 0.0, -1.0, True, (True, 0.0, -2.0)),
        (_lines((0.0, 0.0, 0.0, -1.0)), 0, 2.0, 0.0, 1.0, True, (True, 0.0, 2.0)),
        (_lines((0.0, 0.5, 1.0, 0.0), (0.0, -0.5, -1.0, 0.0), (0.0, 0.0, 0.0, 1.0)), 2, 1.0, 0.0, 0.0, False, (False, 0.0, 0.0)),
    ]
    for lines, line_no, radius, ox, oy, direction_opt, expected in cases:
        got = _linear_program1.py_func(lines, line_no, radius, ox, oy, direction_opt)
        assert got[0] == expected[0]
        assert got[1:] == pytest.approx(expected[1:], abs=1e-12)
        assert got == _linear_program1(lines, line_no, radius, ox, oy, direction_opt)


def test_linear_program2_returns_pref_when_feasible_and_clamps_to_speed() -> None:
    assert _linear_program2.py_func(_lines(), 0, 1.5, 10.0, 0.0, False) == pytest.approx((0, 1.5, 0.0))
    assert _linear_program2.py_func(_lines(), 0, 1.5, 0.3, -0.4, False) == (0, 0.3, -0.4)
    assert _linear_program2.py_func(_lines(), 0, 1.5, 0.6, 0.8, True) == pytest.approx((0, 0.9, 1.2))
    lines = _lines((0.0, -0.5, -1.0, 0.0))
    assert _linear_program2.py_func(lines, 1, 2.0, 1.0, 0.0, False) == pytest.approx((1, 1.0, -0.5))


def test_linear_program2_reports_first_infeasible_line() -> None:
    lines = _lines((0.0, 0.0, 0.0, -1.0), (-0.5, 0.0, 0.0, 1.0))
    fail, rx, ry = _linear_program2.py_func(lines, 2, 1.0, 1.0, 0.0, False)
    assert fail == 1
    assert (rx, ry) == (1.0, 0.0)


def test_linear_program2_solution_satisfies_every_line_on_random_problems() -> None:
    rng = np.random.default_rng(17)
    feasible = 0
    for _ in range(300):
        n = int(rng.integers(1, 7))
        lines = _random_lines(rng, n)
        radius = float(rng.uniform(0.8, 2.0))
        ox, oy = (float(v) for v in rng.uniform(-2.5, 2.5, 2))
        fail, rx, ry = _linear_program2.py_func(lines, n, radius, ox, oy, False)
        assert (fail, rx, ry) == pytest.approx(_linear_program2(lines, n, radius, ox, oy, False), abs=1e-12)
        assert math.hypot(rx, ry) <= radius + 1e-9
        assert np.all(_violation(lines[:fail], rx, ry) <= 1e-9)
        if fail == n:
            feasible += 1
            if np.all(_violation(lines, ox, oy) <= 0.0) and math.hypot(ox, oy) <= radius:
                assert (rx, ry) == (ox, oy)
    assert 0 < feasible < 300


def test_linear_program3_minimizes_the_worst_violation() -> None:
    rng = np.random.default_rng(23)
    grid = np.linspace(-1.0, 1.0, 401)
    gx, gy = np.meshgrid(grid, grid)
    inside = gx**2 + gy**2 <= 1.0
    gx, gy = gx[inside], gy[inside]
    checked = 0
    for _ in range(200):
        n = int(rng.integers(2, 7))
        lines = _random_lines(rng, n)
        fail, rx, ry = _linear_program2.py_func(lines, n, 1.0, 0.5, 0.5, False)
        if fail >= n:
            continue
        checked += 1
        rx, ry = _linear_program3.py_func(lines, n, 0, fail, 1.0, rx, ry)
        assert (rx, ry) == pytest.approx(_linear_program3(lines, n, 0, fail, 1.0, *_linear_program2(lines, n, 1.0, 0.5, 0.5, False)[1:]), abs=1e-12)
        assert math.hypot(rx, ry) <= 1.0 + 1e-9
        best = _violation(lines, gx, gy).max(axis=0).min()
        assert _violation(lines, rx, ry).max() <= best + 1e-9
    assert checked > 20


def test_linear_program3_keeps_obstacle_lines_hard_and_skips_parallel_projections() -> None:
    lines = _lines((0.0, 0.0, 0.0, -1.0), (-0.5, 0.0, 0.0, 1.0), (0.0, 0.3, 1.0, 0.0), (0.0, 0.6, 1.0, 0.0), (0.0, -0.6, -1.0, 0.0))
    fail, rx, ry = _linear_program2.py_func(lines, 5, 1.0, 1.0, 0.0, False)
    assert fail == 1
    rx, ry = _linear_program3.py_func(lines, 5, 1, fail, 1.0, rx, ry)
    assert rx >= -1e-12
    assert math.hypot(rx, ry) <= 1.0 + 1e-9
    assert _violation(lines[1:], rx, ry).max() == pytest.approx(0.6)
    assert (rx, ry) == pytest.approx(_linear_program3(lines, 5, 1, fail, 1.0, *_linear_program2(lines, 5, 1.0, 1.0, 0.0, False)[1:]), abs=1e-12)


def test_linear_program3_skips_parallel_lines_facing_the_same_way() -> None:
    lines = _lines((0.0, 0.3, 1.0, 0.0), (0.0, -0.3, -1.0, 0.0), (0.0, 0.6, 1.0, 0.0))
    fail, rx, ry = _linear_program2.py_func(lines, 3, 1.0, 0.0, 0.0, False)
    assert fail == 1
    rx, ry = _linear_program3.py_func(lines, 3, 0, fail, 1.0, rx, ry)
    assert ry == pytest.approx(0.15)
    assert _violation(lines, rx, ry).max() == pytest.approx(0.45)


def test_linear_program3_with_contradicting_obstacle_lines_keeps_the_last_result() -> None:
    lines = _lines((0.5, 0.0, 0.0, -1.0), (-0.5, 0.0, 0.0, 1.0), (0.0, 0.3, 1.0, 0.0))
    fail, rx, ry = _linear_program2.py_func(lines, 3, 1.0, 0.0, 0.0, False)
    assert fail == 1
    assert _linear_program3.py_func(lines, 3, 2, fail, 1.0, rx, ry) == (rx, ry)
    assert _linear_program3(lines, 3, 2, fail, 1.0, rx, ry) == (rx, ry)


@pytest.mark.parametrize(
    ("rp", "rv", "on_circle"),
    [((2.0, 0.0), (0.5, 0.0), True), ((2.0, 0.0), (1.0, 1.0), False), ((2.0, 0.0), (1.0, -1.0), False), ((0.0, 3.0), (0.2, 2.0), False)],
    ids=["cutoff", "left_leg", "right_leg", "rotated_leg"],
)
def test_agent_line_moves_relative_velocity_onto_the_obstacle_boundary(rp: tuple[float, float], rv: tuple[float, float], on_circle: bool) -> None:
    cr, inv_tau = 0.5, 0.5
    vx, vy = 0.7, -0.1
    lines = np.zeros((1, 4))
    assert _agent_line.py_func(lines, 0, *rp, *rv, vx, vy, cr, inv_tau, 1.0 / _DT) == 1
    jit_lines = np.zeros((1, 4))
    _agent_line(jit_lines, 0, *rp, *rv, vx, vy, cr, inv_tau, 1.0 / _DT)
    assert lines == pytest.approx(jit_lines, abs=1e-12)
    px, py, dx, dy = lines[0]
    assert math.hypot(dx, dy) == pytest.approx(1.0)
    ux, uy = 2.0 * (px - vx), 2.0 * (py - vy)
    bx, by = rv[0] + ux, rv[1] + uy
    if on_circle:
        assert math.hypot(bx - rp[0] * inv_tau, by - rp[1] * inv_tau) == pytest.approx(cr * inv_tau)
    else:
        assert abs(_det.py_func(dx, dy, bx, by)) == pytest.approx(0.0, abs=1e-12)
        assert math.hypot(bx, by) ** 2 * (rp[0] ** 2 + rp[1] ** 2 - cr**2) == pytest.approx((bx * rp[0] + by * rp[1]) ** 2, rel=1e-9)
    assert ux * dx + uy * dy == pytest.approx(0.0, abs=1e-12)


def test_agent_line_for_overlapping_agents_pushes_apart_within_one_step() -> None:
    inv_dt = 1.0 / _DT
    lines = np.zeros((3, 4))
    n = _agent_line.py_func(lines, 0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, inv_dt)
    n = _agent_line.py_func(lines, n, 0.3, 0.0, 0.3 * inv_dt, 0.0, 0.0, 0.0, 0.5, 0.5, inv_dt)
    assert n == 2
    assert lines[:2] == pytest.approx(np.array([[-2.0, 0.0, 0.0, 1.0], [-5.0, 0.0, 0.0, 1.0]]))
    assert _agent_line.py_func(lines, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, inv_dt) == 2


def test_obstacle_line_for_overlapped_wall_points_away_from_contact() -> None:
    lines = np.zeros((4, 4))
    assert _obstacle_line.py_func(lines, 0, _edge(-1.0, 0.2, 1.0, 0.2), 0.0, 0.0, 0.0, 0.0, 0.3, 0.5) == 1
    assert lines[0] == pytest.approx((0.0, 0.0, -1.0, 0.0))
    assert _obstacle_line.py_func(lines, 0, _edge(0.2, 0.1, 2.0, 0.1), 0.0, 0.0, 0.0, 0.0, 0.3, 0.5) == 1
    d = math.hypot(0.2, 0.1)
    assert lines[0] == pytest.approx((0.0, 0.0, -0.1 / d, 0.2 / d))
    assert _obstacle_line.py_func(lines, 0, _edge(-2.0, 0.1, -0.2, 0.1), 0.0, 0.0, 0.0, 0.0, 0.3, 0.5) == 1
    assert lines[0] == pytest.approx((0.0, 0.0, -0.1 / d, -0.2 / d))
    assert _obstacle_line.py_func(lines, 0, _edge(-2.0, -0.1, -0.2, -0.1), 0.0, 0.0, 0.0, 0.0, 0.3, 0.5) == 0


def test_obstacle_line_already_covered_by_an_earlier_line_is_not_repeated() -> None:
    edge = _edge(-1.0, 1.0, 1.0, 1.0)
    lines = np.zeros((2, 4))
    assert _obstacle_line.py_func(lines, 0, edge, 0.0, 0.0, 0.0, 1.0, 0.3, 0.5) == 1
    assert _obstacle_line.py_func(lines, 1, edge, 0.0, 0.0, 0.0, 1.0, 0.3, 0.5) == 1


def test_obstacle_line_ignores_unrelated_earlier_lines() -> None:
    edge = _edge(-1.0, 1.0, 1.0, 1.0)
    alone = np.zeros((1, 4))
    _obstacle_line.py_func(alone, 0, edge, 0.0, 0.0, 0.0, 1.0, 0.3, 0.5)
    lines = np.array([[5.0, 5.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert _obstacle_line.py_func(lines, 1, edge, 0.0, 0.0, 0.0, 1.0, 0.3, 0.5) == 2
    assert lines[1] == pytest.approx(alone[0])


@pytest.mark.parametrize(
    ("edge", "velocity", "center"),
    [
        ((-1.0, 1.0, 1.0, 1.0), (-0.6, 0.3), (-0.5, 0.5)),
        ((-1.0, 1.0, 1.0, 1.0), (0.6, 0.3), (0.5, 0.5)),
        ((1.0, 0.1, 3.0, 0.1), (0.4, 0.05), (0.5, 0.05)),
        ((-3.0, 0.1, -1.0, 0.1), (-0.4, 0.05), (-0.5, 0.05)),
    ],
    ids=["left_cutoff", "right_cutoff", "oblique_left", "oblique_right"],
)
def test_obstacle_line_near_an_endpoint_is_tangent_to_its_cutoff_circle(edge: tuple[float, float, float, float], velocity: tuple[float, float], center: tuple[float, float]) -> None:
    ra, inv_t = 0.3, 0.5
    lines = np.zeros((1, 4))
    assert _obstacle_line.py_func(lines, 0, _edge(*edge), 0.0, 0.0, *velocity, ra, inv_t) == 1
    px, py, dx, dy = lines[0]
    assert math.hypot(px - center[0], py - center[1]) == pytest.approx(ra * inv_t)
    assert (px - center[0]) * dx + (py - center[1]) * dy == pytest.approx(0.0, abs=1e-12)
    assert _violation(lines, *velocity)[0, 0] == pytest.approx(ra * inv_t - math.hypot(velocity[0] - center[0], velocity[1] - center[1]))


def test_obstacle_line_permits_only_collision_free_velocities() -> None:
    rng = np.random.default_rng(29)
    ra, horizon = 0.3, 2.0
    added = 0
    for _ in range(600):
        ax, ay, bx, by = rng.uniform(-3.0, 3.0, 4)
        if _det.py_func(ax, ay, bx - ax, by - ay) >= 0.0 or math.hypot(bx - ax, by - ay) < 0.2:
            continue
        a, b = np.array([ax, ay]), np.array([bx, by])
        if _seg_seg_dist(np.zeros(2), np.zeros(2), a, b) <= ra:
            continue
        vx, vy = rng.uniform(-2.0, 2.0, 2)
        lines = np.zeros((1, 4))
        n = _obstacle_line.py_func(lines, 0, _edge(ax, ay, bx, by), 0.0, 0.0, vx, vy, ra, 1.0 / horizon)
        jit_lines = np.zeros((1, 4))
        assert _obstacle_line(jit_lines, 0, _edge(ax, ay, bx, by), 0.0, 0.0, vx, vy, ra, 1.0 / horizon) == n
        assert lines == pytest.approx(jit_lines, abs=1e-12)
        assert n == 1
        added += 1
        assert math.hypot(lines[0, 2], lines[0, 3]) == pytest.approx(1.0)
        for wx, wy in rng.uniform(-3.0, 3.0, (40, 2)):
            if _violation(lines, wx, wy)[0, 0] > -1e-6:
                continue
            assert _seg_seg_dist(np.zeros(2), horizon * np.array([wx, wy]), a, b) >= ra - 1e-6
    assert added > 100


def test_obstacle_line_on_walls_seen_end_on_stays_conservative() -> None:
    rng = np.random.default_rng(37)
    ra, horizon = 0.3, 2.0
    added = 0
    for _ in range(1500):
        ang = rng.uniform(-math.pi, math.pi)
        a = rng.uniform(0.5, 1.5) * np.array([math.cos(ang), math.sin(ang)])
        out = ang + rng.uniform(-0.4, 0.4)
        b = a + rng.uniform(0.5, 3.0) * np.array([math.cos(out), math.sin(out)])
        if rng.random() < 0.5:
            a, b = b, a
        if _det.py_func(a[0], a[1], b[0] - a[0], b[1] - a[1]) >= 0.0 or _seg_seg_dist(np.zeros(2), np.zeros(2), a, b) <= ra:
            continue
        vx, vy = rng.uniform(-3.0, 3.0, 2)
        lines = np.zeros((1, 4))
        n = _obstacle_line.py_func(lines, 0, _edge(*a, *b), 0.0, 0.0, vx, vy, ra, 1.0 / horizon)
        jit_lines = np.zeros((1, 4))
        assert _obstacle_line(jit_lines, 0, _edge(*a, *b), 0.0, 0.0, vx, vy, ra, 1.0 / horizon) == n
        assert lines == pytest.approx(jit_lines, abs=1e-12)
        assert n == 1
        added += 1
        for wx, wy in rng.uniform(-3.0, 3.0, (10, 2)):
            if _violation(lines, wx, wy)[0, 0] <= -1e-6:
                assert _seg_seg_dist(np.zeros(2), horizon * np.array([wx, wy]), a, b) >= ra - 1e-6
    assert added > 300


def test_compute_matches_compute_pool(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    planner.set_walls([((-3.0, -1.0), (3.0, -1.0))])
    agents = [agent_factory(agent_id=7, x=0.0, y=0.0), agent_factory(agent_id=9, x=1.0, y=-0.2)]
    agents[0].state.velocity = (0.8, 0.1)
    goals = {7: Pose2D(x=4.0, y=0.0), 9: Pose2D(x=-4.0, y=0.0)}
    per_agent = planner.compute(agents, goals, dt=_DT)
    pool = _pool(agent_factory, [((0.0, 0.0), (4.0, 0.0), (0.8, 0.1)), ((1.0, -0.2), (-4.0, 0.0), (0.0, 0.0))])
    planner.compute_pool(pool, dt=_DT)
    assert per_agent[7] == pytest.approx(tuple(pool.vel[0]), abs=1e-12)
    assert per_agent[9] == pytest.approx(tuple(pool.vel[1]), abs=1e-12)
    assert planner.compute([], {}) == {}


def test_kernel_python_matches_compiled_in_walled_crowd(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(max_neighbors=3)
    walls = [((-4.0, -1.2), (4.0, -1.2)), ((-4.0, 1.2), (4.0, 1.2)), ((1.0, -0.2), (1.0, 0.4)), ((2.0, 0.0), (2.0, 0.0))]
    planner.set_walls(walls)
    assert planner._edges.shape == (6, 6)
    rng = np.random.default_rng(2)
    specs = [((float(x), float(y)), (float(-3.5 * np.sign(x)), 0.0), (float(vx), float(vy))) for x, y, vx, vy in np.column_stack([rng.uniform(-3, 3, 9), rng.uniform(-0.9, 0.9, 9), rng.normal(0, 0.8, (9, 2))])]
    specs.append(((0.0, 0.0), None, (0.0, 0.0)))
    specs.append(((3.0, 0.5), (3.0, 0.5), (0.0, 0.0)))
    pool = _pool(agent_factory, specs)
    out = _kernel_both(planner, pool)
    planner.compute_pool(pool, dt=_DT)
    assert out == pytest.approx(pool.vel[: pool.n], abs=1e-12)
    assert np.all(np.hypot(out[:, 0], out[:, 1]) <= pool.max_velocity[: pool.n] + 1e-9)
    assert out[9].tolist() == [0.0, 0.0]


def test_kernel_resolves_infeasible_crush_with_least_violation(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    ring = [((0.4 * math.cos(a), 0.4 * math.sin(a)), (0.0, 0.0), (-math.cos(a), -math.sin(a))) for a in np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)]
    pool = _pool(agent_factory, [((0.0, 0.0), (5.0, 0.0), (0.0, 0.0)), *ring])
    out = _kernel_both(planner, pool)
    assert np.all(np.isfinite(out))
    assert np.all(np.hypot(out[:, 0], out[:, 1]) <= pool.max_velocity[: pool.n] + 1e-9)


def test_kernel_handles_more_walls_than_query_buffer_and_tied_distances(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    walls = [((-2.0 + 0.05 * k, 1.0), (-1.9 + 0.05 * k, 1.0)) for k in range(70)] + [((-1.0, -1.0), (1.0, -1.0)), ((1.0, -1.0), (-1.0, -1.0)), ((-3.0, -3.6), (3.0, -3.6))]
    planner.set_walls(walls)
    pool = _pool(agent_factory, [((0.0, 0.0), (0.0, 5.0), (0.0, 1.0)), ((0.5, -0.4), (0.5, -5.0), (0.0, -1.0))])
    out = _kernel_both(planner, pool)
    assert out[0, 1] < pool.desired_vel[0]
    assert out[1, 1] > -pool.desired_vel[1]


def test_kernel_neighbor_scan_stops_at_cap_and_missing_entries(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(max_neighbors=1)
    pool = _pool(agent_factory, [((0.0, 0.0), (5.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (-5.0, 0.0), (-1.0, 0.0)), ((0.0, 1.0), (0.0, -5.0), (0.0, -1.0))])
    capped = _kernel_both(planner, pool, np.array([[1, 0], [2, 3], [3, 3]], dtype=np.int64))
    only_first = _kernel_both(planner, pool, np.array([[1, 3], [3, 3], [3, 3]], dtype=np.int64))
    assert capped[0] == pytest.approx(only_first[0], abs=1e-12)
    assert capped[2] == pytest.approx((0.0, -pool.desired_vel[2]))
    alone = _kernel_both(planner, pool, np.array([[0, 3], [1, 3], [2, 3]], dtype=np.int64))
    assert alone[0] == pytest.approx((pool.desired_vel[0], 0.0))


def test_compute_pool_on_empty_pool_leaves_it_untouched() -> None:
    planner = ORCAPlanner()
    pool = AgentPool(capacity=2)
    pool.vel[:] = 7.0
    planner.compute_pool(pool)
    assert np.all(pool.vel == 7.0)
