from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.formation.clearance import Clearance, _crowd_kernel, _sides_kernel
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.wall_grid import WallGrid

_CORRIDOR = [((-10.0, -1.0), (10.0, -1.0)), ((-10.0, 1.5), (10.0, 1.5))]


def _sides(kernel: Callable[..., None], segments: list, poses: np.ndarray, stations: np.ndarray) -> np.ndarray:
    grid = WallGrid(segments)
    out = np.empty((len(poses), 2))
    kernel(poses, stations, grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, out)
    return out


def _attached_pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    states: list[tuple[float, float, float, float]],
    neighbors: list[list[int]],
) -> tuple[Clearance, AgentPool]:
    clearance = Clearance()
    pool = pool_empty(capacity=len(states))
    clearance.attach(pool)
    for k, (x, y, vx, vy) in enumerate(states):
        pool.add_agent(agent_factory(agent_id=10 + k, x=x, y=y))
        pool.vel[k] = (vx, vy)
    indptr = np.cumsum([0] + [len(row) for row in neighbors]).astype(np.int32)
    indices = np.array([j for row in neighbors for j in row], dtype=np.int32)
    pool.set_neighbor_csr(indptr, indices)
    return clearance, pool


def test_sides_kernel_measures_distance_to_corridor_walls_on_each_side() -> None:
    poses = np.array([[0.0, 0.0, 0.0, 5.0], [0.0, 0.0, np.pi, 5.0], [0.0, 0.5, np.pi / 2, 5.0]])
    got = _sides(_sides_kernel.py_func, _CORRIDOR, poses, np.array([0.0]))
    assert got == pytest.approx(np.array([[1.5, 1.0], [1.0, 1.5], [5.0, 5.0]]), abs=1e-12)
    assert np.array_equal(got, _sides(_sides_kernel, _CORRIDOR, poses, np.array([0.0])))


def test_sides_kernel_takes_least_room_over_the_stations() -> None:
    walls = [*_CORRIDOR, ((2.0, -0.3), (4.0, -0.3))]
    poses = np.array([[0.0, 0.0, 0.0, 4.0]])
    assert _sides(_sides_kernel.py_func, walls, poses, np.array([0.0])) == pytest.approx(np.array([[1.5, 1.0]]))
    assert _sides(_sides_kernel.py_func, walls, poses, np.array([0.0, 3.0])) == pytest.approx(np.array([[1.5, 0.3]]))


def test_sides_kernel_without_walls_reports_full_reach() -> None:
    poses = np.array([[3.0, -2.0, 0.4, 2.5]])
    assert _sides(_sides_kernel.py_func, [], poses, np.array([0.0, 1.0])).tolist() == [[2.5, 2.5]]


def test_clearance_sides_and_free_follow_the_walls() -> None:
    clearance = Clearance()
    clearance.set_walls(_CORRIDOR)
    assert clearance.sides(np.array([[0.0, 0.0, 0.0, 5.0]]), np.array([0.0, 1.0])) == pytest.approx(np.array([[1.5, 1.0]]))
    free = clearance.free(np.zeros((2, 2)), np.array([[0.0, 1.0], [1.0, 0.0]]), 3.0)
    assert free == pytest.approx([1.5, 3.0])


def test_crowd_kernel_counts_non_members_inside_the_circle() -> None:
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.5, 0.5]])
    indptr = np.array([0, 4, 4, 4, 4, 4], dtype=np.int64)
    indices = np.array([1, 2, 3, 4], dtype=np.int64)
    rows = np.array([0, 0, -1, 7], dtype=np.int64)
    centers = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])
    r2 = np.full(4, 2.25)
    member_ptr = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    member_rows = np.array([4], dtype=np.int64)
    out = np.full(4, -1, dtype=np.int64)
    _crowd_kernel.py_func(rows, centers, r2, indptr, indices, pos, member_ptr, member_rows, out)
    assert out.tolist() == [3, 2, 0, 0]
    jit_out = np.full(4, -1, dtype=np.int64)
    _crowd_kernel(rows, centers, r2, indptr, indices, pos, member_ptr, member_rows, jit_out)
    assert np.array_equal(out, jit_out)


def test_crowds_counts_perceived_agents_and_skips_excluded_and_unknown(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    states = [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0)]
    clearance, _ = _attached_pool(pool_empty, agent_factory, states, [[1, 2, 3], [0], [0], [0]])
    got = clearance.crowds([10, 10, 99], np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]), 1.0, [set(), {11, 99}, set()])
    assert got.tolist() == [2, 1, 0]
    per_row = clearance.crowds([10, 10], np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([1.0, 5.0]), [set(), set()])
    assert per_row.tolist() == [2, 3]


def test_crowds_without_pool_or_observers_is_all_zero(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    detached = Clearance()
    assert detached.crowds([1, 2], np.zeros((2, 2)), 1.0, [set(), set()]).tolist() == [0, 0]
    clearance, _ = _attached_pool(pool_empty, agent_factory, [(0.0, 0.0, 0.0, 0.0)], [[]])
    assert clearance.crowds([], np.zeros((0, 2)), 1.0, []).shape == (0,)


def test_oncoming_counts_agents_walking_against_heading_by_side(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    states = [
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 1.0, -1.0, 0.0),
        (2.0, 0.5, -0.5, 0.3),
        (2.0, -1.0, -1.0, 0.0),
        (2.0, -0.5, 1.0, 0.0),
        (9.0, 1.0, -1.0, 0.0),
        (1.0, 1.0, -1.0, 0.0),
    ]
    clearance, _ = _attached_pool(pool_empty, agent_factory, states, [[1, 2, 3, 4, 5, 6], [], [], [], [], [], []])
    assert clearance.oncoming(10, 0.0, 0.0, 4.0, 0.0, set()) == (3, 1)
    assert clearance.oncoming(10, 0.0, 0.0, 4.0, 0.0, {11, 16}) == (1, 1)
    assert clearance.oncoming(10, 0.0, 0.0, 4.0, np.pi, set()) == (1, 0)


def test_oncoming_is_empty_for_missing_pool_observer_or_neighbor_table(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    assert Clearance().oncoming(10, 0.0, 0.0, 4.0, 0.0, set()) == (0, 0)
    clearance, pool = _attached_pool(pool_empty, agent_factory, [(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, -1.0, 0.0)], [[1], [0]])
    assert clearance.oncoming(42, 0.0, 0.0, 4.0, 0.0, set()) == (0, 0)
    assert clearance.oncoming(10, 0.0, 0.0, 4.0, 0.0, set()) == (0, 1)
    pool.set_neighbor_csr(np.zeros(1, dtype=np.int32), np.empty(0, dtype=np.int32))
    assert clearance.oncoming(10, 0.0, 0.0, 4.0, 0.0, set()) == (0, 0)
