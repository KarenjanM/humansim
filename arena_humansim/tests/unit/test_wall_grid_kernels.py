from __future__ import annotations

import numpy as np
import pytest

from arena_humansim.utils.wall_grid import WallGrid, _cast_kernel, cast_one, cast_rays, query_walls


def _grid_args(grid: WallGrid) -> tuple:
    return (grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny)


def _ladder(n: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [((float(k), 0.0), (float(k), 1.0)) for k in range(n)]


def _reference_cast(seg: np.ndarray, ox: float, oy: float, dx: float, dy: float, cap: float) -> float:
    best = cap
    for ax, ay, bx, by in seg:
        m = np.array([[dx, ax - bx], [dy, ay - by]])
        if abs(np.linalg.det(m)) <= 1e-12:
            continue
        t, u = np.linalg.solve(m, [ax - ox, ay - oy])
        if 0.0 <= t < best and 0.0 <= u <= 1.0:
            best = float(t)
    return best


def test_query_walls_python_matches_compiled_and_lists_each_wall_once_in_order() -> None:
    rng = np.random.default_rng(11)
    walls = [((float(x0), float(y0)), (float(x1), float(y1))) for x0, y0, x1, y1 in rng.uniform(0.0, 12.0, size=(40, 4))]
    grid = WallGrid(walls, cell=1.5)
    for x, y, r in zip(rng.uniform(-2.0, 14.0, 60), rng.uniform(-2.0, 14.0, 60), rng.uniform(0.0, 5.0, 60), strict=True):
        py_out = np.empty(64, dtype=np.int64)
        jit_out = np.empty(64, dtype=np.int64)
        k = query_walls.py_func(*_grid_args(grid), float(x), float(y), float(r), py_out)
        assert k == query_walls(*_grid_args(grid), float(x), float(y), float(r), jit_out)
        found = py_out[:k]
        assert np.array_equal(found, jit_out[:k])
        assert np.all(np.diff(found) > 0)
        a = grid.segments[:, :2]
        d = grid.segments[:, 2:] - a
        t = np.clip(((x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]) / np.maximum((d * d).sum(axis=1), 1e-12), 0.0, 1.0)
        near = np.flatnonzero(np.hypot(x - a[:, 0] - t * d[:, 0], y - a[:, 1] - t * d[:, 1]) <= r)
        assert set(near.tolist()) <= set(found.tolist())


def test_query_walls_overflow_keeps_the_smallest_ids_and_counts_all() -> None:
    grid = WallGrid(_ladder(10), cell=1.0)
    full = np.empty(16, dtype=np.int64)
    k_full = query_walls.py_func(*_grid_args(grid), 4.5, 0.5, 10.0, full)
    small = np.empty(3, dtype=np.int64)
    k_small = query_walls.py_func(*_grid_args(grid), 4.5, 0.5, 10.0, small)
    assert k_full == k_small == 10
    assert np.array_equal(full[:10], np.arange(10))
    assert np.array_equal(small, [0, 1, 2])


def test_query_walls_outside_the_grid_finds_nothing() -> None:
    grid = WallGrid(_ladder(4), cell=1.0)
    out = np.empty(8, dtype=np.int64)
    assert query_walls.py_func(*_grid_args(grid), 100.0, 100.0, 1.0, out) == 0


def test_cast_one_hits_perpendicular_wall_at_its_distance() -> None:
    grid = WallGrid([((2.0, -1.0), (2.0, 1.0))])
    buf = np.empty(1, dtype=np.int64)
    assert cast_one.py_func(0.0, 0.0, 1.0, 0.0, 10.0, grid.segments, *_grid_args(grid), buf) == pytest.approx(2.0)
    assert cast_one.py_func(0.0, 0.0, 0.5, 0.0, 10.0, grid.segments, *_grid_args(grid), buf) == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("origin", "direction"),
    [((0.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (-1.0, 0.0)), ((0.0, 2.0), (1.0, 0.0))],
    ids=["parallel", "behind", "past_endpoint"],
)
def test_cast_one_ignores_parallel_rearward_and_missed_walls(origin: tuple[float, float], direction: tuple[float, float]) -> None:
    grid = WallGrid([((2.0, -1.0), (2.0, 1.0))])
    buf = np.empty(1, dtype=np.int64)
    assert cast_one.py_func(origin[0], origin[1], direction[0], direction[1], 7.5, grid.segments, *_grid_args(grid), buf) == 7.5


def test_cast_one_returns_nearest_of_stacked_walls() -> None:
    grid = WallGrid([((3.0, -1.0), (3.0, 1.0)), ((1.5, -1.0), (1.5, 1.0)), ((5.0, -1.0), (5.0, 1.0))])
    buf = np.empty(3, dtype=np.int64)
    assert cast_one.py_func(0.0, 0.0, 1.0, 0.0, 10.0, grid.segments, *_grid_args(grid), buf) == pytest.approx(1.5)


def test_cast_one_short_ray_through_large_grid_uses_the_cell_query() -> None:
    walls = [((k + 0.5, -0.5), (k + 0.5, 0.5)) for k in range(30)]
    grid = WallGrid(walls, cell=1.0)
    buf = np.empty(len(walls), dtype=np.int64)
    for x0 in (0.1, 7.2, 20.9):
        got = cast_one.py_func(x0, 0.0, 1.0, 0.0, 1.0, grid.segments, *_grid_args(grid), buf)
        assert got == pytest.approx(_reference_cast(grid.segments, x0, 0.0, 1.0, 0.0, 1.0))
        assert got == cast_one(x0, 0.0, 1.0, 0.0, 1.0, grid.segments, *_grid_args(grid), buf)


def test_cast_kernel_python_matches_compiled_and_reference() -> None:
    rng = np.random.default_rng(5)
    walls = [((float(x0), float(y0)), (float(x1), float(y1))) for x0, y0, x1, y1 in rng.uniform(-6.0, 6.0, size=(25, 4))]
    grid = WallGrid(walls, cell=1.0)
    origins = rng.uniform(-5.0, 5.0, size=(40, 2))
    angles = rng.uniform(-np.pi, np.pi, 40)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    cap = rng.uniform(0.2, 8.0, 40)
    py_out = np.empty(40)
    _cast_kernel.py_func(origins, dirs, cap, grid.segments, *_grid_args(grid), py_out)
    assert np.array_equal(py_out, cast_rays(grid, origins, dirs, cap))
    for i in range(40):
        assert py_out[i] == pytest.approx(_reference_cast(grid.segments, *origins[i], *dirs[i], cap[i]), abs=1e-9)


def test_cast_rays_without_walls_or_rays_returns_the_cap() -> None:
    empty = WallGrid([])
    assert np.array_equal(cast_rays(empty, np.zeros((3, 2)), np.ones((3, 2)), 4.0), np.full(3, 4.0))
    walled = WallGrid([((1.0, -1.0), (1.0, 1.0))])
    assert cast_rays(walled, np.zeros((0, 2)), np.zeros((0, 2)), 4.0).shape == (0,)


def test_cast_rays_broadcasts_scalar_and_per_ray_caps() -> None:
    grid = WallGrid([((1.0, -1.0), (1.0, 1.0))])
    origins = np.zeros((2, 2))
    dirs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert np.allclose(cast_rays(grid, origins, dirs, 5.0), [1.0, 5.0])
    assert np.allclose(cast_rays(grid, origins, dirs, np.array([0.5, 2.0])), [0.5, 2.0])
