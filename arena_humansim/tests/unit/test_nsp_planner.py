from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.nsp.planner import NSPPlanner
from arena_humansim.local_planner.nsp.scaling import (
    assemble_supplement,
    meters_to_pixels,
    pixels_to_meters,
    velocity_from_history,
)


def _seed_pool(pool: AgentPool, positions: list[tuple[float, float]], goals: list[tuple[float, float]]) -> None:
    n = len(positions)
    for i in range(n):
        pool.pos[i] = positions[i]
        pool.goal_pos[i] = goals[i]
        pool.has_goal[i] = True
        pool.max_velocity[i] = 1.5
        pool.desired_vel[i] = 1.3
    pool.neighbor_indptr = np.zeros(n + 1, dtype=np.int32)
    pool.neighbor_indices = np.empty(0, dtype=np.int32)


def _stub_planner(forward_vel_px: np.ndarray | None = None, **kwargs: object) -> NSPPlanner:
    p = NSPPlanner(checkpoint_path="/tmp/nonexistent.pt", **kwargs)
    p._ensure_model = lambda: None  # type: ignore[method-assign]
    if forward_vel_px is None:
        p._forward = lambda *a, **kw: np.zeros((a[0].shape[0], 2), dtype=np.float64)  # type: ignore[method-assign]
    else:
        p._forward = lambda *a, **kw: forward_vel_px[: a[0].shape[0]].copy()  # type: ignore[method-assign]
    return p


def test_meters_pixels_roundtrip() -> None:
    p = np.array([[1.234, -5.678], [0.0, 100.0]])
    assert np.allclose(pixels_to_meters(meters_to_pixels(p, 0.05), 0.05), p, atol=1e-12)


def test_velocity_from_history_uses_nsp_dt() -> None:
    history = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ]
    )
    vel = velocity_from_history(history, nsp_dt=0.4)
    assert vel.shape == (1, 3, 2)
    assert np.allclose(vel[0, 0], [0.0, 0.0])
    assert np.allclose(vel[0, 1], [2.5, 0.0])
    assert np.allclose(vel[0, 2], [2.5, 0.0])


def test_assemble_supplement_zero_pads_and_marks_count() -> None:
    own = np.array([[0.0, 0.0]])
    all_pos = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    all_vel = np.array([[0.1, 0.0], [0.0, 0.1], [0.5, 0.5]])
    indptr = np.array([0, 3], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    supp = assemble_supplement(own, indptr, indices, all_pos, all_vel, max_peds=8)
    assert supp.shape == (1, 9, 5)
    assert int(supp[0, -1, 1]) == 3
    assert np.allclose(supp[0, 0, :2], [1.0, 0.0])
    assert np.allclose(supp[0, 2, 2:4], [0.5, 0.5])
    assert np.all(supp[0, 3:-1, :] == 0.0)


def test_assemble_supplement_truncates_to_closest_when_overflow() -> None:
    own = np.array([[0.0, 0.0]])
    all_pos = np.array([[10.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    all_vel = np.zeros_like(all_pos)
    indptr = np.array([0, 3], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    supp = assemble_supplement(own, indptr, indices, all_pos, all_vel, max_peds=2)
    k = int(supp[0, -1, 1])
    assert k == 2
    kept_x = supp[0, :k, 0]
    assert 10.0 not in set(kept_x)


def test_compute_pool_zero_when_no_agents(pool_empty: Callable[..., AgentPool]) -> None:
    pool = pool_empty(capacity=4)
    p = _stub_planner()
    p.compute_pool(pool, dt=0.05)
    assert pool.n == 0


def test_compute_pool_writes_finite_bounded_velocity(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=2)
    _seed_pool(pool, positions=[(0.0, 0.0), (3.0, 0.0)], goals=[(10.0, 0.0), (-10.0, 0.0)])

    pred_vel_px = np.array([[100.0, 0.0], [-100.0, 0.0]])
    p = _stub_planner(forward_vel_px=pred_vel_px, meters_per_pixel=0.05, nsp_dt=0.4)

    p.compute_pool(pool, dt=0.05)
    v = pool.vel[:2].copy()
    assert np.all(np.isfinite(v))
    speed = np.hypot(v[:, 0], v[:, 1])
    assert np.all(speed <= pool.max_velocity[:2] + 1e-9)
    assert v[0, 0] > 0.0
    assert v[1, 0] < 0.0


def test_dt_bridge_caches_between_nsp_intervals(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=2)
    _seed_pool(pool, positions=[(0.0, 0.0), (3.0, 0.0)], goals=[(10.0, 0.0), (-10.0, 0.0)])

    calls = {"n": 0}

    p = NSPPlanner(checkpoint_path="/tmp/none.pt", meters_per_pixel=0.05, nsp_dt=0.4)
    p._ensure_model = lambda: None  # type: ignore[method-assign]

    def stub_forward(history_translated: np.ndarray, *a: object, **kw: object) -> np.ndarray:
        calls["n"] += 1
        return np.full((history_translated.shape[0], 2), 0.5, dtype=np.float64)

    p._forward = stub_forward  # type: ignore[method-assign]

    for _ in range(8):
        p.compute_pool(pool, dt=0.05)
    assert calls["n"] == 1

    p.compute_pool(pool, dt=0.05)
    assert calls["n"] == 2


def test_has_goal_false_zeros_velocity(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=2)
    _seed_pool(pool, positions=[(0.0, 0.0), (3.0, 0.0)], goals=[(10.0, 0.0), (-10.0, 0.0)])
    pool.has_goal[1] = False

    p = _stub_planner(forward_vel_px=np.array([[5.0, 0.0], [5.0, 0.0]]))
    p.compute_pool(pool, dt=0.05)
    assert np.allclose(pool.vel[1], [0.0, 0.0])


def test_history_buffer_fills_across_calls(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    _seed_pool(pool, positions=[(0.0, 0.0)], goals=[(10.0, 0.0)])

    p = _stub_planner(meters_per_pixel=0.05, nsp_dt=0.4, past_length=4)

    for step in range(10):
        pool.pos[0] = (float(step) * 0.1, 0.0)
        p.compute_pool(pool, dt=0.4)

    aid = int(pool.agent_ids[0])
    buf = list(p._history._buf[aid])
    assert len(buf) == 4


def test_apply_policy_params_overrides_defaults() -> None:
    p = _stub_planner()
    p.apply_policy_params('{"meters_per_pixel": 0.1, "max_peds": 12, "device": "cpu"}')
    assert p._meters_per_pixel == 0.1
    assert p._max_peds == 12
