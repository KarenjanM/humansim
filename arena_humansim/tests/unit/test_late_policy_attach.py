"""Planners attached to the pool after agents exist."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.utils.types import Pose2D


def _populate(pool: AgentPool, agent_factory: Callable[..., BaseAgent], n: int = 3) -> list[BaseAgent]:
    agents = []
    for i in range(n):
        agent = agent_factory(agent_id=i + 1, x=float(i) * 0.8, y=0.0)
        agents.append(agent)
        idx = pool.add_agent(agent)
        pool.policy_idx[idx] = 0
    pool.set_goals({a.state.agent_id: Pose2D(x=10.0, y=0.0) for a in agents})
    pool.set_neighbor_csr(np.zeros(n + 1, dtype=np.int32), np.empty(0, dtype=np.int32))
    return agents


def test_unattached_pool_planner_fails_loud(pool_empty, agent_factory) -> None:
    pool = pool_empty(capacity=8)
    attached = SFMPlanner()
    attached.attach(pool)
    _populate(pool, agent_factory)

    orphan = SFMPlanner()
    with pytest.raises(RuntimeError, match="never attached to the pool"):
        orphan.compute_pool(pool, dt=0.05)


def test_attach_late_backfills_every_row(pool_empty, agent_factory) -> None:
    pool = pool_empty(capacity=8)
    default = SFMPlanner()
    default.attach(pool)
    agents = _populate(pool, agent_factory)

    late = HSFMPlanner()
    pool.attach_late(late, agents)

    n = pool.n
    assert late._relaxation_time[:n].shape == (n,)
    assert np.all(late._relaxation_time[:n] > 0.0)
    assert np.all(late._angular_gain[:n] > 0.0)
    late.compute_pool(pool, dt=0.05)


def test_update_agent_reaches_inherited_and_own_rows_of_hsfm(pool_empty, agent_factory) -> None:
    pool = pool_empty(capacity=8)
    planner = HSFMPlanner()
    planner.attach(pool)
    agent = agent_factory(agent_id=1)
    i = pool.add_agent(agent)
    lp = {**agent.params.local_planner_params, "relaxation_time": 0.9, "angular_gain": 7.0}
    agent.params = attrs.evolve(agent.params, local_planner_params=lp)
    pool.update_agent(agent)
    assert (planner._relaxation_time[i], planner._angular_gain[i]) == (0.9, 7.0)


def test_attach_late_covers_agents_added_afterwards(pool_empty, agent_factory) -> None:
    pool = pool_empty(capacity=8)
    SFMPlanner().attach(pool)
    agents = _populate(pool, agent_factory, n=2)
    late = HSFMPlanner()
    pool.attach_late(late, agents)

    added = agent_factory(agent_id=99, x=5.0, y=0.0)
    idx = pool.add_agent(added)
    assert late._relaxation_time[idx] > 0.0


def test_param_falls_back_to_class_default(pool_empty, agent_factory) -> None:
    """An agent sampled under sfm carries no hsfm keys, and hsfm must not KeyError on it."""
    pool = pool_empty(capacity=8)
    SFMPlanner().attach(pool)
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    stripped = {k: v for k, v in agent.params.local_planner_params.items() if not k.startswith(("lateral", "angular"))}
    agent.params = attrs.evolve(agent.params, local_planner_params=stripped)
    pool.add_agent(agent)

    late = HSFMPlanner()
    pool.attach_late(late, [agent])
    assert late._lateral_gain[0] == pytest.approx(HSFMPlanner.PARAM_DEFAULTS["lateral_gain"].mean)
    assert late._angular_gain[0] == pytest.approx(HSFMPlanner.PARAM_DEFAULTS["angular_gain"].mean)


def test_late_planner_changes_the_trajectory(pool_empty, agent_factory) -> None:
    """From rest toward a goal straight ahead sfm and hsfm agree, so roll both with an off-axis goal."""

    def roll(late: SFMPlanner | None) -> np.ndarray:
        pool = pool_empty(capacity=8)
        planner = SFMPlanner()
        planner.attach(pool)
        agents = _populate(pool, agent_factory)
        if late is not None:
            pool.attach_late(late, agents)
            planner = late
        n = pool.n
        pool.theta[:n] = 1.2
        pool.set_goals({a.state.agent_id: Pose2D(x=2.0, y=8.0) for a in agents})
        for _ in range(20):
            planner.compute_pool(pool, dt=0.05)
            pool.pos[:n] += pool.vel[:n] * 0.05
        return pool.pos[:n].copy()

    assert not np.allclose(roll(None), roll(HSFMPlanner())), "hsfm and sfm produced identical trajectories"
