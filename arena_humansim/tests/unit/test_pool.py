from __future__ import annotations

from collections.abc import Callable

import attrs
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool, human_mask, is_human


def test_add_agent_grows_n(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    assert pool.n == 0
    for i in range(5):
        idx = pool.add_agent(agent_factory(agent_id=i + 1, x=float(i)))
        assert idx == i
        assert pool.n == i + 1


def test_update_agent_rewrites_params_and_keeps_the_posture_radius(pool_with_agents: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_with_agents(3)
    agent = agent_factory(agent_id=2)
    i = pool.idx(2)
    pool.agent_radius[i] = 0.9
    agent.params = attrs.evolve(agent.params, max_velocity=4.0, perception=attrs.evolve(agent.params.perception, vision_range=2.0))
    agent.state.desired_velocity = 0.3
    assert pool.update_agent(agent) == i
    assert (pool.max_velocity[i], pool.vision_range[i], pool.desired_vel[i], pool.agent_radius[i]) == (4.0, 2.0, 0.3, 0.9)


def test_multiple_adds_preserve_monotonic_n(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=6)
    assert pool.n == 6
    assert set(pool._id_to_idx.keys()) == {1, 2, 3, 4, 5, 6}


def test_swap_remove_reduces_n(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    pool.swap_remove(2)
    assert pool.n == 3
    assert 2 not in pool._id_to_idx


def test_swap_remove_last_gets_removed_index(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    removed_idx = pool._id_to_idx[2]
    swapped = pool.swap_remove(2)
    assert swapped == 4
    assert pool._id_to_idx[4] == removed_idx


def test_swap_remove_last_element_returns_none(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    last_id = int(pool.agent_ids[pool.n - 1])
    swapped = pool.swap_remove(last_id)
    assert swapped is None
    assert pool.n == 2


def test_id_to_idx_consistent_after_arbitrary_removes(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=6)
    for aid in (3, 1, 5):
        pool.swap_remove(aid)
    assert pool.n == 3
    for aid, idx in pool._id_to_idx.items():
        assert int(pool.agent_ids[idx]) == aid
    assert set(pool._id_to_idx.keys()) == {2, 4, 6}


def test_pos_vel_shape_constant(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    assert pool.pos.shape == (8, 2)
    assert pool.vel.shape == (8, 2)
    for i in range(5):
        pool.add_agent(agent_factory(agent_id=i + 1))
    assert pool.pos.shape == (8, 2)
    assert pool.vel.shape == (8, 2)
    pool.swap_remove(2)
    assert pool.pos.shape == (8, 2)
    assert pool.vel.shape == (8, 2)


def test_reset_clears_n_and_id_map(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    pool.reset()
    assert pool.n == 0
    assert pool._id_to_idx == {}


def test_add_remove_add_cycle_keeps_bijection(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    for i in range(5):
        pool.add_agent(agent_factory(agent_id=i + 1))
    pool.swap_remove(2)
    pool.swap_remove(4)
    pool.add_agent(agent_factory(agent_id=100))
    pool.add_agent(agent_factory(agent_id=101))
    assert pool.n == 5
    for aid, idx in pool._id_to_idx.items():
        assert int(pool.agent_ids[idx]) == aid
    seen_idx = sorted(pool._id_to_idx.values())
    assert seen_idx == list(range(pool.n))


def test_idx_returns_stored_index(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    assert pool.idx(1) == pool._id_to_idx[1]


def test_swap_remove_missing_raises(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    with pytest.raises(KeyError):
        pool.swap_remove(999)


def test_grow_beyond_capacity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=2)
    for i in range(5):
        pool.add_agent(agent_factory(agent_id=i + 1))
    assert pool.n == 5
    assert pool.capacity >= 5
    assert pool.pos.shape == (pool.capacity, 2)
    assert pool.vel.shape == (pool.capacity, 2)


def test_kind_and_policy_columns_default(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=4)
    idx = pool.add_agent(agent_factory(agent_id=1))
    assert pool.kind.shape == (4,)
    assert pool.policy_idx.shape == (4,)
    assert int(pool.kind[idx]) == 0
    assert int(pool.policy_idx[idx]) == -1


def test_kind_and_policy_set_per_row(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=4)
    idx0 = pool.add_agent(agent_factory(agent_id=1))
    idx1 = pool.add_agent(agent_factory(agent_id=2))
    pool.kind[idx0] = 0
    pool.policy_idx[idx0] = 0
    pool.kind[idx1] = 1
    pool.policy_idx[idx1] = -1
    assert int(pool.kind[idx1]) == 1
    assert int(pool.policy_idx[idx1]) == -1
    assert int(pool.kind[idx0]) == 0
    assert int(pool.policy_idx[idx0]) == 0


def test_swap_remove_preserves_kind_and_policy(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=4)
    pool.add_agent(agent_factory(agent_id=1))
    pool.add_agent(agent_factory(agent_id=2))
    pool.add_agent(agent_factory(agent_id=3))
    pool.kind[2] = 1
    pool.policy_idx[2] = 7
    pool.swap_remove(1)
    assert int(pool.kind[0]) == 1
    assert int(pool.policy_idx[0]) == 7


def test_human_mask_and_is_human(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    pool.add_agent(agent_factory(agent_id=1))
    pool.add_agent(agent_factory(agent_id=2))
    pool.add_agent(agent_factory(agent_id=3))
    pool.kind[1] = 1

    mask = human_mask(pool)
    assert mask.tolist() == [True, False, True]
    assert is_human(pool, 0) is True
    assert is_human(pool, 1) is False
    assert is_human(pool, 2) is True


def test_grow_preserves_kind_and_policy(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=2)
    pool.add_agent(agent_factory(agent_id=1))
    pool.kind[0] = 1
    pool.policy_idx[0] = 3
    for i in range(5):
        pool.add_agent(agent_factory(agent_id=i + 10))
    assert int(pool.kind[0]) == 1
    assert int(pool.policy_idx[0]) == 3
    assert pool.kind.shape == (pool.capacity,)
    assert pool.policy_idx.shape == (pool.capacity,)
