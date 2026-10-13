from __future__ import annotations

from collections.abc import Callable

import attrs
import pytest
from arena_humansim.core.agents import CONDITION, SEQUENCE, STEP, BaseAgent, TuneDef, rebase, retune, spawn_params
from arena_humansim.core.agents.tune import effective, polyval


def test_polyval_is_horner() -> None:
    assert polyval((1.0, 2.0, 3.0), 2.0) == 1.0 + 4.0 + 12.0
    assert polyval((0.0, 1.5), 2.0) == 3.0
    assert polyval((2.6,), 99.0) == 2.6


def test_effective_folds_tunes_per_field_in_order(agent_factory: Callable[..., BaseAgent]) -> None:
    base = agent_factory(1).params
    tuned = effective(base, [TuneDef({"desired_velocity": (0.0, 2.0), "perception.vision_range": (2.0,)}), TuneDef({"desired_velocity": (0.1, 1.0), "local_planner_params.repulsion_strength": (0.0, 0.5)})])
    assert tuned.desired_velocity == pytest.approx(base.desired_velocity * 2.0 + 0.1)
    assert tuned.perception.vision_range == 2.0 and tuned.perception.vision_fov == base.perception.vision_fov
    assert tuned.local_planner_params["repulsion_strength"] == pytest.approx(base.local_planner_params["repulsion_strength"] * 0.5)
    assert tuned.max_velocity == base.max_velocity and tuned.agent_radius == base.agent_radius


def test_effective_sets_a_missing_planner_param_and_refuses_to_transform_it(agent_factory: Callable[..., BaseAgent]) -> None:
    base = agent_factory(1).params
    assert effective(base, [TuneDef({"local_planner_params.lookahead": (3.0,)})]).local_planner_params["lookahead"] == 3.0
    with pytest.raises(ValueError, match="no sampled lookahead"):
        effective(base, [TuneDef({"local_planner_params.lookahead": (0.0, 2.0)})])


def test_retune_holds_keys_independently_and_restores_the_base_when_the_last_goes(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(1)
    base = agent.params
    assert retune(agent, "a", TuneDef({"desired_velocity": (0.0, 2.0)}))
    assert retune(agent, "b", TuneDef({"max_velocity": (3.0,)}))
    assert agent.params.desired_velocity == base.desired_velocity * 2.0 and agent.params.max_velocity == 3.0
    assert agent.state.desired_velocity == agent.params.desired_velocity
    assert agent.tunes.dirty
    agent.tunes.dirty = False
    assert retune(agent, "a", None)
    assert agent.params.desired_velocity == base.desired_velocity and agent.params.max_velocity == 3.0
    assert agent.tunes.dirty
    assert retune(agent, "b", None)
    assert agent.params is base and agent.tunes.base is None
    assert agent.state.desired_velocity == base.desired_velocity


def test_retune_is_idempotent_on_unchanged_content(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(1)
    tune = TuneDef({"max_velocity": (3.0,)})
    assert retune(agent, "a", tune)
    agent.tunes.dirty = False
    assert not retune(agent, "a", TuneDef({"max_velocity": (3.0,)}))
    assert not retune(agent, "zzz", None)
    assert not agent.tunes.dirty
    assert retune(agent, "a", TuneDef({"max_velocity": (4.0,)}))
    assert agent.params.max_velocity == 4.0


def test_an_inner_owner_wins_on_the_same_field_until_it_is_released(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(1)
    retune(agent, "sequence", TuneDef({"desired_velocity": (0.0, 2.0)}), SEQUENCE)
    retune(agent, "block", TuneDef({"desired_velocity": (0.5,)}))
    assert agent.params.desired_velocity == 0.5
    retune(agent, "block", None)
    assert agent.params.desired_velocity == agent.tunes.base.desired_velocity * 2.0


def test_fold_order_is_by_owner_level_whatever_the_history(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(1)
    base = agent.params.desired_velocity
    retune(agent, "step", TuneDef({"desired_velocity": (0.0, 2.0)}), STEP)
    retune(agent, "sequence", TuneDef({"desired_velocity": (1.0,)}), SEQUENCE)
    retune(agent, "when/0", TuneDef({"desired_velocity": (0.0, 3.0)}), CONDITION)
    assert agent.params.desired_velocity == 2.0
    retune(agent, "sequence", None)
    assert agent.params.desired_velocity == base * 3.0 * 2.0
    retune(agent, "sequence", TuneDef({"desired_velocity": (1.0,)}), SEQUENCE)
    assert agent.params.desired_velocity == 2.0


def test_rebase_replaces_the_spawn_params_under_active_tunes(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(1)
    retune(agent, "a", TuneDef({"desired_velocity": (0.0, 2.0)}))
    agent.tunes.dirty = False
    rebase(agent, attrs.evolve(spawn_params(agent), desired_velocity=0.5, max_velocity=9.0))
    assert agent.params.desired_velocity == 1.0 and agent.params.max_velocity == 9.0 and agent.tunes.dirty
    retune(agent, "a", None)
    assert agent.params.desired_velocity == 0.5 and agent.params.max_velocity == 9.0 and agent.tunes.base is None
    agent.tunes.dirty = False
    rebase(agent, attrs.evolve(agent.params, max_velocity=2.0))
    assert agent.params.max_velocity == 2.0 and agent.state.desired_velocity == 0.5 and agent.tunes.dirty
