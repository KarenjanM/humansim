from __future__ import annotations

import numpy as np
import pytest
from arena_humansim.core.agents import loader
from arena_humansim.core.agents.types import (
    AgentType,  # noqa: F401
    sample_agent_type,
)

KINEMATIC = ("adult", "elder", "child", "hurried", "distracted")
FIELDS = (
    "desired_velocity",
    "agent_radius",
    "max_velocity",
    "max_acceleration",
    "max_deceleration",
    "min_turning_radius",
    "pivot_angular_velocity",
)


@pytest.fixture(scope="module")
def types() -> dict[str, object]:
    loaded = loader.load_agent_types()
    missing = set(KINEMATIC) - set(loaded)
    if missing:
        pytest.skip(f"builtin types not on this install: {sorted(missing)}")
    return loaded


def test_every_kinematic_type_loads_as_simple_mode(types: dict[str, object]) -> None:
    for name in KINEMATIC:
        assert types[name].mode == "simple", name


def test_extends_gives_every_type_the_full_field_set(types: dict[str, object]) -> None:
    # A knob that lands on `adult` must land on every type derived from it.
    for name in KINEMATIC:
        t = types[name]
        for field in FIELDS:
            assert getattr(t, field) is not None, f"{name}.{field}"
        assert t.perception.vision_range is not None and t.perception.vision_fov is not None
        assert {"relaxation_time", "repulsion_strength", "repulsion_range"} <= set(t.local_planner_params)


def test_distributions_are_well_formed(types: dict[str, object]) -> None:
    for name in KINEMATIC:
        t = types[name]
        for field in FIELDS:
            d = getattr(t, field)
            assert d.clip_low <= d.mean <= d.clip_high, f"{name}.{field}: {d}"
            assert d.std >= 0


def test_the_types_differ_where_they_are_meant_to(types: dict[str, object]) -> None:
    radius = {n: types[n].agent_radius.mean for n in KINEMATIC}
    speed = {n: types[n].desired_velocity.mean for n in KINEMATIC}
    assert radius["child"] < radius["adult"]
    assert speed["elder"] < speed["distracted"] < speed["adult"] < speed["hurried"]
    assert types["distracted"].perception.vision_range.mean < types["adult"].perception.vision_range.mean


def test_every_type_samples(types: dict[str, object]) -> None:
    rng = np.random.default_rng(3)
    for name in KINEMATIC:
        p = sample_agent_type(types[name], rng)
        t = types[name]
        assert t.desired_velocity.clip_low <= p.desired_velocity <= t.desired_velocity.clip_high
        assert t.agent_radius.clip_low <= p.agent_radius <= t.agent_radius.clip_high
