from __future__ import annotations

import math

import attrs
import pytest
from arena_humansim.core.agents.types import SampledParams, SampledPerception
from arena_humansim.utils.types import AgentState, GestureIntent, Pose2D


@pytest.mark.parametrize("grip", [0.0, 0.4, 1.0])
def test_gesture_intent_accepts_partial_closure(grip: float) -> None:
    assert GestureIntent("receive_r", grip=grip).grip == grip
    assert GestureIntent("receive_r").grip == 0.0


@pytest.mark.parametrize("grip", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_gesture_intent_rejects_invalid_closure(grip: float) -> None:
    with pytest.raises(ValueError, match="grip must be between"):
        GestureIntent("receive_r", grip=grip)


def test_pose2d_default_finite() -> None:
    p = Pose2D()
    assert math.isfinite(p.x)
    assert math.isfinite(p.y)
    assert math.isfinite(p.theta)
    assert p.x == 0.0
    assert p.y == 0.0
    assert p.theta == 0.0


def test_pose2d_equality() -> None:
    assert Pose2D(1.0, 2.0, 0.5) == Pose2D(1.0, 2.0, 0.5)
    assert Pose2D(1.0, 2.0, 0.5) != Pose2D(1.0, 2.0, 0.6)


def test_agent_state_defaults() -> None:
    s = AgentState()
    assert s.agent_id == 0
    assert s.pose == Pose2D()
    assert s.velocity == (0.0, 0.0)
    assert s.desired_velocity == 1.3


def test_agent_state_equality() -> None:
    a = AgentState(agent_id=1, pose=Pose2D(0.0, 0.0, 0.0), velocity=(0.0, 0.0), desired_velocity=1.3)
    b = AgentState(agent_id=1, pose=Pose2D(0.0, 0.0, 0.0), velocity=(0.0, 0.0), desired_velocity=1.3)
    assert a == b


def test_sampled_params_from_conftest_helper(agent_factory) -> None:
    agent = agent_factory(agent_id=1)
    p = agent.params
    assert isinstance(p, SampledParams)
    assert p.name == "adult"
    assert p.desired_velocity == 1.1
    assert p.agent_radius == 0.25
    assert isinstance(p.perception, SampledPerception)
    assert isinstance(p.local_planner_params, dict)
    assert p.perception_stack == ("default",)
    assert p.local_planner == "sfm"
    assert p.global_planner == "dijkstra"
    assert p.animation == "noop"


def test_sampled_params_frozen() -> None:
    p = SampledParams(
        name="x",
        desired_velocity=1.0,
        agent_radius=0.3,
        max_velocity=1.5,
        max_acceleration=1.0,
        max_deceleration=2.0,
        min_turning_radius=0.3,
        pivot_angular_velocity=1.5,
        reaction_time=0.4,
        personal_space_min=0.6,
    )
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        object.__setattr__  # touch so linter keeps import
        setattr(p, "name", "y")


def test_pose2d_mutable_assignment() -> None:
    p = Pose2D()
    p.x = 5.0
    p.y = -2.0
    assert p.x == 5.0
    assert p.y == -2.0
