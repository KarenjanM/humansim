"""gesture_mode: the control arm in which agents behave the same but publish no gesture."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_humansim_msgs")
pytest.importorskip("rclpy")

from rclpy.parameter import Parameter

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_manager import CONTACT_ENABLED, GESTURE_DISABLED, GESTURE_ENABLED, check_gesture_mode


class _InteractionManager:
    contact_mode = CONTACT_ENABLED

    def set_contact_mode(self, mode: str, distance: float | None) -> None:
        del mode, distance


class _Node:
    """Enough of AgentManager for the parameter callback."""

    def __init__(self) -> None:
        self._interaction_manager = _InteractionManager()
        self._gesture_mode = GESTURE_ENABLED
        self._gestures_dirty = False


def _set(node: _Node, **params: object):
    return AgentManager._on_contact_params(node, [Parameter(name, value=value) for name, value in params.items()])


def test_gesture_mode_switches_and_rejects_unknown() -> None:
    node = _Node()
    assert _set(node, gesture_mode=GESTURE_DISABLED).successful
    assert node._gesture_mode == GESTURE_DISABLED
    assert _set(node, gesture_mode=GESTURE_ENABLED).successful
    assert node._gesture_mode == GESTURE_ENABLED

    result = _set(node, gesture_mode="off")
    assert not result.successful and "gesture_mode" in result.reason
    assert node._gesture_mode == GESTURE_ENABLED


def test_a_mode_switch_republishes_the_latched_gestures() -> None:
    node = _Node()
    assert _set(node, gesture_mode=GESTURE_ENABLED).successful
    assert not node._gestures_dirty
    assert _set(node, gesture_mode=GESTURE_DISABLED).successful
    assert node._gestures_dirty


def test_unrelated_parameters_pass_through() -> None:
    node = _Node()
    assert _set(node, contact_mode=CONTACT_ENABLED, locomotion_standing_distance=1.2).successful
    assert node._gesture_mode == GESTURE_ENABLED


def test_check_gesture_mode_passes_known_and_rejects_unknown() -> None:
    assert check_gesture_mode(GESTURE_DISABLED) == GESTURE_DISABLED
    with pytest.raises(ValueError, match="gesture_mode"):
        check_gesture_mode("off")
