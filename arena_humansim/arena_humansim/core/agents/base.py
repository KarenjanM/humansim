from __future__ import annotations

__all__ = [
    "BaseAgent",
    "Module",
    "TickPhase",
    "VectorizedModule",
]

import enum
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from arena_humansim.animation import MotionAnimation
    from arena_humansim.core.pool import AgentPool
    from arena_humansim.global_planner import GlobalPlanner
    from arena_humansim.local_planner import LocalPlanner
    from arena_humansim.perception import Perception

import attrs

from arena_humansim.utils.types import AgentState, BehaviorTreeMovement, BeliefState, NeedsState, WaypointMovement

from .tune import Tunes
from .types import SampledParams


class TickPhase(enum.IntEnum):
    SENSE = 0
    PLAN = 1
    ACT = 2


@runtime_checkable
class Module(Protocol):
    def phase(self) -> TickPhase: ...
    def step_batch(self, agents: Iterable[BaseAgent], dt: float) -> None: ...


@runtime_checkable
class VectorizedModule(Protocol):
    def phase(self) -> TickPhase: ...
    def step_pool(self, pool: AgentPool, n: int, dt: float) -> None: ...


@attrs.define
class BaseAgent:
    state: AgentState
    params: SampledParams

    global_planner: GlobalPlanner
    local_planner: LocalPlanner
    animation: MotionAnimation

    perception: list[Perception] = attrs.Factory(list)
    modules: dict[str, Module] = attrs.Factory(dict)
    belief: BeliefState | None = None
    movement: WaypointMovement | BehaviorTreeMovement = attrs.Factory(WaypointMovement)
    needs: NeedsState | None = None
    tunes: Tunes = attrs.Factory(Tunes)
