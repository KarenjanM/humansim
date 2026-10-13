from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import py_trees

from arena_humansim.core.agents import BaseAgent, ParamDist, TuneDef, retune
from arena_humansim.core.behavior.nodes.helpers import (
    _at_target,
    _nav_command,
    _sample_param_dist,
)
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import BehaviorTreeMovement, InteractionOutcome, Pose2D, SeekSpec

AgentLookup = Callable[[int], BaseAgent | None]


class SeekNode(py_trees.behaviour.Behaviour):
    """Call im.seek each tick until the agent is bound into a matching interaction."""

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        spec: SeekSpec,
        ctx: StepContext,
        duration_source: ParamDist | None,
        rng: np.random.Generator,
        wait_for_outcome: bool = False,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._spec = spec
        self._ctx = ctx
        self._duration_source = duration_source
        self._rng = rng
        self._wait_for_outcome = wait_for_outcome
        self._duration: float | None = None

    def initialise(self) -> None:
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None
        self._ctx.sought = False

    def _resolved_spec(self) -> SeekSpec:
        spec = self._spec
        if spec.interaction_type.is_object_bound and self._ctx.target_object_id is not None:
            spec = attrs.evolve(spec, target=self._ctx.target_object_id)
        if self._duration != spec.duration:
            spec = attrs.evolve(spec, duration=self._duration)
        return spec

    def update(self) -> py_trees.common.Status:
        mv = self._agent.movement
        bt_mv = mv if isinstance(mv, BehaviorTreeMovement) else None

        if bt_mv is not None and bt_mv.last_outcome is not None:
            outcome = bt_mv.last_outcome
            if outcome == InteractionOutcome.INTERRUPTED:
                bt_mv.last_outcome = None
                return py_trees.common.Status.FAILURE
            if outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.CANCELED):
                bt_mv.last_outcome = None
                return py_trees.common.Status.SUCCESS

        im = self._ctx.im
        agent_id = self._agent.state.agent_id
        resolved_spec = self._resolved_spec()

        if im is not None:
            if im.is_bound_matching(agent_id, resolved_spec):
                wait = self._wait_for_outcome or (self._duration is not None and self._duration > 0)
                if wait:
                    return py_trees.common.Status.RUNNING
                return py_trees.common.Status.SUCCESS
            im.seek(agent_id, resolved_spec)
            self._ctx.sought = True
            return py_trees.common.Status.RUNNING

        # Fallback: no IM reference (tests that don't inject one).
        lookup = self._ctx.is_bound_lookup
        bound = bool(lookup(agent_id)) if lookup is not None else False
        if bound:
            wait = self._wait_for_outcome or (self._duration is not None and self._duration > 0)
            if wait:
                return py_trees.common.Status.RUNNING
            return py_trees.common.Status.SUCCESS
        self._ctx.sought = True
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status


class CancelNode(py_trees.behaviour.Behaviour):
    """Call im.stop/force_stop with reason=CANCELED and return SUCCESS in one tick."""

    def __init__(self, name: str, agent: BaseAgent, im: InteractionManager | None = None) -> None:
        super().__init__(name)
        self._agent = agent
        self._im = im

    def initialise(self) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        im = self._im
        agent_id = self._agent.state.agent_id
        mv = self._agent.movement
        iid: int | None = mv.interaction_id if isinstance(mv, BehaviorTreeMovement) else None

        if im is not None:
            if iid is not None:
                im.stop(agent_id, iid, reason=InteractionOutcome.CANCELED)
            else:
                im.force_stop(agent_id, reason=InteractionOutcome.CANCELED)
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status


class BlockNode(py_trees.behaviour.Behaviour):
    """Pursue a target agent, then seek a BLOCK interaction on arrival."""

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        target_agent_id: int,
        agent_lookup: AgentLookup,
        duration_source: ParamDist | None,
        rng: np.random.Generator,
        ctx: StepContext | None = None,
        tolerance: float = DISTANCE_TOLERANCE,
        lookahead: float = 1.0,
        velocity_boost: float = 1.5,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._target_agent_id = target_agent_id
        self._agent_lookup = agent_lookup
        self._duration_source = duration_source
        self._rng = rng
        self._ctx = ctx
        self._tolerance = tolerance
        self._lookahead = lookahead
        self._velocity_boost = velocity_boost
        self._target: BaseAgent | None = None
        self._duration: float | None = None
        self._sought: bool = False

    def initialise(self) -> None:
        self._target = self._agent_lookup(self._target_agent_id)
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None
        self._sought = False

    def _future_pose(self, target: BaseAgent) -> Pose2D:
        state = target.state
        vx, vy = state.velocity[0], state.velocity[1]
        return Pose2D(
            x=state.pose.x + vx * self._lookahead,
            y=state.pose.y + vy * self._lookahead,
            theta=state.pose.theta,
        )

    def update(self) -> py_trees.common.Status:
        if self._target is None:
            return py_trees.common.Status.FAILURE

        if not self._sought:
            retune(self._agent, self.name, TuneDef({"desired_velocity": (self._target.state.desired_velocity * self._velocity_boost,)}))
            future = self._future_pose(self._target)
            if _at_target(self._agent, future, self._tolerance):
                spec = SeekSpec(
                    interaction_type=InteractionType.BLOCK,
                    target=self._target.state.agent_id,
                    duration=self._duration,
                )
                im = self._ctx.im if self._ctx is not None else None
                if im is not None:
                    im.seek(self._agent.state.agent_id, spec)
                self._sought = True
                return py_trees.common.Status.RUNNING
            self._agent.movement.command = _nav_command(self._agent, future)
            return py_trees.common.Status.RUNNING

        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement) and mv.last_outcome is not None:
            outcome = mv.last_outcome
            if outcome in (InteractionOutcome.COMPLETED, InteractionOutcome.CANCELED):
                mv.last_outcome = None
                return py_trees.common.Status.SUCCESS
            if outcome == InteractionOutcome.INTERRUPTED:
                mv.last_outcome = None
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        retune(self._agent, self.name, None)
        if self._sought:
            im = self._ctx.im if self._ctx is not None else None
            if im is not None:
                im.force_stop(self._agent.state.agent_id, reason=InteractionOutcome.CANCELED)
            self._sought = False
