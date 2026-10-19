import numpy as np
import py_trees

from arena_humansim.core.agents import CONDITION, STEP, BaseAgent, ConditionalTune, ParamDist, TuneDef, retune
from arena_humansim.core.behavior.nodes.helpers import _nav_command, _sample_param_dist, _set_command
from arena_humansim.core.behavior.nodes.utility import preconditions_met
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.utils import DT
from arena_humansim.utils.types import BehaviorTreeMovement


class ClearOutcomeNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, agent: BaseAgent) -> None:
        super().__init__(name)
        self._agent = agent

    def update(self) -> py_trees.common.Status:
        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement):
            mv.last_outcome = None
            mv.interaction_id = None
        return py_trees.common.Status.SUCCESS


class PatienceWatchdogNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        patience_source: ParamDist | None,
        rng: np.random.Generator,
        dt: float,
    ) -> None:
        super().__init__(name)
        self._patience_source = patience_source
        self._rng = rng
        self._dt = dt
        self._patience: float | None = None
        self._elapsed: float = 0.0

    def initialise(self) -> None:
        self._elapsed = 0.0
        self._patience = _sample_param_dist(self._patience_source, self._rng) if self._patience_source is not None else None

    def update(self) -> py_trees.common.Status:
        # Never returns SUCCESS - Parallel(SuccessOnOne) tracks the sibling Sequence instead.
        if self._patience is None:
            return py_trees.common.Status.RUNNING
        if self._elapsed >= self._patience:
            return py_trees.common.Status.FAILURE
        self._elapsed += self._dt
        return py_trees.common.Status.RUNNING


class HoldNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        duration_source: ParamDist | None,
        rng: np.random.Generator,
        dt: float,
        ctx: StepContext | None = None,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._duration_source = duration_source
        self._rng = rng
        self._dt = dt
        self._ctx = ctx
        self._duration: float | None = None
        self._elapsed: float = 0.0

    def initialise(self) -> None:
        self._elapsed = 0.0
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None

    def _bound(self) -> bool:
        lookup = self._ctx.is_bound_lookup if self._ctx is not None else None
        return bool(lookup(self._agent.state.agent_id)) if lookup is not None else False

    def update(self) -> py_trees.common.Status:
        if self._duration is None:
            return py_trees.common.Status.SUCCESS
        # Bound agents are driven by the formation emitter; overwriting movement.command
        # here would freeze followers mid-ride. Let formation own motion, just tick time.
        if not self._bound():
            cmd = _nav_command(self._agent, self._agent.state.pose)
            cmd.desired_velocity = 0.0
            _set_command(self._agent, cmd)
        if self._elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        self._elapsed += self._dt
        return py_trees.common.Status.RUNNING


class SatisfyNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        satisfies: dict[str, float],
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._satisfies = satisfies

    def update(self) -> py_trees.common.Status:
        if self._agent.needs is not None and self._satisfies:
            self._agent.needs.satisfy(self._satisfies)
        return py_trees.common.Status.SUCCESS


class NeedsDecayNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, agent: BaseAgent, dt: float = DT) -> None:
        super().__init__(name=name)
        self._agent = agent
        self._dt = dt

    def update(self) -> py_trees.common.Status:
        if self._agent.needs is not None:
            self._agent.needs.decay(self._dt)
        return py_trees.common.Status.RUNNING


class TuneNode(py_trees.behaviour.Behaviour):
    """Holds `tune` on the agent under its own name from initialise to terminate."""

    def __init__(self, name: str, agent: BaseAgent, tune: TuneDef) -> None:
        super().__init__(name=name)
        self._agent = agent
        self._tune = tune

    def initialise(self) -> None:
        retune(self._agent, self.name, self._tune, STEP)

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        retune(self._agent, self.name, None)


class ConditionalTunesNode(py_trees.behaviour.Behaviour):
    """Holds each `tunes:` entry of the agent type while its `when` holds on the needs."""

    def __init__(self, name: str, agent: BaseAgent, tunes: tuple[ConditionalTune, ...]) -> None:
        super().__init__(name=name)
        self._agent = agent
        self._tunes = tunes

    def update(self) -> py_trees.common.Status:
        needs = self._agent.needs.needs if self._agent.needs is not None else {}
        for i, entry in enumerate(self._tunes):
            retune(self._agent, f"{self.name}/{i}", entry.tune if preconditions_met(needs, entry.when) else None, CONDITION)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        for i in range(len(self._tunes)):
            retune(self._agent, f"{self.name}/{i}", None)
