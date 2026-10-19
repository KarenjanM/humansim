import numpy as np
import py_trees

from arena_humansim.core.agents import ActionDef, BaseAgent, StepDef
from arena_humansim.core.behavior.nodes.helpers import _sample_param_dist, _set_command
from arena_humansim.core.behavior.nodes.utility import preconditions_met, score_actions
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils import DT
from arena_humansim.utils.event_bus import EventBus

_RUNNING = py_trees.common.Status.RUNNING


class AutonomousNode(py_trees.behaviour.Behaviour):
    """Score the actions while idle and run the winner's compiled subtree to its end."""

    def __init__(
        self,
        name: str,
        step_def: StepDef,
        agent: BaseAgent,
        action_defs: dict[str, ActionDef],
        action_trees: dict[str, py_trees.behaviour.Behaviour],
        utility_weights: dict[str, float],
        world: WorldKnowledge,
        event_bus: EventBus,
        rng: np.random.Generator,
        dt: float = DT,
    ) -> None:
        super().__init__(name=name)
        self._step = step_def
        self._agent = agent
        self._world = world
        self._event_bus = event_bus
        self._rng = rng
        self._dt = dt
        self._utility_weights = utility_weights

        self._actions = self._filter_actions(action_defs)
        self._trees = action_trees

        self._duration: float | None = None
        self._elapsed: float = 0.0
        self._running: py_trees.behaviour.Behaviour | None = None

    def _filter_actions(self, action_defs: dict[str, ActionDef]) -> dict[str, ActionDef]:
        if self._step.allowed_actions is not None:
            allowed = set(self._step.allowed_actions)
            return {k: v for k, v in action_defs.items() if k in allowed}
        if self._step.blocked_actions is not None:
            blocked = set(self._step.blocked_actions)
            return {k: v for k, v in action_defs.items() if k not in blocked}
        return dict(action_defs)

    def initialise(self) -> None:
        self._elapsed = 0.0
        self._duration = _sample_param_dist(self._step.duration, self._rng) if self._step.duration is not None else None

    def update(self) -> py_trees.common.Status:
        agent_id = self._agent.state.agent_id
        needs = self._agent.needs.needs if self._agent.needs else {}

        if self._step.until is not None:
            if self._event_bus.has(self._step.until, agent_id):
                self._event_bus.consume(self._step.until, agent_id)
                return py_trees.common.Status.SUCCESS

        if self._step.until_need is not None:
            if preconditions_met(needs, self._step.until_need):
                return py_trees.common.Status.SUCCESS

        if self._running is None and self._duration is not None and self._elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        self._elapsed += self._dt

        if self._running is None:
            scored = score_actions(needs, self._actions, self._utility_weights, self._world)
            if not scored:
                _set_command(self._agent, None)
                return _RUNNING
            self._running = self._trees[scored[0][0]]

        self._running.tick_once()
        if self._running.status != _RUNNING:
            self._running = None
        return _RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        if self._running is not None:
            self._running.stop(py_trees.common.Status.INVALID)
            self._running = None
