from __future__ import annotations

import py_trees

from arena_humansim.core.agents import BaseAgent, SampledParams, SequenceDef, TuneDef
from arena_humansim.core.behavior.nodes.helpers import _bt_logger
from arena_humansim.core.behavior.nodes.utility import preconditions_met
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.utils.types import BehaviorTreeMovement, InteractionOutcome


class SequenceStateMachine(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        sequences: dict[str, py_trees.behaviour.Behaviour],
        sequence_defs: dict[str, SequenceDef],
        initial: str,
        agent: BaseAgent,
        im: InteractionManager | None = None,
    ) -> None:
        super().__init__(name=name)
        self._sequences = sequences
        self._sequence_defs = sequence_defs
        self._initial = initial
        self._agent = agent
        self._im = im
        self._current_name: str = initial
        self._current_node: py_trees.behaviour.Behaviour = sequences[initial]
        # the agent's params and speed from before the running sequence's `tune`; None while untuned
        self._untuned: tuple[SampledParams, float] | None = None

    def initialise(self) -> None:
        self._current_name = self._initial
        self._current_node = self._sequences[self._initial]
        self._retune(self._sequence_defs[self._initial].tune)
        self._current_node.initialise()

    def update(self) -> py_trees.common.Status:
        redirect = self._check_transitions()
        if redirect is not None and redirect in self._sequences and self._only_retunes(redirect):
            # same steps, other tune: the running step and any interaction it holds carry on
            self._retune(self._sequence_defs[redirect].tune)
            self._current_name = redirect
        elif redirect is not None:
            return self._goto(redirect)

        self._current_node.tick_once()
        status = self._current_node.status

        if status == py_trees.common.Status.SUCCESS:
            seq_def = self._sequence_defs[self._current_name]
            if seq_def.then is None:
                return py_trees.common.Status.SUCCESS
            return self._goto(seq_def.then)

        if status == py_trees.common.Status.FAILURE:
            seq_def = self._sequence_defs[self._current_name]
            if seq_def.on_failure is None:
                return py_trees.common.Status.FAILURE
            return self._goto(seq_def.on_failure)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        self._current_node.terminate(new_status)
        self._clear_gestures()
        self._retune(None)

    def _only_retunes(self, target: str) -> bool:
        here, there = self._sequence_defs[self._current_name], self._sequence_defs[target]
        return target != self._current_name and bool(here.steps) and there.steps == here.steps and there.attention == here.attention

    def _retune(self, tune: TuneDef | None) -> None:
        """Hold `tune` over the untuned params, or put those back when `tune` is None."""
        if tune is None and self._untuned is None:
            return
        agent = self._agent
        if self._untuned is None:
            self._untuned = (agent.params, agent.state.desired_velocity)
        params, speed = self._untuned
        if tune is None:
            self._untuned = None
        else:
            params, speed = tune.apply(params, speed)
        agent.params = params
        agent.state.desired_velocity = speed
        if isinstance(agent.movement, BehaviorTreeMovement):
            agent.movement.retuned = True

    def _clear_gestures(self) -> None:
        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement):
            mv.gestures = ()
            mv.heading_goal = None

    def _goto(self, target: str) -> py_trees.common.Status:
        if target not in self._sequences:
            _bt_logger.warning(f'Agent {self._agent.state.agent_id}: invalid transition target "{target}"')
            return py_trees.common.Status.FAILURE
        _bt_logger.debug(f"Agent {self._agent.state.agent_id}: {self._current_name} -> {target}")
        # stop() cascades terminate() down composites; terminate() alone is a leaf-only hook.
        self._current_node.stop(py_trees.common.Status.INVALID)
        self._clear_gestures()
        # Evict from lingering interaction memberships only when switching sequences. A self-loop
        # (e.g. chat `then: chat`) means the agent is continuing the same behavior - ejecting them
        # would kick everyone out of a group conversation the moment it re-enters the seek step.
        if target != self._current_name and self._im is not None:
            self._im.force_stop(self._agent.state.agent_id, reason=InteractionOutcome.INTERRUPTED)
        # after stop(), so a node restoring its own speed in terminate() restores the tuned one
        if target != self._current_name:
            self._retune(self._sequence_defs[target].tune)
        self._current_name = target
        self._current_node = self._sequences[target]
        self._current_node.initialise()
        return py_trees.common.Status.RUNNING

    def _check_transitions(self) -> str | None:
        seq_def = self._sequence_defs.get(self._current_name)
        if seq_def is None:
            return None
        needs = self._agent.needs.needs if self._agent.needs else {}
        for transition in seq_def.transitions:
            if preconditions_met(needs, transition.when):
                return transition.goto
        return None
