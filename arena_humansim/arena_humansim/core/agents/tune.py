from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

from arena_humansim.core.agents.types import Poly, SampledParams, TuneDef

if TYPE_CHECKING:
    from arena_humansim.core.agents.base import BaseAgent

CONDITION, SEQUENCE, STEP, NODE = 0, 1, 2, 3


def polyval(poly: Poly, x: float) -> float:
    y = 0.0
    for c in reversed(poly):
        y = y * x + c
    return y


def effective(base: SampledParams, tunes: list[TuneDef]) -> SampledParams:
    top: dict[str, float] = {}
    perception: dict[str, float] = {}
    lp = dict(base.local_planner_params)
    for tune in tunes:
        for key, poly in tune.fields.items():
            group, _, name = key.partition(".")
            if group == "perception":
                perception[name] = polyval(poly, perception.get(name, float(getattr(base.perception, name))))
            elif group == "local_planner_params":
                if name not in lp and len(poly) > 1:
                    raise ValueError(f"tune {key}: {base.name} has no sampled {name} to transform")
                lp[name] = polyval(poly, lp.get(name, 0.0))
            else:
                top[key] = polyval(poly, top.get(key, float(getattr(base, key))))
    changes: dict[str, object] = dict(top)
    if perception:
        changes["perception"] = attrs.evolve(base.perception, **perception)
    if lp != base.local_planner_params:
        changes["local_planner_params"] = lp
    return attrs.evolve(base, **changes)


class Tunes:
    """Keyed tunes over the spawn params; the base is taken at the first set and forgotten when the last key goes."""

    def __init__(self) -> None:
        self.base: SampledParams | None = None
        self.active: dict[str, tuple[int, TuneDef]] = {}
        self.dirty = False

    def set(self, key: str, tune: TuneDef | None, level: int) -> bool:
        if tune is None:
            if key not in self.active:
                return False
            del self.active[key]
            return True
        if self.active.get(key) == (level, tune):
            return False
        self.active[key] = (level, tune)
        return True

    def ordered(self) -> list[TuneDef]:
        """Active tunes folded in owner order: CONDITION, SEQUENCE, STEP, NODE, then by the order taken."""
        return [tune for _, tune in sorted(self.active.values(), key=lambda entry: entry[0])]


def retune(agent: BaseAgent, key: str, tune: TuneDef | None, level: int = NODE) -> bool:
    """Hold `tune` on the agent under `key` at owner `level`, or release that key; returns whether the params changed."""
    tunes = agent.tunes
    if tunes.base is None:
        if tune is None:
            return False
        tunes.base = agent.params
    if not tunes.set(key, tune, level):
        return False
    _refold(agent)
    return True


def rebase(agent: BaseAgent, params: SampledParams) -> None:
    """Replace the spawn params, under whatever tunes are active."""
    if agent.tunes.base is None:
        agent.params = params
        agent.state.desired_velocity = params.desired_velocity
        agent.tunes.dirty = True
        return
    agent.tunes.base = params
    _refold(agent)


def spawn_params(agent: BaseAgent) -> SampledParams:
    base = agent.tunes.base
    return agent.params if base is None else base


def _refold(agent: BaseAgent) -> None:
    tunes = agent.tunes
    assert tunes.base is not None
    if tunes.active:
        params = effective(tunes.base, tunes.ordered())
    else:
        params, tunes.base = tunes.base, None
    agent.params = params
    agent.state.desired_velocity = params.desired_velocity
    tunes.dirty = True
