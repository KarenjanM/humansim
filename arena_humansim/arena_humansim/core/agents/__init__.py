from __future__ import annotations

__all__ = [
    "BaseAgent",
    "Module",
    "TickPhase",
    "VectorizedModule",
    "ActionDef",
    "AgentType",
    "BUILTIN_AGENTS",
    "NeedCondition",
    "NeedDist",
    "ParamDist",
    "SampledNeed",
    "SampledParams",
    "SequenceDef",
    "StepDef",
    "TransitionDef",
    "CONDITION",
    "NODE",
    "SEQUENCE",
    "STEP",
    "Tunes",
    "TuneDef",
    "ConditionalTune",
    "rebase",
    "retune",
    "spawn_params",
    "VarDef",
    "sample_agent_type",
    "create_agent",
    "load_agent_types",
]

from arena_humansim.utils.scenario_loader import converter as _converter

from .base import BaseAgent, Module, TickPhase, VectorizedModule
from .factory import create_agent
from .loader import _load_default_agent_types_raw, load_agent_types, resolve_extends
from .tune import CONDITION, NODE, SEQUENCE, STEP, Tunes, rebase, retune, spawn_params
from .types import ActionDef, AgentType, ConditionalTune, NeedCondition, NeedDist, ParamDist, SampledNeed, SampledParams, SequenceDef, StepDef, TransitionDef, TuneDef, VarDef, sample_agent_type

BUILTIN_AGENTS: dict[str, AgentType] = {name: _converter.structure(raw, AgentType) for name, raw in resolve_extends({name: raw for name, (raw, _) in _load_default_agent_types_raw().items()}, {}).items()}
