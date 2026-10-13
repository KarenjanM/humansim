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
    "TuneDef",
    "VarDef",
    "sample_agent_type",
    "create_agent",
    "load_agent_types",
]

from arena_humansim.utils.scenario_loader import converter as _converter

from .base import BaseAgent, Module, TickPhase, VectorizedModule  # noqa: F401
from .factory import create_agent  # noqa: F401
from .loader import _load_default_agent_types_raw, load_agent_types, resolve_extends  # noqa: F401
from .types import ActionDef, AgentType, NeedCondition, NeedDist, ParamDist, SampledNeed, SampledParams, SequenceDef, StepDef, TransitionDef, TuneDef, VarDef, sample_agent_type  # noqa: F401

BUILTIN_AGENTS: dict[str, AgentType] = {name: _converter.structure(raw, AgentType) for name, raw in resolve_extends({name: raw for name, (raw, _) in _load_default_agent_types_raw().items()}, {}).items()}
