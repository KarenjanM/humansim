"""cattrs structuring hooks for scenario/agent-type YAML loading."""

import attrs
import cattrs

from arena_humansim.core.agents.types import (
    ARM_CHANNELS,
    CHANNEL_SLOTS,
    HEADING_SOURCES,
    ActionDef,
    AgentType,
    AttentionDef,
    AttentionRef,
    AttentionStepDef,
    CadenceDist,
    ChannelDef,
    ClipDef,
    GoToStepDef,
    LocomotionDist,
    NeedCondition,
    NeedDist,
    PerceptionDist,
    Pose3,
    RelativeRef,
    RobotRef,
    SequenceDef,
    StepDef,
    TransitionDef,
    VarDef,
    _as_paramdist,
)
from arena_humansim.core.interaction_kinds import POSTURES, HandleKind, InteractionType
from arena_humansim.utils.types import AnchorKind, FormationSpec, Pose2D, WaypointMode

converter = cattrs.Converter()


def _structure_waypoint_mode(val: object, _: type) -> WaypointMode:
    if isinstance(val, WaypointMode):
        return val
    if isinstance(val, str):
        try:
            return WaypointMode[val.upper()]
        except KeyError:
            valid = ", ".join(m.name.lower() for m in WaypointMode)
            raise ValueError(f"unknown waypoint_mode {val!r}; valid: {valid}") from None
    if isinstance(val, int):
        return WaypointMode(val)
    raise ValueError(f"waypoint_mode must be a string or int, got {type(val).__name__}")


converter.register_structure_hook(WaypointMode, _structure_waypoint_mode)


def _structure_need_dist(val: object, _: type) -> NeedDist:
    if isinstance(val, (int, float)):
        return NeedDist(initial=float(val))
    if isinstance(val, NeedDist):
        return val
    return NeedDist(**dict(val))


converter.register_structure_hook(NeedDist, _structure_need_dist)


def _structure_need_condition(val: object, _: type) -> NeedCondition:
    if isinstance(val, NeedCondition):
        return val
    return NeedCondition(**val)


converter.register_structure_hook(NeedCondition, _structure_need_condition)


def _structure_action_def(val: object, _: type) -> ActionDef:
    if isinstance(val, ActionDef):
        return val
    d = dict(val)
    if "attention" in d or d.get("kind") == "attention":
        raise ValueError("'attention' is not supported in the autonomous 'actions' library")
    if "when" in d and isinstance(d["when"], dict):
        d["when"] = {k: converter.structure(v, NeedCondition) for k, v in d["when"].items()}
    return ActionDef(**d)


converter.register_structure_hook(ActionDef, _structure_action_def)


_STEP_KINDS = ("object_interact", "go_to", "attention")

_ADVANCES = ("dwell", "unreachable")
_HOLDS = ("release", "keep")
_CHANNEL_KEYS = ("at", "dwell", "advance", "hold", "at_z")
_BLOCK_KEYS = (*CHANNEL_SLOTS, "clip", "posture", "face", "required")
_CLIP_WHEN = ("always", "bound")


def _structure_attention_ref(val: object) -> AttentionRef:
    if isinstance(val, (Pose3, RobotRef, RelativeRef)):
        return val
    if isinstance(val, bool):
        raise ValueError(f"attention 'at' ref must not be a bool, got {val!r}")
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        if not val:
            raise ValueError("attention 'at' ref must be a non-empty string")
        if val.startswith("robot:"):
            name = val[len("robot:") :]
            if not name:
                raise ValueError("attention 'at: robot:<name>' requires a name")
            return RobotRef(name)
        return val
    if isinstance(val, dict):
        keys = set(val)
        if keys == {"x", "y", "z"}:
            return Pose3(x=float(val["x"]), y=float(val["y"]), z=float(val["z"]))
        if keys in ({"azimuth", "elevation"}, {"azimuth", "elevation", "distance"}):
            for k in keys:
                if isinstance(val[k], bool) or not isinstance(val[k], (int, float)):
                    raise ValueError(f"attention relative ref {k!r} must be a number, got {val[k]!r}")
            distance = float(val.get("distance", 3.0))
            if distance <= 0:
                raise ValueError("attention relative ref 'distance' must be > 0")
            return RelativeRef(azimuth=float(val["azimuth"]), elevation=float(val["elevation"]), distance=distance)
        raise ValueError(f"attention 'at' mapping must be {{x, y, z}} or {{azimuth, elevation[, distance]}}, got keys {sorted(keys)}")
    raise ValueError(f"attention 'at' ref must be a keyword, object id, agent name, robot:<name>, agent id, {{x, y, z}} or {{azimuth, elevation}}, got {val!r}")


def _structure_targets(val: object, where: str) -> tuple[AttentionRef, ...]:
    if isinstance(val, (list, tuple)):
        if not val:
            raise ValueError(f"attention {where} 'at' list must not be empty")
        return tuple(_structure_attention_ref(r) for r in val)
    return (_structure_attention_ref(val),)


def _structure_channel(name: str, val: object) -> ChannelDef:
    if isinstance(val, ChannelDef):
        return val
    if not isinstance(val, dict) or "at" not in val:
        if isinstance(val, dict) and set(val) & set(_CHANNEL_KEYS):
            raise ValueError(f"attention {name} mapping requires 'at'")
        return ChannelDef(at=_structure_targets(val, name))
    d = dict(val)
    unknown = set(d) - set(_CHANNEL_KEYS)
    if unknown:
        raise ValueError(f"unknown attention {name} fields: {sorted(unknown)}")
    refs = _structure_targets(d["at"], name)
    dwell = d.get("dwell", 1.0)
    if isinstance(dwell, bool) or not isinstance(dwell, (int, float)) or dwell <= 0:
        raise ValueError(f"attention {name} 'dwell' must be a number > 0, got {dwell!r}")
    advance = d.get("advance", "dwell")
    if advance not in _ADVANCES:
        raise ValueError(f"attention {name} 'advance' must be one of {_ADVANCES}, got {advance!r}")
    hold = d.get("hold", "release")
    if hold not in _HOLDS:
        raise ValueError(f"attention {name} 'hold' must be one of {_HOLDS}, got {hold!r}")
    at_z = d.get("at_z")
    if at_z is not None:
        if isinstance(at_z, bool) or not isinstance(at_z, (int, float)):
            raise ValueError(f"attention {name} 'at_z' must be a number, got {at_z!r}")
        if any(isinstance(r, (Pose3, RelativeRef)) for r in refs):
            raise ValueError(f"attention {name} 'at_z' is only valid with entity refs, not literal or relative ones")
        at_z = float(at_z)
    return ChannelDef(at=refs, dwell=float(dwell), advance=advance, hold=hold, at_z=at_z)


def _structure_face(val: object) -> bool | AttentionRef | None:
    if val is None or val == "auto":
        return None
    if isinstance(val, bool):
        return val
    if val == "true":
        return True
    if val == "false":
        return False
    try:
        ref = _structure_attention_ref(val)
    except ValueError:
        raise ValueError(f"attention 'face' must be auto | true | false | <ref>, got {val!r}") from None
    if isinstance(ref, RelativeRef):
        raise ValueError("attention 'face' ref must not be relative, {azimuth, elevation} never drives face")
    return ref


def _structure_clip(val: object) -> ClipDef:
    if isinstance(val, ClipDef):
        return val
    if isinstance(val, str):
        val = {"name": val}
    if not isinstance(val, dict):
        raise ValueError(f"attention 'clip' must be a clip name or {{name[, when]}}, got {val!r}")
    unknown = set(val) - {"name", "when"}
    if unknown:
        raise ValueError(f"unknown attention clip fields: {sorted(unknown)}")
    name = val.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("attention clip requires a non-empty 'name'")
    when = val.get("when", "always")
    if when not in _CLIP_WHEN:
        raise ValueError(f"attention clip 'when' must be one of {_CLIP_WHEN}, got {when!r}")
    return ClipDef(name=name, when=when)


def _structure_attention_def(val: object) -> AttentionDef:
    if isinstance(val, AttentionDef):
        return val
    if not isinstance(val, dict):
        raise ValueError(f"'attention' must be a mapping, got {type(val).__name__}")
    d = dict(val)
    unknown = set(d) - set(_BLOCK_KEYS)
    if unknown:
        raise ValueError(f"unknown attention fields: {sorted(unknown)}")
    channels = {name: _structure_channel(name, d[name]) for name in CHANNEL_SLOTS if d.get(name) is not None}
    clip = _structure_clip(d["clip"]) if d.get("clip") is not None else None
    posture = d.get("posture", "")
    if posture not in ("", *POSTURES):
        raise ValueError(f"attention 'posture' must be one of {POSTURES}, got {posture!r}")
    if not channels and clip is None and not posture:
        raise ValueError(f"attention requires at least one channel: {[*CHANNEL_SLOTS, 'clip', 'posture']}")
    arms = [name for name in ARM_CHANNELS if name in channels]
    if len(arms) > 1:
        raise ValueError(f"attention takes exactly one arm channel ({' | '.join(ARM_CHANNELS)}), got {arms}")
    face = _structure_face(d.get("face"))
    required = d.get("required", False)
    if not isinstance(required, bool):
        raise ValueError(f"attention 'required' must be a bool, got {required!r}")
    att = AttentionDef(face=face, required=required, clip=clip, posture=posture, **channels)
    if face is True:
        winner = att.face_channel()
        assert winner is not None
        if isinstance(channels[winner].at[0], RelativeRef):
            raise ValueError(f"attention 'face: true' is not valid when the first entry of {winner!r} is a relative {{azimuth, elevation}} ref")
    return att


def _check_bare_attention(att: AttentionDef, raw: dict, timed: bool) -> None:
    if raw.get("required") is False:
        raise ValueError("bare attention steps are always required, drop 'required: false'")
    if not timed:
        cycling = [name for name, ch in att.channels().items() if ch.advance == "unreachable"]
        if cycling:
            raise ValueError(f"bare attention step with cycling channel(s) {cycling} needs 'duration' or 'patience'")
        if not att.channels():
            raise ValueError("bare attention step with only a clip or posture needs 'duration' or 'patience'")


def _structure_go_to_step_def(val: object, _: type) -> GoToStepDef:
    if isinstance(val, GoToStepDef):
        return val
    d = dict(val)
    d.pop("kind", None)
    has_pose = "target_pose" in d and d["target_pose"] is not None
    has_target = "target" in d and d["target"] is not None
    if has_pose and has_target:
        raise ValueError("go_to step accepts either 'target_pose: {x, y}' or 'target: <id|type>', not both")
    if not has_pose and not has_target:
        raise ValueError("go_to step requires either 'target_pose: {x, y}' or 'target: <id|type>'")
    if has_pose:
        pose = d["target_pose"]
        if isinstance(pose, Pose2D):
            d["target_pose"] = pose
        else:
            pose_d = dict(pose)
            d["target_pose"] = Pose2D(x=float(pose_d["x"]), y=float(pose_d["y"]), theta=float(pose_d.get("theta", 0.0)))
    if d.get("attention") is not None:
        d["attention"] = _structure_attention_def(d["attention"])
    known = {f.name for f in attrs.fields(GoToStepDef)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown go_to step fields: {sorted(unknown)}")
    return GoToStepDef(**d)


converter.register_structure_hook(GoToStepDef, _structure_go_to_step_def)


def _structure_attention_step_def(val: object, _: type) -> AttentionStepDef:
    if isinstance(val, AttentionStepDef):
        return val
    d = dict(val)
    explicit_kind = d.pop("kind", None) is not None
    if "attention" not in d:
        raise ValueError("attention step requires an 'attention:' block")
    raw_att = d["attention"]
    att = _structure_attention_def(raw_att)
    _check_bare_attention(att, raw_att if isinstance(raw_att, dict) else {}, d.get("duration") is not None or d.get("patience") is not None)
    d["attention"] = attrs.evolve(att, required=True)
    known = {f.name for f in attrs.fields(AttentionStepDef)}
    unknown = set(d) - known
    if unknown:
        step_only = sorted(unknown & {f.name for f in attrs.fields(StepDef)})
        if step_only and not explicit_kind:
            raise ValueError(f"step has attention plus interaction-only fields {step_only}, add kind or interaction")
        raise ValueError(f"unknown attention step fields: {sorted(unknown)}")
    return AttentionStepDef(**d)


converter.register_structure_hook(AttentionStepDef, _structure_attention_step_def)


def _structure_step_def(val: object, _: type) -> StepDef:
    if isinstance(val, StepDef):
        return val
    d = dict(val)
    d.pop("kind", None)
    if "accept" in d:
        raise ValueError("step field 'accept' removed; drop it and use 'interaction:' (absence of 'offer: true' means seeker)")
    if "interaction" in d and d["interaction"] == "FOLLOW":
        raise ValueError("interaction FOLLOW removed; use 'interaction: SERVICE, target: <tag>, offer: true, min_participants: 2, max_participants: 2, queueable: false, formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}'")
    for legacy in ("target_object_id", "target_object_type", "target_agent", "service_tag"):
        if legacy in d:
            raise ValueError(f"step field {legacy!r} removed; use 'target:' instead (meaning determined by interaction type)")
    interaction = d.get("interaction")
    offer = bool(d.get("offer", False))
    target = d.get("target")
    kind = None
    if interaction is not None:
        try:
            kind = InteractionType[interaction].kind
        except KeyError:
            raise ValueError(f"unknown interaction type {interaction!r}") from None
    if offer and (kind is None or not kind.allows_offer):
        raise ValueError(f"'offer: true' is not valid for interaction={interaction!r}")
    provider_fields = ("queueable", "min_participants", "max_participants", "formation_spec")
    for pf in provider_fields:
        if pf == "formation_spec" and kind is not None and kind.render_pose_override:
            continue
        if pf in d and d[pf] is not None and not offer:
            raise ValueError(f"step field {pf!r} is provider-side configuration; requires 'offer: true'")
    if kind is not None and not d.get("cancel", False):
        hk = kind.handle.kind
        strategy = kind.handle.strategy
        shapes = strategy.target_shape
        if hk == HandleKind.NONE:
            if target is not None:
                raise ValueError(f"interaction: {interaction} takes no target; 'target' must not be set")
        elif hk == HandleKind.TAG:
            if offer and not isinstance(target, shapes):
                raise ValueError(f"interaction: {interaction} with 'offer: true' requires 'target: <tag:str>'")
            if not offer and target is not None and not isinstance(target, shapes):
                raise ValueError(f"interaction: {interaction} seeker 'target' must be a tag string or omitted")
        elif strategy.target_required_for_create and not isinstance(target, shapes):
            expected = " | ".join(t.__name__ for t in shapes)
            raise ValueError(f"interaction: {interaction} requires 'target: <{expected}>'")
    if "until_need" in d and isinstance(d["until_need"], dict):
        d["until_need"] = {k: converter.structure(v, NeedCondition) for k, v in d["until_need"].items()}
    if "allowed_actions" in d and d["allowed_actions"] is not None:
        d["allowed_actions"] = tuple(d["allowed_actions"])
    if "blocked_actions" in d and d["blocked_actions"] is not None:
        d["blocked_actions"] = tuple(d["blocked_actions"])
    if "formation_spec" in d and isinstance(d["formation_spec"], dict):
        fs = d["formation_spec"]
        pose_cfg = fs.get("anchor_pose")
        anchor_pose = None
        if pose_cfg is not None:
            anchor_pose = Pose2D(
                x=float(pose_cfg.get("x", 0.0)),
                y=float(pose_cfg.get("y", 0.0)),
                theta=float(pose_cfg.get("theta", 0.0)),
            )
        raw_kind = fs.get("anchor_kind", "object")
        try:
            anchor_kind_val = AnchorKind(raw_kind)
        except ValueError:
            raise ValueError(f"Unknown anchor_kind {raw_kind!r}. Valid: {[k.value for k in AnchorKind]}") from None
        d["formation_spec"] = FormationSpec(
            type=fs["type"],
            params=dict(fs.get("params", {})),
            anchor_kind=anchor_kind_val,
            anchor_ref=fs.get("anchor_ref"),
            anchor_pose=anchor_pose,
        )
    if d.get("attention") is not None:
        if d.get("autonomous", False):
            raise ValueError("'attention' is not supported on 'autonomous: true' steps")
        d["attention"] = _structure_attention_def(d["attention"])
    known = {f.name for f in attrs.fields(StepDef)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown step fields: {sorted(unknown)}")
    return StepDef(**d)


converter.register_structure_hook(StepDef, _structure_step_def)


_BARE_ATTENTION_EXCLUDES = ("interaction", "target", "target_pose", "cancel", "autonomous")


def _structure_step_variant(val: object) -> StepDef | GoToStepDef | AttentionStepDef:
    if isinstance(val, (StepDef, GoToStepDef, AttentionStepDef)):
        return val
    d = dict(val)
    kind = d.get("kind")
    if kind is None:
        bare = "attention" in d and not any(k in d for k in _BARE_ATTENTION_EXCLUDES)
        kind = "attention" if bare else "object_interact"
    if kind == "go_to":
        return converter.structure(val, GoToStepDef)
    if kind == "attention":
        return converter.structure(val, AttentionStepDef)
    if kind == "object_interact":
        return converter.structure(val, StepDef)
    raise ValueError(f"unknown step kind {kind!r}; expected one of {_STEP_KINDS}")


def _structure_transition_def(val: object, _: type) -> TransitionDef:
    if isinstance(val, TransitionDef):
        return val
    d = dict(val)
    when = d["when"]
    if isinstance(when, str):
        raise ValueError(f"String-based transition condition '{when}' is not supported. Use dict[str, NeedCondition] instead.")
    d["when"] = {k: converter.structure(v, NeedCondition) for k, v in when.items()}
    return TransitionDef(**d)


converter.register_structure_hook(TransitionDef, _structure_transition_def)


def _structure_sequence_def(val: object, _: type) -> SequenceDef:
    if isinstance(val, SequenceDef):
        return val
    d = dict(val)
    if "steps" in d:
        d["steps"] = {k: _structure_step_variant(v) for k, v in d["steps"].items()}
    if "transitions" in d:
        d["transitions"] = tuple(converter.structure(t, TransitionDef) for t in d["transitions"])
    else:
        d["transitions"] = ()
    if d.get("attention") is not None:
        d["attention"] = _structure_attention_def(d["attention"])
    return SequenceDef(**d)


converter.register_structure_hook(SequenceDef, _structure_sequence_def)


def _structure_perception_dist(val: object, _: type) -> PerceptionDist:
    if isinstance(val, PerceptionDist):
        return val
    return PerceptionDist(**dict(val))


converter.register_structure_hook(PerceptionDist, _structure_perception_dist)


_LOCOMOTION_KEYS = ("kinematics", "cadence", "phase_warp", "speed_profile", "lateral_profile", "footprint_length", "recovery")
_CADENCE_KEYS = ("base", "per_speed", "min", "max")


def _sub_block(d: dict, key: str, allowed: tuple[str, ...]) -> dict:
    block = d.get(key)
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"locomotion '{key}' must be a mapping, got {type(block).__name__}")
    unknown = set(block) - set(allowed)
    if unknown:
        raise ValueError(f"unknown locomotion {key} fields: {sorted(unknown)}, expected {list(allowed)}")
    return block


def _structure_locomotion_dist(val: object, _: type) -> LocomotionDist:
    if isinstance(val, LocomotionDist):
        return val
    d = dict(val) if val is not None else {}
    unknown = set(d) - set(_LOCOMOTION_KEYS)
    if unknown:
        raise ValueError(f"unknown locomotion fields: {sorted(unknown)}, expected {list(_LOCOMOTION_KEYS)}")
    flat: dict = {}
    if "kinematics" in d:
        flat["kinematics"] = d["kinematics"]
    if "footprint_length" in d:
        flat["footprint_length"] = d["footprint_length"]
    cadence = _sub_block(d, "cadence", _CADENCE_KEYS)
    if cadence:
        flat["cadence"] = CadenceDist(**cadence)
    warp = _sub_block(d, "phase_warp", ("split",))
    if "split" in warp:
        flat["phase_warp_split"] = warp["split"]
    speed = _sub_block(d, "speed_profile", ("harmonics", "amplitude_scale"))
    if "harmonics" in speed:
        flat["speed_profile"] = speed["harmonics"]
    if "amplitude_scale" in speed:
        flat["speed_amplitude_scale"] = speed["amplitude_scale"]
    lateral = _sub_block(d, "lateral_profile", ("harmonics",))
    if "harmonics" in lateral:
        flat["lateral_profile"] = lateral["harmonics"]
    recovery = _sub_block(d, "recovery", ("stall_after_s", "reverse_m"))
    if "stall_after_s" in recovery:
        flat["recovery_stall_after_s"] = recovery["stall_after_s"]
    if "reverse_m" in recovery:
        flat["recovery_reverse_m"] = recovery["reverse_m"]
    return LocomotionDist(**flat)


converter.register_structure_hook(LocomotionDist, _structure_locomotion_dist)


def _structure_local_planner_param(key: str, val: object) -> object:
    if key == "heading_source" and isinstance(val, str):
        if val not in HEADING_SOURCES:
            raise ValueError(f"heading_source must be one of {sorted(HEADING_SOURCES)} or a number, got {val!r}")
        return _as_paramdist(HEADING_SOURCES[val])
    return _as_paramdist(val)


def _structure_agent_type(val: object, _: type) -> AgentType:
    if isinstance(val, AgentType):
        return val
    d = dict(val)
    if "perception" in d and isinstance(d["perception"], dict):
        d["perception"] = converter.structure(d["perception"], PerceptionDist)
    if "local_planner_params" in d and isinstance(d["local_planner_params"], dict):
        d["local_planner_params"] = {k: _structure_local_planner_param(k, v) for k, v in d["local_planner_params"].items()}
    if "locomotion" in d:
        d["locomotion"] = converter.structure(d["locomotion"], LocomotionDist)
        d["locomotion_active"] = True
    if "assets" in d and isinstance(d["assets"], list):
        d["assets"] = tuple(d["assets"])
    if "needs" in d and isinstance(d["needs"], dict):
        d["needs"] = {k: converter.structure(v, NeedDist) for k, v in d["needs"].items()}
    if "actions" in d and isinstance(d["actions"], dict):
        d["actions"] = {k: converter.structure(v, ActionDef) for k, v in d["actions"].items()}
    if "sequences" in d and isinstance(d["sequences"], dict):
        d["sequences"] = {k: converter.structure(v, SequenceDef) for k, v in d["sequences"].items()}
    if "vars" in d and isinstance(d["vars"], dict):
        d["vars"] = {k: VarDef(**v) if isinstance(v, dict) else v for k, v in d["vars"].items()}
    if "perception_stack" in d and isinstance(d["perception_stack"], list):
        d["perception_stack"] = tuple(d["perception_stack"])
    return AgentType(**d)


converter.register_structure_hook(AgentType, _structure_agent_type)


def _structure_var_def(val: object, _: type) -> VarDef:
    if isinstance(val, VarDef):
        return val
    return VarDef(**val)


converter.register_structure_hook(VarDef, _structure_var_def)
