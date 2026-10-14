from __future__ import annotations

import pytest

from arena_humansim.core.agents.types import AgentType, AttentionDef, AttentionStepDef, ChannelDef, ClipDef, GoToStepDef, Pose3, RelativeRef, RobotRef, SequenceDef, StepDef, TuneDef
from arena_humansim.utils.scenario import _structure_manual
from arena_humansim.utils.scenario_loader import converter


def _minimal(extra: dict | None = None) -> dict:
    data: dict = {"name": "t", "simulation": {"seed": 1, "dt": 0.05, "max_ticks": 1}, "modules": {}}
    if extra:
        data.update(extra)
    return data


def test_valid_object_bound_step_with_target_type_loads() -> None:
    data = _minimal(
        {
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "rest": {
                            "steps": {
                                "sit_step": {
                                    "interaction": "SIT_ON",
                                    "target": "bench",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["rest"].steps["sit_step"].target == "bench"


def test_unresolvable_step_target_raises() -> None:
    data = _minimal(
        {
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "rest": {
                            "steps": {
                                "go_atm": {
                                    "interaction": "USE",
                                    "target": "atm",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target="):
        _structure_manual(data)


def test_unresolvable_action_target_raises() -> None:
    data = _minimal(
        {
            "world_objects": [],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "actions": {
                        "sit": {
                            "interaction": "SIT_ON",
                            "target": "chair",
                            "satisfies": {"rest": 50.0},
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target="):
        _structure_manual(data)


def test_valid_target_object_id_loads() -> None:
    data = _minimal(
        {
            "world_objects": [
                {"object_id": "atm_main", "type": "atm", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1},
                {"object_id": "atm_backup", "type": "atm", "pose": {"x": 5.0, "y": 0.0, "theta": 0.0}, "capacity": 1},
            ],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "queue": {
                            "steps": {
                                "go_specific_atm": {
                                    "interaction": "QUEUE_USE",
                                    "target": "atm_backup",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["queue"].steps["go_specific_atm"].target == "atm_backup"


def test_none_target_skips_validation() -> None:
    data = _minimal(
        {
            "world_objects": [],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"idle": {"steps": {"pause": {"duration": {"mean": 1.0, "std": 0.0, "clip_low": 0.1, "clip_high": 5.0}}}}},
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["idle"].steps["pause"].target is None


def _with_step(fields: dict) -> dict:
    return _minimal(
        {
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"step": fields}}},
                }
            },
        }
    )


def test_legacy_accept_field_raises() -> None:
    with pytest.raises(ValueError, match="'accept' removed"):
        _structure_manual(_with_step({"accept": True, "interaction": "SIT_ON"}))


def test_legacy_target_object_type_raises() -> None:
    with pytest.raises(ValueError, match="'target_object_type' removed"):
        _structure_manual(_with_step({"target_object_type": "bench"}))


def test_legacy_target_object_id_raises() -> None:
    with pytest.raises(ValueError, match="'target_object_id' removed"):
        _structure_manual(_with_step({"target_object_id": "b1"}))


def test_legacy_target_agent_raises() -> None:
    with pytest.raises(ValueError, match="'target_agent' removed"):
        _structure_manual(_with_step({"target_agent": 2}))


def test_legacy_service_tag_raises() -> None:
    with pytest.raises(ValueError, match="'service_tag' removed"):
        _structure_manual(_with_step({"interaction": "SERVICE", "service_tag": "water"}))


def test_follow_interaction_raises() -> None:
    with pytest.raises(ValueError, match="FOLLOW removed"):
        _structure_manual(_with_step({"interaction": "FOLLOW"}))


def test_offer_true_on_non_service_raises() -> None:
    with pytest.raises(ValueError, match="'offer: true' is not valid"):
        _structure_manual(_with_step({"interaction": "TALK_TO", "offer": True}))


def test_offer_true_on_service_without_target_raises() -> None:
    with pytest.raises(ValueError, match="'target: <tag:str>'"):
        _structure_manual(_with_step({"interaction": "SERVICE", "offer": True}))


def test_object_bound_without_target_raises() -> None:
    with pytest.raises(ValueError, match="requires 'target:"):
        _structure_manual(_with_step({"interaction": "SIT_ON"}))


def test_symmetric_with_target_raises() -> None:
    with pytest.raises(ValueError, match="takes no target"):
        _structure_manual(_with_step({"interaction": "TALK_TO", "target": "someone"}))


def test_contact_seeker_formation_spec_loads_without_offer() -> None:
    spec = {"type": "dyad", "anchor_kind": "centroid", "params": {"separation": 0.8}}
    scn = _structure_manual(_with_step({"interaction": "SHAKE_HAND", "formation_spec": spec}))
    loaded = scn.agent_types["walker"].sequences["default"].steps["step"].formation_spec
    assert loaded.type == "dyad"
    assert loaded.params["separation"] == 0.8


def test_non_contact_seeker_formation_spec_raises() -> None:
    spec = {"type": "dyad", "anchor_kind": "centroid", "params": {"separation": 0.8}}
    with pytest.raises(ValueError, match="provider-side configuration"):
        _structure_manual(_with_step({"interaction": "TALK_TO", "formation_spec": spec}))


def test_block_with_valid_target_agent_loads() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 99, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"interaction": "BLOCK", "target": 99, "duration": {"mean": 5.0, "std": 0.0, "clip_low": 0.1, "clip_high": 30.0}}}}},
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["default"].steps["pursue"].target == 99


def test_block_with_unknown_target_agent_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"interaction": "BLOCK", "target": 99}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="BLOCK requires"):
        _structure_manual(data)


def test_block_with_autonomous_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 99, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"interaction": "BLOCK", "target": 99, "autonomous": True}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="BLOCK and autonomous"):
        _structure_manual(data)


def test_provider_field_without_offer_raises() -> None:
    with pytest.raises(ValueError, match="provider-side"):
        _structure_manual(_with_step({"interaction": "SERVICE", "target": "water", "max_participants": 3}))


def test_go_to_step_with_both_target_pose_and_target_raises() -> None:
    data = _minimal(
        {
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"walk": {"kind": "go_to", "target_pose": {"x": 1.0, "y": 2.0}, "target": "bench"}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="not both"):
        _structure_manual(data)


def _step(fields: dict) -> StepDef | GoToStepDef | AttentionStepDef:
    return _structure_manual(_with_step(fields)).agent_types["walker"].sequences["default"].steps["step"]


def test_attention_step_structures() -> None:
    step = _step({"kind": "attention", "attention": {"point": {"at": "bench", "dwell": 0.5, "hold": "keep"}, "face": "auto"}, "duration": {"mean": 1.5}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention == AttentionDef(point=ChannelDef(at=("bench",), dwell=0.5, hold="keep"), face=None, required=True)
    assert step.duration is not None
    assert step.duration.mean == 1.5


def test_attention_step_dispatch_without_kind() -> None:
    step = _step({"attention": {"gaze": "bench"}, "duration": 2.0, "patience": 5.0, "on_failure": "skip"})
    assert isinstance(step, AttentionStepDef)
    assert step.on_failure == "skip"


def test_attention_channel_defaults_and_shorthand() -> None:
    step = _step({"attention": {"point": "bench"}})
    assert isinstance(step, AttentionStepDef)
    ch = step.attention.point
    assert ch == ChannelDef(at=("bench",))
    assert (ch.dwell, ch.advance, ch.hold, ch.at_z) == (1.0, "dwell", "release", None)
    assert step.attention.face is None
    assert step.attention.required is True
    assert step.attention.gaze is None


def test_attention_channel_list_and_options() -> None:
    step = _step({"attention": {"gaze": ["partner", 3, {"x": 0, "y": 0, "z": 0}], "point_l": {"at": ["a", "b"], "advance": "unreachable", "at_z": 1.4}}, "duration": 2.0})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.gaze == ChannelDef(at=("partner", 3, Pose3(0.0, 0.0, 0.0)))
    assert step.attention.point_l == ChannelDef(at=("a", "b"), advance="unreachable", at_z=1.4)
    assert step.attention.point is None
    assert step.attention.point_r is None
    assert step.attention.face_channel() == "point_l"


def test_attention_rider_not_required_by_default() -> None:
    step = _step({"duration": 2.0, "interaction": "TALK_TO", "attention": {"gaze": "partner"}})
    assert isinstance(step, StepDef)
    assert step.attention == AttentionDef(gaze=ChannelDef(at=("partner",)))
    assert step.attention.required is False
    step = _step({"duration": 2.0, "attention": {"gaze": "partner", "required": True}, "interaction": "TALK_TO"})
    assert isinstance(step, StepDef)
    assert step.attention.required is True


def test_attention_on_cancel_step() -> None:
    step = _step({"cancel": True, "attention": {"point_r": "partner"}})
    assert isinstance(step, StepDef)
    assert step.cancel is True
    assert step.attention is not None


def test_attention_on_go_to_step() -> None:
    step = _step({"kind": "go_to", "target_pose": {"x": 1.0, "y": 2.0}, "attention": {"point": {"at": ["partner", "ped_1"], "at_z": 1.4}, "face": True}})
    assert isinstance(step, GoToStepDef)
    assert step.attention == AttentionDef(point=ChannelDef(at=("partner", "ped_1"), at_z=1.4), face=True)


def test_sequence_attention_rider() -> None:
    data = _minimal(
        {
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"attention": {"gaze": {"at": "robot:bot", "hold": "keep"}, "face": "bench_1"}, "steps": {"wait": {"duration": 1.0}}}},
                }
            },
        }
    )
    seq = _structure_manual(data).agent_types["walker"].sequences["default"]
    assert seq.attention == AttentionDef(gaze=ChannelDef(at=(RobotRef("bot"),), hold="keep"), face="bench_1")


def test_go_to_step_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="unknown go_to step fields"):
        _step({"kind": "go_to", "target_pose": {"x": 1.0, "y": 2.0}, "gesture": "point"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("partner", "partner"),
        ("partners", "partners"),
        ("target", "target"),
        ("goal", "goal"),
        ("robot:bot", RobotRef("bot")),
        ("bench_1", "bench_1"),
        (7, 7),
        ({"x": 1.0, "y": 2.0, "z": 3.0}, Pose3(1.0, 2.0, 3.0)),
        ({"azimuth": 90, "elevation": 10}, RelativeRef(90.0, 10.0, 3.0)),
        ({"azimuth": -45, "elevation": 0, "distance": 1.5}, RelativeRef(-45.0, 0.0, 1.5)),
    ],
)
def test_attention_ref_kinds(raw: object, expected: object) -> None:
    step = _step({"attention": {"gaze": raw}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.gaze == ChannelDef(at=(expected,))


@pytest.mark.parametrize(
    ("face", "expected"),
    [(True, True), ("true", True), (False, False), ("false", False), ("auto", None), (None, None), ("bench_1", "bench_1"), ({"x": 1, "y": 2, "z": 3}, Pose3(1.0, 2.0, 3.0)), (4, 4)],
)
def test_attention_face_forms(face: object, expected: object) -> None:
    step = _step({"attention": {"gaze": "bench", "face": face}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.face == expected


@pytest.mark.parametrize(
    ("att", "match"),
    [
        ({}, "at least one channel"),
        ({"face": True}, "at least one channel"),
        ({"gesture": "point", "at": "bench"}, "unknown attention fields"),
        ({"gaze": "bench", "hand": "left"}, "unknown attention fields"),
        ({"gaze": "bench", "hold": "keep"}, "unknown attention fields"),
        ({"gaze": "bench", "dwell": 2}, "unknown attention fields"),
        ({"gaze": "bench", "at": "bench"}, "unknown attention fields"),
        ({"point": "a", "point_l": "b"}, "exactly one arm channel"),
        ({"point_r": "a", "point_l": "b"}, "exactly one arm channel"),
        ({"gaze": {"at": "bench", "hand": "left"}}, "unknown attention gaze fields"),
        ({"gaze": {"dwell": 1.0}}, "requires 'at'"),
        ({"gaze": []}, "must not be empty"),
        ({"gaze": {"at": []}}, "must not be empty"),
        ({"gaze": True}, "bool"),
        ({"gaze": ""}, "non-empty"),
        ({"gaze": "robot:"}, "robot:<name>"),
        ({"gaze": {"x": 1.0, "y": 2.0}}, "x, y, z"),
        ({"gaze": {"azimuth": 1.0}}, "azimuth"),
        ({"gaze": {"azimuth": "a", "elevation": 0}}, "number"),
        ({"gaze": {"azimuth": 0, "elevation": 0, "distance": 0}}, "distance"),
        ({"gaze": 1.5}, "ref must be"),
        ({"gaze": "bench", "face": 1.5}, "face"),
        ({"gaze": "bench", "face": ""}, "face"),
        ({"gaze": "bench", "face": {"azimuth": 0, "elevation": 0}}, "never drives face"),
        ({"gaze": {"at": "bench", "hold": "forever"}}, "hold"),
        ({"gaze": {"at": "bench", "advance": "never"}}, "advance"),
        ({"gaze": {"at": "bench", "dwell": 0}}, "dwell"),
        ({"gaze": {"at": "bench", "dwell": "x"}}, "dwell"),
        ({"gaze": {"at": "bench", "at_z": "high"}}, "at_z"),
        ({"gaze": {"at": {"x": 1.0, "y": 2.0, "z": 3.0}, "at_z": 0.5}}, "at_z"),
        ({"gaze": {"at": {"azimuth": 0, "elevation": 0}, "at_z": 0.5}}, "at_z"),
        ({"gaze": {"at": ["bench", {"azimuth": 0, "elevation": 0}], "at_z": 0.5}}, "at_z"),
        ({"gaze": {"azimuth": 0, "elevation": 0}, "face": True}, "face: true"),
        ({"gaze": "bench", "point": [{"azimuth": 0, "elevation": 0}, "bench"], "face": "true"}, "face: true"),
        ({"gaze": "bench", "required": "yes"}, "required"),
        ({"gaze": "bench", "required": False}, "always required"),
        ({"gaze": {"at": ["a", "b"], "advance": "unreachable"}}, "needs 'duration' or 'patience'"),
        ({"clip": "wave"}, "only a clip"),
        ({"clip": ""}, "non-empty 'name'"),
        ({"clip": {"name": "wave", "when": "sometimes"}}, "when"),
        ({"clip": {"name": "wave", "loop": True}}, "unknown attention clip fields"),
        ({"clip": 3}, "clip name"),
        ({"posture": "prone"}, "only a clip or posture"),
        ({"gaze": "bench", "posture": "flying"}, "posture"),
    ],
)
def test_attention_validation(att: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _step({"attention": att})


def test_attention_face_true_allows_relative_later_entries() -> None:
    step = _step({"attention": {"point": ["bench", {"azimuth": 0, "elevation": 0}], "face": True}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.face is True


def test_attention_clip_forms() -> None:
    step = _step({"attention": {"clip": "wave_high"}, "duration": 2.0})
    assert isinstance(step, AttentionStepDef)
    assert step.attention == AttentionDef(clip=ClipDef(name="wave_high"), required=True)
    step = _step({"interaction": "WAVE_AT", "attention": {"clip": {"name": "wave", "when": "bound"}, "gaze": "partner"}})
    assert isinstance(step, StepDef)
    assert step.attention is not None and step.attention.clip == ClipDef(name="wave", when="bound")
    assert step.attention.gaze == ChannelDef(at=("partner",))


def test_attention_posture_forms() -> None:
    step = _step({"attention": {"clip": "collapse_to_ground", "posture": "prone"}, "duration": 3.0})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.posture == "prone" and step.attention.clip == ClipDef(name="collapse_to_ground")


def test_attention_cycling_bare_step_with_duration_or_patience() -> None:
    assert isinstance(_step({"attention": {"gaze": {"at": ["a", "b"], "advance": "unreachable"}}, "duration": 2.0}), AttentionStepDef)
    assert isinstance(_step({"attention": {"gaze": {"at": ["a", "b"], "advance": "unreachable"}}, "patience": 2.0}), AttentionStepDef)


def test_attention_rider_accepts_cycling_and_required_false() -> None:
    step = _step({"duration": 2.0, "attention": {"gaze": {"at": ["a", "b"], "advance": "unreachable"}, "required": False}, "interaction": "TALK_TO"})
    assert isinstance(step, StepDef)


def test_attention_not_a_mapping_raises() -> None:
    with pytest.raises(ValueError, match="mapping"):
        _step({"attention": "point"})


def test_attention_at_z_numeric_with_entity_ref() -> None:
    assert _step({"attention": {"gaze": {"at": "bench", "at_z": 1}}}).attention.gaze.at_z == 1.0
    assert _step({"attention": {"gaze": {"at": ["partner", 3], "at_z": 1.5}}}).attention.gaze.at_z == 1.5


def test_attention_step_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="unknown attention step fields"):
        _step({"kind": "attention", "attention": {"gaze": "bench"}, "target": "bench"})


def test_attention_with_step_only_fields_names_them() -> None:
    with pytest.raises(ValueError, match=r"step has attention plus interaction-only fields \['offer', 'until'\], add kind or interaction"):
        _step({"attention": {"gaze": "bench"}, "offer": True, "until": "x"})


def test_attention_on_autonomous_step_raises() -> None:
    with pytest.raises(ValueError, match="not supported on 'autonomous: true' steps"):
        _step({"autonomous": True, "attention": {"gaze": "bench"}})


def test_attention_kind_requires_block() -> None:
    with pytest.raises(ValueError, match="attention step requires"):
        _step({"kind": "attention", "duration": 1.0})


def test_attention_in_actions_raises() -> None:
    data = _minimal(
        {
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "actions": {"show": {"attention": {"gaze": "bench"}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="not supported in the autonomous 'actions' library"):
        _structure_manual(data)


def test_unknown_step_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown step kind"):
        _step({"kind": "gesture", "gesture": "point", "at": "bench"})


def test_sequence_tune_structures_every_value_form() -> None:
    tune = {"desired_velocity": {"mul": 1.5}, "max_velocity": 2.6, "reaction_time": {"add": 0.2}, "perception": {"vision_range": {"mul": 0.5, "add": 1}}, "local_planner_params": {"repulsion_strength": {"poly": [0, 0, 2]}}}
    seq = converter.structure({"steps": {}, "tune": tune}, SequenceDef)
    assert seq.tune == TuneDef({"desired_velocity": (0.0, 1.5), "max_velocity": (2.6,), "reaction_time": (0.2, 1.0), "perception.vision_range": (1.0, 0.5), "local_planner_params.repulsion_strength": (0.0, 0.0, 2.0)})


def test_step_tune_structures_on_every_step_kind() -> None:
    tune = {"max_velocity": 2.0}
    assert _step({"kind": "go_to", "target_pose": {"x": 1.0, "y": 0.0}, "tune": tune}).tune == TuneDef({"max_velocity": (2.0,)})
    assert _step({"interaction": "TALK_TO", "tune": tune}).tune == TuneDef({"max_velocity": (2.0,)})
    assert _step({"attention": {"gaze": "robot:r"}, "duration": {"mean": 1.0}, "tune": tune}).tune == TuneDef({"max_velocity": (2.0,)})


def test_agent_type_tunes_need_their_needs_declared() -> None:
    raw = {"name": "t", "tunes": [{"when": {"alarm": {"above": 50.0}}, "tune": {"desired_velocity": {"mul": 1.5}}}]}
    with pytest.raises(ValueError, match="does not declare: \\['alarm'\\]"):
        converter.structure(raw, AgentType)


def test_agent_type_tunes_structure_with_their_conditions() -> None:
    raw = {"name": "t", "needs": {"alarm": {"initial": 0.0, "decay_rate": 0.0}}, "tunes": [{"when": {"alarm": {"above": 50.0}}, "tune": {"desired_velocity": {"mul": 1.5}}}]}
    t = converter.structure(raw, AgentType)
    assert len(t.tunes) == 1 and t.tunes[0].when["alarm"].above == 50.0 and t.tunes[0].tune == TuneDef({"desired_velocity": (0.0, 1.5)})


@pytest.mark.parametrize(
    ("tune", "match"),
    [
        ({"speed_scale": 2.0}, "unknown tune field"),
        ({"agent_radius": 0.1}, "unknown tune field"),
        ({"perception": {"vision_rang": 2.0}}, "unknown tune perception field"),
        ({"max_velocity": {"mul": 2.0, "poly": [1.0]}}, "poly excludes"),
        ({"max_velocity": {"scale": 2.0}}, "expected a number, mul"),
        ({"max_velocity": {"poly": []}}, "at least one coefficient"),
        ({"max_velocity": {"poly": "12"}}, "at least one coefficient"),
        ({"max_velocity": {"mul": True}}, "expected a number"),
        ({"max_velocity": {"mul": "2"}}, "expected a number"),
        ({"max_velocity": {}}, "expected a number, mul"),
        ({"max_velocity": "fast"}, "expected a number"),
    ],
)
def test_sequence_tune_typo_raises(tune: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        converter.structure({"steps": {}, "tune": tune}, SequenceDef)
