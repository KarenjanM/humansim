from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import BehaviorTreeMovement, GestureIntent
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentGestures as AgentGesturesMsg
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from rclpy.executors import SingleThreadedExecutor

from ._capture import LATCHED, STREAM, settle, spin_until, subscribe


def _spawn(mgr: AgentManager, name: str, kind: int, x: float, handedness: str = "") -> SpawnAgents.Response:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.name = name
    msg.kind = kind
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.3
    msg.agent_type = "adult"
    msg.handedness = handedness
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return resp


def test_meta_and_gestures_carry_name_handedness_and_gestures(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(name="gesture_pub", simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    mgr = manager_factory(scenario, node_name="test_gesture_pub")
    metas = subscribe(mgr, AgentMetaMsg, "agent_meta", LATCHED)
    gestures = subscribe(mgr, AgentGesturesMsg, "agent_gestures", LATCHED)
    frames = subscribe(mgr, AgentFrameMsg, "agent_states", STREAM)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)

    ped = _spawn(mgr, "ped_1", AgentStateMsg.KIND_HUMAN, 0.0).spawned_ids[0]
    bot = _spawn(mgr, "bot", AgentStateMsg.KIND_ROBOT, 2.0).spawned_ids[0]
    lefty = _spawn(mgr, "", AgentStateMsg.KIND_HUMAN, 4.0, handedness="l").spawned_ids[0]
    assert mgr._agent_name_to_id["ped_1"] == ped
    assert mgr._agent_name_to_id["bot"] == bot
    assert mgr._agents[ped].params.handedness in ("l", "r")
    assert mgr._agents[lefty].params.handedness == "l"

    mv = BehaviorTreeMovement(gestures=(GestureIntent("arm", 1.0, 2.0, 3.0, hand="l"), GestureIntent("head", 4.0, 5.0, 6.0), GestureIntent("receive_r", 2.0, 0.0, 1.2, grip=0.4)))
    mgr._agents[ped].movement = mv

    mgr.tick()
    spin_until(executor, lambda: len(frames) == 1 and len(metas) == 1 and len(gestures) == 1)
    assert metas[0].header.stamp == frames[0].header.stamp
    assert gestures[0].header.stamp == frames[0].header.stamp
    meta = metas[-1]
    assert list(meta.agent_id) == [ped, bot, lefty]
    names = dict(zip(meta.agent_id, meta.name, strict=True))
    hands = dict(zip(meta.agent_id, meta.handedness, strict=True))
    assert names[ped] == "ped_1"
    assert hands[ped] == mgr._agents[ped].params.handedness
    assert names[bot] == "bot"
    assert names[lefty] == ""
    assert hands[lefty] == "l"
    assert list(gestures[-1].agent_id) == [ped, ped, ped]
    assert [g.slot for g in gestures[-1].gestures] == ["arm", "head", "receive_r"]
    arm, head, receive = gestures[-1].gestures
    assert (arm.at.x, arm.at.y, arm.at.z) == (1.0, 2.0, 3.0)
    assert arm.hand == "l"
    assert (head.at.x, head.at.y, head.at.z) == (4.0, 5.0, 6.0)
    assert head.hand == "" and head.clip == ""
    assert arm.grip == head.grip == 0.0
    assert receive.grip == pytest.approx(0.4)

    mv.gestures = ()
    while mgr._tick_count % mgr._bt_tick_interval != 0:
        mgr.tick()
    mgr.tick()
    spin_until(executor, lambda: len(gestures) == 2)
    assert gestures[-1].header.stamp == frames[-1].header.stamp
    assert list(gestures[-1].agent_id) == []
    assert list(gestures[-1].gestures) == []
    executor.remove_node(mgr)

    assert mgr._lookup_agent_name("bot") == bot
    assert mgr._lookup_agent_name("bot", AgentStateMsg.KIND_ROBOT) == bot
    assert mgr._lookup_agent_name("ped_1", AgentStateMsg.KIND_ROBOT) is None
    assert mgr._lookup_agent_name("ped_1", AgentStateMsg.KIND_HUMAN) == ped

    mgr._remove_agent(bot)
    assert mgr._agent_name_to_id == {"ped_1": ped}


def test_spawn_rejects_reserved_names_and_bad_handedness(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(name="gesture_reserved", simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    mgr = manager_factory(scenario, node_name="test_gesture_reserved")

    resp = _spawn(mgr, "partner", AgentStateMsg.KIND_HUMAN, 0.0)
    assert resp.success is False
    assert "reserved" in resp.message
    assert mgr._agents == {}

    resp = _spawn(mgr, "ped_1", AgentStateMsg.KIND_HUMAN, 0.0, handedness="left")
    assert resp.success is False
    assert "handedness" in resp.message
    assert mgr._agents == {}
