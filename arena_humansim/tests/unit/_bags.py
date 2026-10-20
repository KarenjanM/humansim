"""Write one agent trajectory as a real mcap bag in the nested or the flat agent_states layout."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentGestures as AgentGesturesMsg
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg
from geometry_msgs.msg import Vector3
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata

NESTED = "nested"
FLAT = "flat"
LAYOUTS = (NESTED, FLAT)


class Agent(NamedTuple):
    agent_id: int
    x: float
    y: float
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.35
    policy: str = "sfm"
    kind: int = 0
    name: str = ""
    handedness: str = ""
    source_name: str = ""
    source_seq: int = 0


Frames = list[tuple[int, list[Agent]]]


def _stamp(msg: AgentFrameMsg | AgentMetaMsg | AgentGesturesMsg | AgentStatesMsg, t_ns: int) -> None:
    msg.header.stamp.sec = t_ns // 1_000_000_000
    msg.header.stamp.nanosec = t_ns % 1_000_000_000


def nested_msg(t_ns: int, agents: list[Agent]) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    _stamp(msg, t_ns)
    for a in agents:
        m = AgentStateMsg()
        m.agent_id = a.agent_id
        m.pose = Pose2DMsg(x=a.x, y=a.y, theta=a.theta)
        m.velocity = Vector3(x=a.vx, y=a.vy, z=0.0)
        m.radius = a.radius
        m.policy = a.policy
        m.kind = a.kind
        m.name = a.name
        m.handedness = a.handedness
        msg.agents.append(m)
    return msg


def frame_msg(t_ns: int, agents: list[Agent], policies: list[str]) -> AgentFrameMsg:
    msg = AgentFrameMsg()
    _stamp(msg, t_ns)
    msg.agent_id = [a.agent_id for a in agents]
    msg.x = [a.x for a in agents]
    msg.y = [a.y for a in agents]
    msg.theta = [a.theta for a in agents]
    msg.vx = [a.vx for a in agents]
    msg.vy = [a.vy for a in agents]
    msg.desired_velocity = [0.0 for _ in agents]
    msg.radius = [a.radius for a in agents]
    msg.kind = [a.kind for a in agents]
    msg.animation_state = [0 for _ in agents]
    msg.policy_idx = [policies.index(a.policy) if a.policy else -1 for a in agents]
    msg.gait_phase = [0.0 for _ in agents]
    msg.gait_cadence = [0.0 for _ in agents]
    return msg


def meta_msg(t_ns: int, agents: list[Agent], policies: list[str]) -> AgentMetaMsg:
    msg = AgentMetaMsg()
    _stamp(msg, t_ns)
    msg.policies = list(policies)
    msg.agent_id = [a.agent_id for a in agents]
    msg.name = [a.name for a in agents]
    msg.handedness = [a.handedness for a in agents]
    msg.source_name = [a.source_name for a in agents]
    msg.source_seq = [a.source_seq for a in agents]
    return msg


def write_bag(bag_dir: Path, layout: str, frames: Frames, geometry: WorldGeometryMsg | None = None, with_meta: bool = True, late_meta: bool = False) -> None:
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(bag_dir), storage_id="mcap"), ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    states_type = "arena_humansim_msgs/msg/AgentStates" if layout == NESTED else "arena_humansim_msgs/msg/AgentFrame"
    writer.create_topic(TopicMetadata(id=0, name="/agent_states", type=states_type, serialization_format="cdr"))
    if geometry is not None:
        writer.create_topic(TopicMetadata(id=1, name="/arena_humansim/world_geometry", type="arena_humansim_msgs/msg/WorldGeometry", serialization_format="cdr"))
        writer.write("/arena_humansim/world_geometry", serialize_message(geometry), 0)
    if layout == FLAT:
        writer.create_topic(TopicMetadata(id=2, name="/agent_meta", type="arena_humansim_msgs/msg/AgentMeta", serialization_format="cdr"))
        writer.create_topic(TopicMetadata(id=3, name="/agent_gestures", type="arena_humansim_msgs/msg/AgentGestures", serialization_format="cdr"))
        gestures = AgentGesturesMsg()
        writer.write("/agent_gestures", serialize_message(gestures), 0)
    policies: list[str] = []
    last_meta: tuple | None = None
    pending: list[tuple[int, AgentMetaMsg]] = []
    for t_ns, agents in frames:
        if layout == NESTED:
            writer.write("/agent_states", serialize_message(nested_msg(t_ns, agents)), t_ns)
            continue
        policies.extend(p for p in dict.fromkeys(a.policy for a in agents) if p and p not in policies)
        key = (tuple(policies), tuple((a.agent_id, a.name, a.handedness, a.source_name, a.source_seq) for a in agents))
        if with_meta and key != last_meta:
            last_meta = key
            pending.append((t_ns, meta_msg(t_ns, agents, policies)))
        if not late_meta:
            for stamp, meta in pending:
                writer.write("/agent_meta", serialize_message(meta), stamp)
            pending.clear()
        writer.write("/agent_states", serialize_message(frame_msg(t_ns, agents, policies)), t_ns)
    for stamp, meta in pending:
        writer.write("/agent_meta", serialize_message(meta), stamp)
    del writer
