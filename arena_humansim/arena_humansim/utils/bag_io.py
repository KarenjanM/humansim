import itertools
import struct
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

warnings.filterwarnings("ignore", category=RuntimeWarning)

AGENT_STATES_TYPE = "arena_humansim_msgs/msg/AgentStates"
AGENT_FRAME_TYPE = "arena_humansim_msgs/msg/AgentFrame"
AGENT_META_TYPE = "arena_humansim_msgs/msg/AgentMeta"


def _frame_size(rawdata: bytes | memoryview) -> int:
    """Agent count of a serialized AgentFrame, read from the agent_id length after the header."""
    order = "<" if rawdata[1] else ">"
    (frame_id_len,) = struct.unpack_from(f"{order}I", rawdata, 12)
    pos = 16 + frame_id_len
    (count,) = struct.unpack_from(f"{order}I", rawdata, pos + -pos % 4)
    return count


def policy_names(meta_t: np.ndarray, meta_policies: list[list[str]], frame_t: np.ndarray, counts: np.ndarray, policy_idx: np.ndarray) -> np.ndarray:
    """Planner name per agent row from the latest agent_meta at or before its frame, empty when unknown."""
    pidx = np.asarray(policy_idx, dtype=np.int64)
    if not meta_policies:
        return np.full(len(pidx), "", dtype=object)
    order = np.argsort(meta_t, kind="stable")
    sorted_t = np.asarray(meta_t)[order]
    ordered = [meta_policies[i] for i in order]
    n_pol = np.array([len(p) for p in ordered], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_pol)[:-1]])
    table = np.array([*itertools.chain.from_iterable(ordered), ""], dtype=object)
    k = np.repeat(np.searchsorted(sorted_t, frame_t, side="right") - 1, counts)
    kc = np.maximum(k, 0)
    valid = (k >= 0) & (pidx >= 0) & (pidx < n_pol[kc])
    return table[np.where(valid, offsets[kc] + pidx, len(table) - 1)]


def _frame_rows(frames: list[tuple[int, Any]], metas: list[tuple[int, list[str]]]) -> pd.DataFrame:
    msgs = [m for _, m in frames]
    counts = np.array([len(m.agent_id) for m in msgs], dtype=np.int64)
    frame_t = np.array([t for t, _ in frames], dtype=np.int64)
    meta_t = np.array([t for t, _ in metas], dtype=np.int64)
    pidx = np.concatenate([m.policy_idx for m in msgs]).astype(np.int64)
    return pd.DataFrame(
        {
            "time": np.repeat(frame_t * 1e-9, counts),
            "agent_id": np.concatenate([m.agent_id for m in msgs]).astype(np.int64),
            "x": np.concatenate([m.x for m in msgs]).astype(np.float64),
            "y": np.concatenate([m.y for m in msgs]).astype(np.float64),
            "vx": np.concatenate([m.vx for m in msgs]).astype(np.float64),
            "vy": np.concatenate([m.vy for m in msgs]).astype(np.float64),
            "radius": np.concatenate([m.radius for m in msgs]).astype(np.float64),
            "planner": policy_names(meta_t, [p for _, p in metas], frame_t, counts, pidx),
        }
    )


def extract_agent_states(bag_path: Path) -> pd.DataFrame:
    extracted_data = []
    frames: list[tuple[int, Any]] = []
    metas: list[tuple[int, list[str]]] = []
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if ("agent_states" in c.topic and c.msgtype in (AGENT_STATES_TYPE, AGENT_FRAME_TYPE)) or c.msgtype == AGENT_META_TYPE]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if connection.msgtype == AGENT_FRAME_TYPE:
                if _frame_size(rawdata):
                    frames.append((timestamp, reader.deserialize(rawdata, connection.msgtype)))
            elif connection.msgtype == AGENT_META_TYPE:
                meta: Any = reader.deserialize(rawdata, connection.msgtype)
                metas.append((timestamp, list(meta.policies)))
            else:
                msg: Any = reader.deserialize(rawdata, connection.msgtype)
                time_sec = timestamp * 1e-9
                for agent in msg.agents:
                    extracted_data.append(
                        {
                            "time": time_sec,
                            "agent_id": agent.agent_id,
                            "x": agent.pose.x,
                            "y": agent.pose.y,
                            "vx": agent.velocity.x,
                            "vy": agent.velocity.y,
                            "radius": agent.radius,
                            "planner": agent.policy,
                        }
                    )
    df = pd.DataFrame(extracted_data)
    if frames:
        flat = _frame_rows(frames, metas)
        df = flat if df.empty else pd.concat([df, flat], ignore_index=True)
    return df
