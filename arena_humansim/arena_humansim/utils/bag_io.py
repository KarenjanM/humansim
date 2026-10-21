import itertools
import struct
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

warnings.filterwarnings("ignore", category=RuntimeWarning)

AGENT_STATES_TYPE = "arena_humansim_msgs/msg/AgentStates"
AGENT_FRAME_TYPE = "arena_humansim_msgs/msg/AgentFrame"
AGENT_META_TYPE = "arena_humansim_msgs/msg/AgentMeta"


class AgentSources(NamedTuple):
    agent_id: list[int]
    source_name: list[str]
    source_seq: list[int]


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


def source_attribution(meta_t: np.ndarray, meta_sources: list[AgentSources], frame_t: np.ndarray, counts: np.ndarray, agent_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source name and release index per agent row from the latest agent_meta at or before its frame, else from the first later one naming the agent, empty and 0 when unknown."""
    aid = np.asarray(agent_id, dtype=np.int64)
    if not meta_sources:
        return np.full(len(aid), "", dtype=object), np.zeros(len(aid), dtype=np.int64)
    order = np.argsort(meta_t, kind="stable")
    ordered = [meta_sources[i] for i in order]
    sorted_t = np.asarray(meta_t, dtype=np.int64)[order]
    sizes = [len(s.agent_id) for s in ordered]
    table = pd.DataFrame(
        {
            "k": np.repeat(np.arange(len(ordered)), sizes),
            "t": np.repeat(sorted_t, sizes),
            "agent_id": np.array([*itertools.chain.from_iterable(s.agent_id for s in ordered)], dtype=np.int64),
            "source_name": np.array([*itertools.chain.from_iterable(s.source_name for s in ordered)], dtype=object),
            "source_seq": np.array([*itertools.chain.from_iterable(s.source_seq for s in ordered)], dtype=np.int64),
        }
    ).drop_duplicates(["k", "agent_id"])
    row_t = np.repeat(np.asarray(frame_t, dtype=np.int64), counts)
    k = np.searchsorted(sorted_t, row_t, side="right") - 1
    rows = pd.DataFrame({"k": k, "agent_id": aid}).merge(table.drop(columns="t"), on=["k", "agent_id"], how="left")
    unnamed = np.flatnonzero(rows["source_name"].isna().to_numpy())
    if len(unnamed):
        probe = pd.DataFrame({"t": row_t[unnamed], "agent_id": aid[unnamed], "row": unnamed}).sort_values("t", kind="stable")
        later = pd.merge_asof(probe, table.drop(columns="k").sort_values("t", kind="stable"), on="t", by="agent_id", direction="forward").dropna(subset=["source_name"])
        rows.loc[later["row"].to_numpy(), ["source_name", "source_seq"]] = later[["source_name", "source_seq"]].to_numpy()
    return rows["source_name"].fillna("").to_numpy(dtype=object), rows["source_seq"].fillna(0).to_numpy(dtype=np.int64)


def _frame_rows(frames: list[tuple[int, Any]], metas: list[tuple[int, list[str], AgentSources]]) -> pd.DataFrame:
    msgs = [m for _, m in frames]
    counts = np.array([len(m.agent_id) for m in msgs], dtype=np.int64)
    frame_t = np.array([t for t, _ in frames], dtype=np.int64)
    meta_t = np.array([t for t, _, _ in metas], dtype=np.int64)
    pidx = np.concatenate([m.policy_idx for m in msgs]).astype(np.int64)
    agent_id = np.concatenate([m.agent_id for m in msgs]).astype(np.int64)
    source_name, source_seq = source_attribution(meta_t, [s for _, _, s in metas], frame_t, counts, agent_id)
    return pd.DataFrame(
        {
            "time": np.repeat(frame_t * 1e-9, counts),
            "agent_id": agent_id,
            "x": np.concatenate([m.x for m in msgs]).astype(np.float64),
            "y": np.concatenate([m.y for m in msgs]).astype(np.float64),
            "vx": np.concatenate([m.vx for m in msgs]).astype(np.float64),
            "vy": np.concatenate([m.vy for m in msgs]).astype(np.float64),
            "radius": np.concatenate([m.radius for m in msgs]).astype(np.float64),
            "planner": policy_names(meta_t, [p for _, p, _ in metas], frame_t, counts, pidx),
            "source_name": source_name,
            "source_seq": source_seq,
        }
    )


def _agent_sources(agent_id: Sequence[int], source_name: Sequence[str], source_seq: Sequence[int]) -> AgentSources:
    ids = [int(a) for a in agent_id]
    if len(source_name) == len(ids) and len(source_seq) == len(ids):
        return AgentSources(ids, [str(n) for n in source_name], [int(q) for q in source_seq])
    return AgentSources(ids, [""] * len(ids), [0] * len(ids))


def extract_agent_states(bag_path: Path) -> pd.DataFrame:
    extracted_data = []
    frames: list[tuple[int, Any]] = []
    metas: list[tuple[int, list[str], AgentSources]] = []
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if ("agent_states" in c.topic and c.msgtype in (AGENT_STATES_TYPE, AGENT_FRAME_TYPE)) or c.msgtype == AGENT_META_TYPE]
        attributed = any(c.msgtype == AGENT_META_TYPE for c in connections) and {"source_name", "source_seq"} <= {name for name, _ in reader.typestore.fielddefs[AGENT_META_TYPE][1]}
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if connection.msgtype == AGENT_FRAME_TYPE:
                if _frame_size(rawdata):
                    frames.append((timestamp, reader.deserialize(rawdata, connection.msgtype)))
            elif connection.msgtype == AGENT_META_TYPE:
                meta: Any = reader.deserialize(rawdata, connection.msgtype)
                metas.append((timestamp, list(meta.policies), _agent_sources(meta.agent_id, meta.source_name, meta.source_seq) if attributed else _agent_sources(meta.agent_id, (), ())))
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
                            "source_name": "",
                            "source_seq": 0,
                        }
                    )
    df = pd.DataFrame(extracted_data)
    if frames:
        flat = _frame_rows(frames, metas)
        df = flat if df.empty else pd.concat([df, flat], ignore_index=True)
    return df
