from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from arena_humansim.utils.evaluation.bag_cache import load_multi
from arena_humansim.utils.evaluation.buckets import SCENARIO_BUCKET
from arena_humansim.utils.evaluation.robots import (
    GOAL_REACH_THRESHOLD_M,
    _load_snapshot,
    parse_robots_trial_dir,
    robot_ids_in,
)

_FREEZE_SPEED_THRESHOLD = 0.05
_FREEZE_WINDOW_S = 5.0
_OVERSHOOT_NEAR_M = 0.5
_OVERSHOOT_END_M = 1.0
_STATIC_NEIGHBOR_FAR_M = 2.0
_STATIC_NEIGHBOR_SPEED = 0.05

CAUSES = (
    "success",
    "collision_with_reactor",
    "collision_static",
    "freezing_timeout",
    "goal_overshoot",
    "timeout_no_progress",
    "other",
)


def _classify_one_robot(
    trial_df: pd.DataFrame,
    robot_id: int,
    goal: tuple[float, float] | None,
) -> str:
    robot = trial_df[trial_df["agent_id"] == robot_id].sort_values("time")  # pyright: ignore[reportCallIssue]
    if robot.empty:
        return "other"

    times = robot["time"].to_numpy()
    xy = robot[["x", "y"]].to_numpy()
    speeds = np.linalg.norm(robot[["vx", "vy"]].to_numpy(), axis=1)
    radii = robot["radius"].to_numpy()
    radius = float(radii[0]) if len(radii) > 0 else 0.3
    t_end = float(times[-1])

    reached_goal = False
    overshot = False
    if goal is not None:
        gx, gy = goal
        d_to_goal = np.hypot(xy[:, 0] - gx, xy[:, 1] - gy)
        was_near = d_to_goal <= _OVERSHOOT_NEAR_M
        end_far = d_to_goal[-1] >= _OVERSHOOT_END_M
        reached_goal = bool(d_to_goal[-1] <= GOAL_REACH_THRESHOLD_M)
        overshot = bool(was_near.any() and end_far and not reached_goal)

    if reached_goal:
        return "success"

    collision_t: float | None = None
    collision_static = False
    for _t, frame in trial_df.groupby("time"):
        rb = frame[frame["agent_id"] == robot_id]
        if rb.empty:
            continue
        rb_x = float(rb["x"].iloc[0])  # pyright: ignore[reportAttributeAccessIssue]
        rb_y = float(rb["y"].iloc[0])  # pyright: ignore[reportAttributeAccessIssue]
        others = frame[frame["agent_id"] != robot_id]
        if others.empty:
            continue
        ox = others["x"].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
        oy = others["y"].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
        orad = others["radius"].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
        ovx = others["vx"].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
        ovy = others["vy"].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
        ospeed = np.hypot(ovx, ovy)
        d = np.hypot(ox - rb_x, oy - rb_y)
        hit = d < (radius + orad)
        if hit.any():
            collision_t = float(_t)  # pyright: ignore[reportArgumentType]
            hit_idx = int(np.argmax(hit))
            min_other_dist = float(d.min())
            collision_static = ospeed[hit_idx] < _STATIC_NEIGHBOR_SPEED or min_other_dist > _STATIC_NEIGHBOR_FAR_M
            break

    if collision_t is not None:
        return "collision_static" if collision_static else "collision_with_reactor"

    if overshot:
        return "goal_overshoot"

    freeze_window_mask = times >= (t_end - _FREEZE_WINDOW_S)
    end_speed = float(speeds[freeze_window_mask].mean()) if freeze_window_mask.any() else 0.0
    if end_speed < _FREEZE_SPEED_THRESHOLD:
        return "freezing_timeout"

    return "timeout_no_progress"


def classify_trial(
    trial_df: pd.DataFrame,
    goals: dict[int, tuple[float, float] | None] | None = None,
) -> dict[int, str]:
    """Per-robot failure-cause classification keyed by robot_id."""
    goals = goals or {}
    return {rid: _classify_one_robot(trial_df, rid, goals.get(rid)) for rid in robot_ids_in(trial_df)}


def classify_run(recordings_dirs: Path | list[Path]) -> pd.DataFrame:
    if isinstance(recordings_dirs, Path):
        recordings_dirs = [recordings_dirs]
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()
    nested = load_multi(recordings_dirs)
    for recordings_dir in recordings_dirs:
        for trial_name, df in nested.get(recordings_dir.name, {}).items():
            parsed = parse_robots_trial_dir(trial_name)
            if parsed is None:
                continue
            scenario, planner, robot_policy, seed = parsed
            key = (scenario, planner, robot_policy, seed)
            if key in seen:
                continue
            if df.empty:
                continue
            seen.add(key)
            canonical, goals = _load_snapshot(recordings_dir / trial_name)
            bucket_key = canonical or scenario
            for robot_id, cause in classify_trial(df, goals).items():
                rows.append(
                    {
                        "scenario": scenario,
                        "bucket": SCENARIO_BUCKET.get(bucket_key, "unknown"),
                        "ped_planner": planner,
                        "robot_policy": robot_policy,
                        "seed": seed,
                        "robot_id": robot_id,
                        "source_dir": recordings_dir.name,
                        "cause": cause,
                    }
                )
    return pd.DataFrame(rows)


def cause_distribution(per_trial_df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    if per_trial_df.empty:
        return per_trial_df
    counts = per_trial_df.groupby([*group_by, "cause"]).size().rename("n").reset_index()  # pyright: ignore[reportArgumentType, reportCallIssue]
    totals = per_trial_df.groupby(group_by).size().rename("total").reset_index()  # pyright: ignore[reportArgumentType, reportCallIssue]
    out = counts.merge(totals, on=group_by, how="left")
    out["fraction"] = out["n"] / out["total"]
    return out


def run_failure_analysis(
    recordings_dirs: Path | list[Path],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(recordings_dirs, Path):
        recordings_dirs = [recordings_dirs]
    print(f"Classifying failures from {len(recordings_dirs)} dir(s): {[d.name for d in recordings_dirs]}")
    per_trial = classify_run(recordings_dirs)
    if per_trial.empty:
        raise RuntimeError(f"no trials under {[str(d) for d in recordings_dirs]}")
    per_trial_path = out_dir / "failures_per_trial.csv"
    per_trial.to_csv(per_trial_path, index=False)
    print(f"  wrote {per_trial_path}")

    by_policy = cause_distribution(per_trial, ["robot_policy"])
    by_policy_path = out_dir / "failures_by_policy.csv"
    by_policy.to_csv(by_policy_path, index=False)
    print(f"  wrote {by_policy_path}")

    by_pb = cause_distribution(per_trial, ["robot_policy", "bucket"])
    by_pb_path = out_dir / "failures_by_policy_bucket.csv"
    by_pb.to_csv(by_pb_path, index=False)
    print(f"  wrote {by_pb_path}")

    print()
    print("=== Failure cause fraction by robot_policy x bucket ===")
    pivot = by_pb.pivot_table(index=["robot_policy", "bucket"], columns="cause", values="fraction", fill_value=0.0)
    print(pivot.to_string(float_format="{:.3f}".format))

    return {"per_trial": per_trial, "by_policy": by_policy, "by_policy_bucket": by_pb}
