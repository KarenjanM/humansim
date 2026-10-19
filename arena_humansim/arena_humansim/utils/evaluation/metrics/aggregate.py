import numpy as np
import pandas as pd


def calculate_kinematic_metrics(agent_df: pd.DataFrame) -> pd.Series:
    dt = 0.05
    if len(agent_df) < 2:
        return pd.Series({"mean_jerk": np.nan, "mean_curvature": np.nan})
    v = agent_df[["vx", "vy"]].to_numpy()
    v_next = np.roll(v, -1, axis=0)
    v_prev = np.roll(v, 1, axis=0)
    jerk_vec = (v_next - 2 * v + v_prev) / (dt**2)
    jerk_vec = jerk_vec[1:-1]
    jerk_mag = np.linalg.norm(jerk_vec, axis=1)
    mean_jerk = np.nanmean(jerk_mag) if jerk_mag.size else np.nan
    vx = agent_df["vx"].to_numpy()
    vy = agent_df["vy"].to_numpy()
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    numerator = np.abs(vx * ay - vy * ax)
    denominator = (vx**2 + vy**2) ** 1.5
    curvature = np.where(denominator > 1e-5, numerator / denominator, 0)
    mean_curvature = np.nanmean(curvature)
    return pd.Series({"mean_jerk": mean_jerk, "mean_curvature": mean_curvature})


def calculate_run_collisions(run_df: pd.DataFrame) -> int:
    collision_pairs = set()
    for _, frame in run_df.groupby("time"):
        if len(frame) < 2:
            continue
        agent_ids = frame["agent_id"].to_numpy()
        coords = frame[["x", "y"]].to_numpy()
        radii = frame["radius"].to_numpy()
        dx = coords[:, 0:1] - coords[:, 0]
        dy = coords[:, 1:2] - coords[:, 1]
        distances = np.sqrt(dx**2 + dy**2)
        thresholds = radii[:, None] + radii
        collision_mask = distances < thresholds
        np.fill_diagonal(collision_mask, False)
        colliding_indices = np.argwhere(collision_mask)
        for i, j in colliding_indices:
            pair = frozenset([agent_ids[i], agent_ids[j]])
            collision_pairs.add(pair)
    return len(collision_pairs)
