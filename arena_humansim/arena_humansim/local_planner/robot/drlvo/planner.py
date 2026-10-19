from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import Pose2D, Segments

from ..base import RobotPolicy, fetch_to_disk, policy_cache_dir
from .obs import (
    NUM_BEAMS,
    OBS_FLAT,
    ScanHistory,
    assemble_observation,
)

_UPSTREAM_CHECKPOINT_URL = "https://github.com/TempleRAIL/drl_vo_nav/raw/drl_vo/drl_vo/src/model/drl_vo.zip"

# Upstream stop-gate constants (drl_vo_inference.py).
_GOAL_MARGIN = 0.9
_OBSTACLE_MARGIN = 0.4
_VX_MIN = 0.0
_VX_MAX = 0.5
_WZ_MIN = -2.0
_WZ_MAX = 2.0
_OBSTACLE_TURN_RATE = 0.7
_ARRIVAL_M = 0.05
_DEFAULT_NEIGHBOR_RADIUS = 0.35


class DRLVOPlanner(RobotPolicy):
    bypasses_kinematic_constraints: bool = True
    needs_global_subgoal: bool = True
    supports_pool: bool = False

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        num_beams: int = NUM_BEAMS,
        max_range: float = 6.0,
        grid_size: int = 80,
        grid_extent: float = 12.0,
        seed: int = 0,
    ):
        cache = policy_cache_dir("drlvo")
        self._checkpoint_path = checkpoint_path or str(cache / "drl_vo.zip")
        self._device_str = device
        self._num_beams = int(num_beams)
        self._max_range = float(max_range)
        # grid_size / grid_extent retained as constructor knobs but the upstream checkpoint locks
        # the obs layout to 80x80 over (20m x 20m); we honor the upstream layout.
        self._grid_size = int(grid_size)
        self._grid_extent = float(grid_extent)
        self._seed = int(seed)

        self._wall_segments: Segments = []
        self._wall_p1: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._wall_d: np.ndarray = np.empty((0, 2), dtype=np.float64)

        self._scan_history = ScanHistory(num_beams=self._num_beams)

        self._model = None

    def apply_policy_params(self, params_json: str) -> None:
        if not params_json:
            return
        try:
            blob = json.loads(params_json)
        except (ValueError, TypeError):
            return
        if not isinstance(blob, dict):
            return
        if isinstance(blob.get("checkpoint_path"), str):
            self._checkpoint_path = blob["checkpoint_path"]
            self._model = None
        if isinstance(blob.get("device"), str):
            self._device_str = blob["device"]
            self._model = None
        if "max_range" in blob:
            self._max_range = float(blob["max_range"])
        if "seed" in blob:
            self._seed = int(blob["seed"])

    def set_walls(self, segments: Segments) -> None:
        self._wall_segments = list(segments)
        if segments:
            arr = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
            self._wall_p1 = arr[:, 0, :]
            self._wall_d = arr[:, 1, :] - arr[:, 0, :]
        else:
            self._wall_p1 = np.empty((0, 2), dtype=np.float64)
            self._wall_d = np.empty((0, 2), dtype=np.float64)
        self._logger.info(f"Loaded {len(self._wall_segments)} wall segment(s)")

    def _ensure_checkpoint_on_disk(self) -> None:
        from pathlib import Path

        path = Path(self._checkpoint_path)
        if path.exists():
            return
        self._logger.info(f"Fetching DRL-VO checkpoint from {_UPSTREAM_CHECKPOINT_URL} -> {path}")
        fetch_to_disk(_UPSTREAM_CHECKPOINT_URL, path)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._ensure_checkpoint_on_disk()
        import sys

        import gymnasium
        import numpy as np
        from stable_baselines3 import PPO

        from . import custom_cnn_full

        # SB3 cloudpickled `custom_cnn_full.CustomCNN` and legacy gym.spaces.Box
        # instances at training time. The module aliases let PPO.load resolve
        # those names against what /opt/venv has (gymnasium, no legacy gym).
        sys.modules.setdefault("custom_cnn_full", custom_cnn_full)
        sys.modules.setdefault("gym", gymnasium)
        sys.modules.setdefault("gym.spaces", gymnasium.spaces)

        # Override pickle-reconstructed objects that don't survive Python 3.7->3.12
        # nor SB3 v1->v2 (cloudpickled lr_schedule/clip_range bytecode invalid;
        # legacy Box class identity != gymnasium.spaces.Box even after alias).
        custom_objects: dict[str, object] = {
            "observation_space": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(19202,), dtype=np.float32),
            "action_space": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "lr_schedule": (lambda _: 3.0e-4),
            "clip_range": (lambda _: 0.2),
        }

        self._seed_rngs(self._seed)
        self._model = PPO.load(self._checkpoint_path, device=self._device_str, custom_objects=custom_objects)
        self._yaw: dict[int, float] = {}
        self._logger.info(f"Loaded DRL-VO checkpoint from {self._checkpoint_path}")

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            return {}

        self._ensure_model()
        assert self._model is not None

        out: dict[int, tuple[float, float]] = {}
        keep: set[int] = set()

        for agent in agents:
            agent_id = int(agent.state.agent_id)
            keep.add(agent_id)
            goal = global_goals.get(agent_id)
            if goal is None:
                out[agent_id] = (0.0, 0.0)
                continue

            px = agent.state.pose.x
            py = agent.state.pose.y
            yaw = self._yaw.get(agent_id)
            if yaw is None:
                yaw = float(agent.state.pose.theta)
                self._yaw[agent_id] = yaw

            neighbors = []
            if agent.belief is not None:
                neighbors = [ob for ob in agent.belief.observed_agents if ob.agent_id != agent_id]

            obs, min_scan = assemble_observation(
                px=px,
                py=py,
                yaw=yaw,
                goal_xy=(goal.x, goal.y),
                walls_p1=self._wall_p1,
                walls_d=self._wall_d,
                neighbors=neighbors,
                neighbor_radius=_DEFAULT_NEIGHBOR_RADIUS,
                scan_history=self._scan_history,
                agent_id=agent_id,
                num_beams=self._num_beams,
                max_range=self._max_range,
                fov_rad=2.0 * math.pi,
            )

            # Upstream goal margin and obstacle margin checks (drl_vo_inference.py).
            goal_dx = goal.x - px
            goal_dy = goal.y - py
            goal_dist = math.hypot(goal_dx, goal_dy)
            if goal_dist <= _GOAL_MARGIN:
                if goal_dist <= _ARRIVAL_M:
                    out[agent_id] = (0.0, 0.0)
                    continue
                yaw = math.atan2(goal_dy, goal_dx)
                self._yaw[agent_id] = yaw
                out[agent_id] = _twist_to_world(min(_VX_MAX, goal_dist), yaw)
                continue
            if min_scan <= _OBSTACLE_MARGIN:
                yaw = _wrap_angle(yaw + _OBSTACLE_TURN_RATE * dt)
                self._yaw[agent_id] = yaw
                out[agent_id] = _twist_to_world(0.0, yaw)
                continue

            # Sanity: obs is finite and shape-correct.
            if obs.shape[0] != OBS_FLAT or not np.all(np.isfinite(obs)):
                out[agent_id] = (0.0, 0.0)
                continue

            # Upstream clips the goal scaler to [-1, 1] post-MaxAbsScaler implicitly when goal is
            # within [g_min, g_max]; outside we clamp to keep the obs in distribution.
            obs[-2:] = np.clip(obs[-2:], -1.0, 1.0)

            action, _states = self._model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            v_lin = (action[0] + 1.0) * (_VX_MAX - _VX_MIN) / 2.0 + _VX_MIN
            # Upstream caps v_lin at 0.5 m/s; rescale so the agent's desired_velocity is the new ceiling.
            if _VX_MAX > 0.0:
                v_lin *= float(agent.params.desired_velocity) / _VX_MAX
            w_ang = (action[1] + 1.0) * (_WZ_MAX - _WZ_MIN) / 2.0 + _WZ_MIN

            yaw = _wrap_angle(yaw + float(w_ang) * dt)
            self._yaw[agent_id] = yaw
            out[agent_id] = _twist_to_world(float(v_lin), yaw)

        self._scan_history.evict(keep)
        self._yaw = {aid: yaw for aid, yaw in self._yaw.items() if aid in keep}
        return out


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _twist_to_world(v_lin: float, yaw: float) -> tuple[float, float]:
    return v_lin * math.cos(yaw), v_lin * math.sin(yaw)


if __name__ == "__main__":
    # Determinism gate: build a synthetic agent, run compute twice, assert identity.
    import attrs

    from arena_humansim.utils.types import AgentState, BeliefState

    @attrs.define
    class _FakeParams:
        desired_velocity: float = 1.3
        max_velocity: float = 1.5
        agent_radius: float = 0.35

    @attrs.define
    class _FakeAgent:
        state: AgentState = attrs.Factory(lambda: AgentState(agent_id=0, pose=Pose2D(0.0, 0.0, 0.0), velocity=(0.0, 0.0)))
        params: _FakeParams = attrs.Factory(_FakeParams)
        belief: BeliefState = attrs.Factory(lambda: BeliefState(agent_id=0))

    a = _FakeAgent()
    a.belief.observed_agents.append(AgentState(agent_id=1, pose=Pose2D(2.0, 0.5, 0.0), velocity=(0.1, 0.0)))
    a.belief.observed_agents.append(AgentState(agent_id=2, pose=Pose2D(3.5, -1.0, 0.0), velocity=(-0.2, 0.05)))
    goals = {0: Pose2D(5.0, 0.0, 0.0)}
    walls: Segments = [((0.0, -3.0), (10.0, -3.0)), ((0.0, 3.0), (10.0, 3.0))]

    p1 = DRLVOPlanner(seed=42)
    p1.set_walls(walls)
    out1 = p1.compute([a], goals)  # type: ignore[arg-type]

    a2 = _FakeAgent()
    a2.belief.observed_agents.append(AgentState(agent_id=1, pose=Pose2D(2.0, 0.5, 0.0), velocity=(0.1, 0.0)))
    a2.belief.observed_agents.append(AgentState(agent_id=2, pose=Pose2D(3.5, -1.0, 0.0), velocity=(-0.2, 0.05)))

    p2 = DRLVOPlanner(seed=42)
    p2.set_walls(walls)
    out2 = p2.compute([a2], goals)  # type: ignore[arg-type]

    v1 = out1[0]
    v2 = out2[0]
    assert v1 == v2, f"non-deterministic: {v1} vs {v2}"
    print("DRLVO determinism OK")
