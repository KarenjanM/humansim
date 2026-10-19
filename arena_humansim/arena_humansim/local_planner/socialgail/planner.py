from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import Pose2D

from .. import LocalPlanner

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool


_TORCH_INSTALL_HINT = "SocialGAIL requires torch and torch_geometric. Install with: pip install torch torch_geometric"

_DEFAULT_DECISION_INTERVAL_STEPS = 8
_DEFAULT_RADIUS = 6.0
_DEFAULT_PAST_LEN = 5
_DEFAULT_PADD_TO_NUMBER = 60
_DEFAULT_FEATURE_LEN = 5
_DEFAULT_ACTION_SCALE = 2.0
_DEFAULT_GRAPH_FEATURE_CHANNELS = 5
_DEFAULT_ACTION_DIM = 2
_GOAL_REFRESH_DIST = 0.1
_GOAL_V_MIN_NORM = 1e-3

_WEIGHTS_URL = "https://github.com/William-island/SocialGAIL/raw/main/logs/good_models/socialgail/best.pt"


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "arena_humansim" / "socialgail"


def _fetch_weights(dest: Path, url: str = _WEIGHTS_URL) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as resp, tmp_path.open("wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _try_import_torch() -> tuple:
    try:
        import torch
        from torch_geometric.data import Batch
    except ImportError as e:
        raise ImportError(_TORCH_INSTALL_HINT) from e
    return torch, Batch


class SocialGAILPlanner(LocalPlanner):
    """ICRA 2024 SocialGAIL crowd-sim policy.

    Inference-only wrapper around the upstream HGNN actor; weights from
    https://github.com/William-island/SocialGAIL (MIT). Decision rate is fixed
    at 0.4 s (training distribution); on intermediate sim ticks the velocity
    held over from the last decision is preserved (zero-order hold).

    Walls are not modeled by the network. Rely on the wall_projection collision
    pass downstream to keep agents off geometry.
    """

    supports_pool: bool = True
    needs_global_subgoal: bool = True
    bypasses_kinematic_constraints: bool = True

    def __init__(
        self,
        weights_path: str = "",
        device: str = "cpu",
        decision_interval_steps: int = _DEFAULT_DECISION_INTERVAL_STEPS,
        radius: float = _DEFAULT_RADIUS,
        past_len: int = _DEFAULT_PAST_LEN,
        padd_to_number: int = _DEFAULT_PADD_TO_NUMBER,
        feature_len: int = _DEFAULT_FEATURE_LEN,
        action_scale: float = _DEFAULT_ACTION_SCALE,
    ):
        self._weights_path = weights_path or str(_default_cache_dir() / "best.pt")
        self._device_str = device
        self.decision_interval_steps = max(1, int(decision_interval_steps))
        self.radius = float(radius)
        self.past_len = int(past_len)
        self.padd_to_number = int(padd_to_number)
        self.feature_len = int(feature_len)
        self.action_scale = float(action_scale)

        from .history import HistoryBuffer

        self._history = HistoryBuffer(past_len=self.past_len)
        # Per-agent pinned (goal, R, R_inv) - recomputed only when the goal
        # changes by more than _GOAL_REFRESH_DIST. Upstream pins R at episode
        # start; we approximate by pinning at first sight of each goal.
        self._rot_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._tick = 0
        self._actor = None
        self._torch = None
        self._Batch = None
        self._device = None

    def _ensure_actor(self) -> None:
        if self._actor is not None:
            return
        torch, Batch = _try_import_torch()
        from .net import GraphStateIndependentPolicy

        self._torch = torch
        self._Batch = Batch
        self._device = torch.device(self._device_str)

        actor = GraphStateIndependentPolicy(
            in_channels=_DEFAULT_GRAPH_FEATURE_CHANNELS,
            action_shape=(_DEFAULT_ACTION_DIM,),
            final_mlp_hidden_width=64,
        ).to(self._device)

        weights = Path(self._weights_path)
        if not weights.exists():
            self._logger.warning(f"Fetching SocialGAIL weights to {weights} from {_WEIGHTS_URL}")
            _fetch_weights(weights)

        try:
            ckpt = torch.load(str(weights), map_location=self._device, weights_only=True)
        except (TypeError, RuntimeError):
            ckpt = torch.load(str(weights), map_location=self._device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        self._actor = actor
        self._logger.info(f"Loaded SocialGAIL weights from {weights} on {self._device_str}")

    def apply_policy_params(self, params_json: str) -> None:
        if not params_json:
            return
        try:
            blob = json.loads(params_json)
        except (ValueError, TypeError):
            return
        if not isinstance(blob, dict):
            return
        if "weights_path" in blob and isinstance(blob["weights_path"], str):
            self._weights_path = blob["weights_path"]
            self._actor = None
        if "device" in blob and isinstance(blob["device"], str):
            self._device_str = blob["device"]
            self._actor = None
        if "decision_interval_steps" in blob:
            self.decision_interval_steps = max(1, int(blob["decision_interval_steps"]))

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            return

        active = pool.has_goal[:n]
        is_decision = (self._tick % self.decision_interval_steps) == 0
        self._tick += 1

        if not is_decision:
            return
        if not np.any(active):
            self._history.evict(int(a) for a in pool.agent_ids[:n])
            return

        active_idx = np.where(active)[0]
        active_aids = pool.agent_ids[active_idx]
        active_pos = pool.pos[active_idx]
        active_goal = pool.goal_pos[active_idx]

        all_aids = pool.agent_ids[:n]
        all_pos = pool.pos[:n]

        self._ensure_actor()

        from .obs import build_obs_for_agent, rotate_from_goal_frame, rotate_to_goal_frame

        decision_dt = self.decision_interval_steps * dt

        graphs = []
        unrot = np.empty((len(active_idx), 2, 2), dtype=np.float64)
        for k in range(len(active_idx)):
            i = int(active_idx[k])
            aid = int(active_aids[k])
            pos_i = active_pos[k]
            goal_i = active_goal[k]

            cached = self._rot_cache.get(aid)
            if cached is None or float(np.hypot(goal_i[0] - cached[0][0], goal_i[1] - cached[0][1])) > _GOAL_REFRESH_DIST:
                pin_goal_v = goal_i - pos_i
                if float(np.hypot(pin_goal_v[0], pin_goal_v[1])) < _GOAL_V_MIN_NORM:
                    pin_goal_v = np.array([1.0, 0.0])
                R = rotate_to_goal_frame(pin_goal_v)
                R_inv = rotate_from_goal_frame(pin_goal_v)
                self._rot_cache[aid] = (goal_i.copy(), R, R_inv)
            else:
                _, R, R_inv = cached

            start = int(pool.neighbor_indptr[i])
            stop = int(pool.neighbor_indptr[i + 1])
            nbr_idx = pool.neighbor_indices[start:stop]
            nbr_aids = pool.agent_ids[nbr_idx]
            nbr_pos = pool.pos[nbr_idx]

            graphs.append(
                build_obs_for_agent(
                    aid=aid,
                    pos=pos_i,
                    goal=goal_i,
                    R=R,
                    neighbor_aids=nbr_aids,
                    neighbor_positions=nbr_pos,
                    history=self._history,
                    decision_dt=decision_dt,
                    radius=self.radius,
                    past_len=self.past_len,
                    padd_to_number=self.padd_to_number,
                    feature_len=self.feature_len,
                )
            )
            unrot[k] = R_inv

        torch = self._torch
        assert torch is not None and self._Batch is not None and self._actor is not None
        batch = self._Batch.from_data_list(graphs).to(self._device)
        with torch.no_grad():
            actions = self._actor(batch).cpu().numpy()

        actions_scaled = actions.astype(np.float64) * self.action_scale
        vel_world = np.einsum("ij,ijk->ik", actions_scaled, unrot)

        max_v = pool.max_velocity[active_idx]
        speeds = np.hypot(vel_world[:, 0], vel_world[:, 1])
        too_fast = speeds > max_v
        if np.any(too_fast):
            scale = np.where(speeds > 0, max_v / np.maximum(speeds, 1e-9), 1.0)
            vel_world[too_fast] *= scale[too_fast, None]

        pool.vel[active_idx] = vel_world

        self._history.update_many(all_aids, all_pos)
        alive = {int(a) for a in all_aids}
        self._history.evict(alive)
        for aid in list(self._rot_cache.keys()):
            if aid not in alive:
                del self._rot_cache[aid]

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        velocities: dict[int, tuple[float, float]] = {}
        for agent in agents:
            velocities[agent.state.agent_id] = (0.0, 0.0)
        return velocities
