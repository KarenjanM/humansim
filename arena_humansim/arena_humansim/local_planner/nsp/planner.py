"""Neural Social Physics local planner (Yue/Manocha/Wang, ECCV 2022).

Loads SDD-pretrained `_wo` checkpoints, runs the goal+collision branches at the model's
native 0.4 s cadence, and holds the predicted velocity between calls so the per-tick
contract is satisfied at any sim dt. Walls are not fed to the model - they are handled
downstream by collision/wall_projection.
"""

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
from ..socialgail.history import HistoryBuffer
from .scaling import assemble_supplement, meters_to_pixels, pixels_to_meters, velocity_from_history

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool

    from .model import NSP

_EPS = 1e-6
_DT_BRIDGE_EPS = 1e-9
_DEFAULT_NSP_DT = 0.4
_DEFAULT_PAST_LENGTH = 8
_DEFAULT_FUTURE_LENGTH = 12
_DEFAULT_MAX_PEDS = 25
_DEFAULT_METERS_PER_PIXEL = 0.05
_DEFAULT_SIGMA = 100.0

_UPSTREAM_CHECKPOINT_URL = "https://raw.githubusercontent.com/realcrane/Human-Trajectory-Prediction-via-Neural-Social-Physics/main/saved_models/SDD_nsp_wo.pt"
_FETCH_TIMEOUT_SECONDS = 60.0


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "arena_humansim" / "nsp"


_DEFAULT_CHECKPOINT = _default_cache_dir() / "SDD_nsp_wo.pt"


class NSPPlanner(LocalPlanner):
    supports_pool: bool = True
    needs_global_subgoal: bool = True
    bypasses_kinematic_constraints: bool = True

    def __init__(
        self,
        checkpoint_path: str = "",
        meters_per_pixel: float = _DEFAULT_METERS_PER_PIXEL,
        past_length: int = _DEFAULT_PAST_LENGTH,
        future_length: int = _DEFAULT_FUTURE_LENGTH,
        max_peds: int = _DEFAULT_MAX_PEDS,
        nsp_dt: float = _DEFAULT_NSP_DT,
        device: str = "cpu",
        sigma: float = _DEFAULT_SIGMA,
    ):
        self._checkpoint_path = checkpoint_path or str(_DEFAULT_CHECKPOINT)
        self._meters_per_pixel = float(meters_per_pixel)
        self._past_length = int(past_length)
        self._future_length = int(future_length)
        self._max_peds = int(max_peds)
        self._nsp_dt = float(nsp_dt)
        self._device_str = device
        self._sigma_val = float(sigma)

        self._history = HistoryBuffer(past_len=past_length)
        self._cached_vel_m: dict[int, np.ndarray] = {}
        self._sim_time_since_nsp: float = nsp_dt

        self._model: NSP | None = None
        self._device = None
        self._sigma_t = None
        self._torch = None

    def apply_policy_params(self, params_json: str) -> None:
        if not params_json:
            return
        try:
            blob = json.loads(params_json)
        except (ValueError, TypeError):
            return
        if not isinstance(blob, dict):
            return
        if "checkpoint_path" in blob and isinstance(blob["checkpoint_path"], str):
            self._checkpoint_path = blob["checkpoint_path"]
            self._model = None
        if "device" in blob and isinstance(blob["device"], str):
            self._device_str = blob["device"]
            self._model = None
        if "meters_per_pixel" in blob:
            self._meters_per_pixel = float(blob["meters_per_pixel"])
        if "nsp_dt" in blob:
            self._nsp_dt = float(blob["nsp_dt"])
        if "max_peds" in blob:
            self._max_peds = int(blob["max_peds"])
        if "future_length" in blob:
            self._future_length = int(blob["future_length"])

    def _ensure_checkpoint_on_disk(self) -> None:
        path = Path(self._checkpoint_path)
        if path.exists():
            return
        if path != _DEFAULT_CHECKPOINT:
            raise FileNotFoundError(f"NSP checkpoint not found at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        tmp = Path(tmp_name)
        self._logger.info(f"Fetching NSP checkpoint from {_UPSTREAM_CHECKPOINT_URL} -> {path}")
        try:
            with os.fdopen(fd, "wb") as f, urllib.request.urlopen(_UPSTREAM_CHECKPOINT_URL, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        from .model import NSP

        if not self._checkpoint_path:
            raise RuntimeError("NSPPlanner requires checkpoint_path; pass nsp.checkpoint_path in config")
        self._ensure_checkpoint_on_disk()

        self._torch = torch
        self._device = torch.device(self._device_str)
        try:
            ckpt = torch.load(self._checkpoint_path, map_location=self._device, weights_only=True)
        except (TypeError, RuntimeError):
            ckpt = torch.load(self._checkpoint_path, map_location=self._device)
        params = ckpt["hyper_params"]
        enc = tuple(params["enc_size"]) if isinstance(params["enc_size"], (list, tuple)) else (params["enc_size"],)
        dec = tuple(params["dec_size"]) if isinstance(params["dec_size"], (list, tuple)) else (params["dec_size"],)

        model = (
            NSP(
                input_size=int(params["input_size"]),
                embedding_size=int(params["embedding_size"]),
                rnn_size=int(params["rnn_size"]),
                output_size=int(params["output_size"]),
                enc_size=enc,
                dec_size=dec,
            )
            .double()
            .eval()
            .to(self._device)
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._sigma_t = torch.tensor(self._sigma_val, dtype=torch.float64, device=self._device)
        self._logger.info(f"Loaded NSP checkpoint from {self._checkpoint_path}")

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            self._cached_vel_m.clear()
            return

        self._sim_time_since_nsp += dt
        if self._sim_time_since_nsp + _DT_BRIDGE_EPS < self._nsp_dt and self._cached_vel_m:
            self._apply_cached_to_pool(pool, n)
            return
        self._sim_time_since_nsp = 0.0

        self._ensure_model()

        agent_ids = pool.agent_ids[:n]
        pos_m = pool.pos[:n]
        vel_m = pool.vel[:n]
        goal_m = pool.goal_pos[:n]
        has_goal = pool.has_goal[:n]
        max_v = pool.max_velocity[:n]

        pos_px = meters_to_pixels(pos_m, self._meters_per_pixel)
        all_vel_px = meters_to_pixels(vel_m, self._meters_per_pixel)
        goal_px = meters_to_pixels(goal_m, self._meters_per_pixel)

        self._history.update_many(agent_ids, pos_px)

        history_px = self._gather_history(agent_ids, pos_px)
        first_frame_px = history_px[:, 0, :].copy()
        history_translated = history_px - first_frame_px[:, None, :]
        history_vel = velocity_from_history(history_px, self._nsp_dt)

        supp_px = assemble_supplement(
            pos_px,
            pool.neighbor_indptr,
            pool.neighbor_indices,
            pos_px,
            all_vel_px,
            self._max_peds,
        )
        for i in range(n):
            k = int(supp_px[i, -1, 1])
            if k > 0:
                supp_px[i, :k, :2] -= first_frame_px[i]

        goal_translated_px = goal_px - first_frame_px
        last_obs = history_translated[:, -1, :]
        desired_vel_px = pool.desired_vel[:n] / self._meters_per_pixel
        initial_speeds_px = self._initial_speeds(goal_translated_px, last_obs, desired_vel_px)

        w_v_px = self._forward(history_translated, history_vel, supp_px, goal_translated_px, initial_speeds_px)

        new_vel_m = pixels_to_meters(w_v_px, self._meters_per_pixel)

        speed = np.hypot(new_vel_m[:, 0], new_vel_m[:, 1])
        too_fast = speed > max_v
        if np.any(too_fast):
            new_vel_m[too_fast] *= (max_v[too_fast] / np.maximum(speed[too_fast], _EPS))[:, None]
        new_vel_m[~has_goal] = 0.0

        self._cached_vel_m = {int(agent_ids[i]): new_vel_m[i].copy() for i in range(n)}
        pool.vel[:n] = new_vel_m

        self._history.evict(agent_ids)

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            return {}

        self._sim_time_since_nsp += dt
        if self._sim_time_since_nsp + _DT_BRIDGE_EPS < self._nsp_dt and self._cached_vel_m:
            return self._cached_dict_for(agents)
        self._sim_time_since_nsp = 0.0

        self._ensure_model()

        n = len(agents)
        agent_ids_list = [int(a.state.agent_id) for a in agents]
        agent_ids = np.array(agent_ids_list, dtype=np.int32)

        pos_m = np.array([[a.state.pose.x, a.state.pose.y] for a in agents], dtype=np.float64)
        vel_m = np.array([list(a.state.velocity) for a in agents], dtype=np.float64)
        max_v = np.array([a.params.max_velocity for a in agents], dtype=np.float64)

        goal_m = np.zeros_like(pos_m)
        has_goal = np.zeros(n, dtype=np.bool_)
        for i, a in enumerate(agents):
            g = global_goals.get(a.state.agent_id)
            if g is not None:
                goal_m[i, 0] = g.x
                goal_m[i, 1] = g.y
                has_goal[i] = True

        pos_px = meters_to_pixels(pos_m, self._meters_per_pixel)
        goal_px = meters_to_pixels(goal_m, self._meters_per_pixel)

        self._history.update_many(agent_ids, pos_px)
        history_px = self._gather_history(agent_ids, pos_px)
        first_frame_px = history_px[:, 0, :].copy()
        history_translated = history_px - first_frame_px[:, None, :]
        history_vel = velocity_from_history(history_px, self._nsp_dt)

        nbr_indptr = np.zeros(n + 1, dtype=np.int32)
        nbr_indices_list: list[int] = []
        nbr_pos_list: list[np.ndarray] = []
        nbr_vel_list: list[np.ndarray] = []
        agent_index = {aid: i for i, aid in enumerate(agent_ids_list)}
        for i, a in enumerate(agents):
            count = 0
            for ob in a.belief.observed_agents if a.belief is not None else ():
                if ob.agent_id == a.state.agent_id:
                    continue
                idx = agent_index.get(ob.agent_id)
                if idx is not None:
                    nbr_indices_list.append(idx)
                else:
                    nbr_pos_list.append(meters_to_pixels(np.array([ob.pose.x, ob.pose.y]), self._meters_per_pixel))
                    nbr_vel_list.append(meters_to_pixels(np.array(list(ob.velocity)), self._meters_per_pixel))
                    nbr_indices_list.append(n + len(nbr_pos_list) - 1)
                count += 1
            nbr_indptr[i + 1] = nbr_indptr[i] + count

        if nbr_pos_list:
            extra_pos = np.stack(nbr_pos_list)
            extra_vel = np.stack(nbr_vel_list)
        else:
            extra_pos = np.zeros((0, 2), dtype=np.float64)
            extra_vel = np.zeros((0, 2), dtype=np.float64)

        all_pos_px = np.concatenate([pos_px, extra_pos], axis=0) if extra_pos.size else pos_px
        all_vel_px = np.concatenate([meters_to_pixels(vel_m, self._meters_per_pixel), extra_vel], axis=0) if extra_vel.size else meters_to_pixels(vel_m, self._meters_per_pixel)
        nbr_indices = np.array(nbr_indices_list, dtype=np.int32) if nbr_indices_list else np.empty(0, dtype=np.int32)

        supp_px = assemble_supplement(pos_px, nbr_indptr, nbr_indices, all_pos_px, all_vel_px, self._max_peds)
        for i in range(n):
            k = int(supp_px[i, -1, 1])
            if k > 0:
                supp_px[i, :k, :2] -= first_frame_px[i]

        goal_translated_px = goal_px - first_frame_px
        last_obs = history_translated[:, -1, :]
        desired_vel_px = np.array([a.params.desired_velocity for a in agents], dtype=np.float64) / self._meters_per_pixel
        initial_speeds_px = self._initial_speeds(goal_translated_px, last_obs, desired_vel_px)

        w_v_px = self._forward(history_translated, history_vel, supp_px, goal_translated_px, initial_speeds_px)
        new_vel_m = pixels_to_meters(w_v_px, self._meters_per_pixel)

        speed = np.hypot(new_vel_m[:, 0], new_vel_m[:, 1])
        too_fast = speed > max_v
        if np.any(too_fast):
            new_vel_m[too_fast] *= (max_v[too_fast] / np.maximum(speed[too_fast], _EPS))[:, None]
        new_vel_m[~has_goal] = 0.0

        out: dict[int, tuple[float, float]] = {}
        self._cached_vel_m.clear()
        for i, aid in enumerate(agent_ids_list):
            v = (float(new_vel_m[i, 0]), float(new_vel_m[i, 1]))
            out[aid] = v
            self._cached_vel_m[aid] = new_vel_m[i].copy()
        self._history.evict(agent_ids)
        return out

    def _initial_speeds(self, goal_translated_px: np.ndarray, last_obs_px: np.ndarray, desired_vel_px: np.ndarray) -> np.ndarray:
        # NSP's F0 = (initial_speeds*e - current_vel) / tau. Upstream training set initial_speeds to
        # ||dest - last_obs|| / 4.8 because dest was always a far-future trajectory endpoint, so
        # this evaluated to roughly the agent's walking speed. In our deployment dest is whatever
        # subgoal the global planner produced - often only meters away - so that formula collapses
        # F0 to an exponential decay that never lets the agent arrive. Use the agent's desired
        # walking speed instead, capped only when remaining distance would overshoot in one step.
        dist_px = np.linalg.norm(goal_translated_px - last_obs_px, axis=1)
        cap_px = dist_px / self._nsp_dt
        speeds = np.minimum(desired_vel_px, cap_px)
        return speeds[:, None]

    def _gather_history(self, agent_ids: np.ndarray, pos_px: np.ndarray) -> np.ndarray:
        n = pos_px.shape[0]
        out = np.empty((n, self._past_length, 2), dtype=np.float64)
        L = self._past_length
        for i in range(n):
            aid = int(agent_ids[i])
            buf = self._history._buf.get(aid)
            stored = list(buf) if buf is not None else []
            pad = stored[-1] if stored else pos_px[i]
            for k in range(L):
                age = L - 1 - k
                out[i, k, :] = stored[age] if age < len(stored) else pad
        return out

    def _forward(
        self,
        history_translated: np.ndarray,
        history_vel: np.ndarray,
        supp_px: np.ndarray,
        goal_translated_px: np.ndarray,
        initial_speeds_px: np.ndarray,
    ) -> np.ndarray:
        torch = self._torch
        device = self._device
        model = self._model
        assert torch is not None and device is not None and model is not None

        n = history_translated.shape[0]
        traj = np.concatenate([history_translated, history_vel], axis=-1)
        traj_t = torch.from_numpy(traj).to(device)
        supp_t = torch.from_numpy(supp_px).to(device)
        dest_t = torch.from_numpy(goal_translated_px).to(device)
        init_speeds_t = torch.from_numpy(initial_speeds_px).to(device)

        rnn_size = model.cell1.hidden_size
        h1 = torch.zeros(n, rnn_size, dtype=torch.float64, device=device)
        c1 = torch.zeros(n, rnn_size, dtype=torch.float64, device=device)
        h2 = torch.zeros(n, rnn_size, dtype=torch.float64, device=device)
        c2 = torch.zeros(n, rnn_size, dtype=torch.float64, device=device)

        out1 = None
        out2 = None
        with torch.no_grad():
            for m in range(1, self._past_length):
                input_lstm = traj_t[:, m, :]
                out1, h1, c1, out2, h2, c2 = model.forward_lstm(input_lstm, h1, c1, h2, c2)
            assert out1 is not None and out2 is not None and self._sigma_t is not None
            current_step = traj_t[:, -1, :2]
            current_vel = traj_t[:, -1, 2:]
            coefficients, curr_supp = model.forward_coefficient_people(out2, supp_t, current_step, current_vel, device)
            _, w_v = model.forward_next_step(current_step, current_vel, init_speeds_t, dest_t, out1, coefficients, curr_supp, self._sigma_t, device)

        return w_v.cpu().numpy()

    def _apply_cached_to_pool(self, pool: AgentPool, n: int) -> None:
        ids = pool.agent_ids[:n]
        for i in range(n):
            v = self._cached_vel_m.get(int(ids[i]))
            if v is None:
                pool.vel[i] = 0.0
            else:
                pool.vel[i] = v

    def _cached_dict_for(self, agents: Sequence[BaseAgent]) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for a in agents:
            v = self._cached_vel_m.get(int(a.state.agent_id))
            if v is None:
                out[int(a.state.agent_id)] = (0.0, 0.0)
            else:
                out[int(a.state.agent_id)] = (float(v[0]), float(v[1]))
        return out
