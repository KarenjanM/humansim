from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import Pose2D

from ..base import RobotPolicy, fetch_to_disk, policy_cache_dir
from .obs import build_robot_node, build_spatial_edges, build_temporal_edge

if TYPE_CHECKING:
    from torch import Tensor

    from .model import SRNN

_CHECKPOINT_URL = "https://github.com/Shuijing725/CrowdNav_DSRNN/raw/main/data/example_model_unicycle/checkpoints/55554.pt"
_CHECKPOINT_FILENAME = "55554.pt"
_TRAINING_HUMAN_NUM = 5
_TRAINING_TIME_STEP = 0.1
_DELTA_V_CLIP = 0.1
_DELTA_THETA_CLIP = 0.1


class DSRNNPlanner(RobotPolicy):
    """Decentralized structural RNN robot navigator (Liu et al., ICRA 2021).

    Loads the upstream `example_model_unicycle/55554.pt` checkpoint trained with
    human_num=5 at dt=0.1. Action is (delta_v, delta_theta); we integrate
    desiredVelocity and theta per agent, then emit world-frame (vx, vy)."""

    bypasses_kinematic_constraints: bool = True
    needs_global_subgoal: bool = True
    supports_pool: bool = False

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        history_len: int = 8,
        max_humans: int = 20,
        seed: int = 0,
    ):
        self._checkpoint_path = checkpoint_path or str(policy_cache_dir("dsrnn") / _CHECKPOINT_FILENAME)
        self._device_str = device
        self._history_len = int(history_len)
        self._max_humans = int(max_humans)
        self._seed = int(seed)

        self._model: SRNN | None = None
        self._device = None
        self._torch = None

        self._hidden: dict[int, dict[str, Tensor]] = {}
        self._desired_v: dict[int, float] = {}
        self._theta: dict[int, float] = {}

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
        if "history_len" in blob:
            self._history_len = int(blob["history_len"])
        if "max_humans" in blob:
            self._max_humans = int(blob["max_humans"])
        if "seed" in blob:
            self._seed = int(blob["seed"])
            self._model = None

    def _ensure_checkpoint_on_disk(self) -> None:
        path = Path(self._checkpoint_path)
        if path.exists():
            return
        default = policy_cache_dir("dsrnn") / _CHECKPOINT_FILENAME
        if path != default:
            raise FileNotFoundError(f"DS-RNN checkpoint not found at {path}")
        self._logger.info(f"Fetching DS-RNN checkpoint from {_CHECKPOINT_URL} -> {path}")
        fetch_to_disk(_CHECKPOINT_URL, path)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._seed_rngs(self._seed)
        import torch

        from .model import SRNN

        self._ensure_checkpoint_on_disk()
        self._torch = torch
        self._device = torch.device(self._device_str)

        # The checkpoint is saved as a Policy state_dict with a `base.` prefix
        # for SRNN params and a `dist.` prefix for the action head; we load both
        # with stripped prefixes to populate our submodule.
        try:
            ckpt = torch.load(self._checkpoint_path, map_location=self._device, weights_only=True)
        except (TypeError, RuntimeError):
            ckpt = torch.load(self._checkpoint_path, map_location=self._device)

        state = ckpt
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]

        base_state: dict[str, object] = {}
        for k, v in state.items():
            if k.startswith("base."):
                base_state[k[len("base.") :]] = v
            elif k.startswith("dist."):
                base_state["dist." + k[len("dist.") :]] = v
            else:
                base_state[k] = v

        model = SRNN(human_num=_TRAINING_HUMAN_NUM).to(self._device)
        model.load_state_dict(base_state, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._logger.info(f"Loaded DS-RNN checkpoint from {self._checkpoint_path} on {self._device_str}")

    def _get_or_init_state(self, agent: BaseAgent) -> tuple[dict[str, Tensor], float, float]:
        aid = agent.state.agent_id
        hxs = self._hidden.get(aid)
        if hxs is None:
            assert self._model is not None and self._device is not None
            hxs = self._model.initial_hidden(self._device)
            self._hidden[aid] = hxs
        v = self._desired_v.get(aid)
        if v is None:
            v = 0.0
            self._desired_v[aid] = v
        theta = self._theta.get(aid)
        if theta is None:
            vx, vy = agent.state.velocity
            theta = math.atan2(vy, vx) if (vx * vx + vy * vy) > 1e-9 else 0.0
            self._theta[aid] = theta
        return hxs, v, theta

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            self._hidden.clear()
            self._desired_v.clear()
            self._theta.clear()
            return {}

        self._ensure_model()
        torch = self._torch
        assert torch is not None and self._model is not None and self._device is not None

        out: dict[int, tuple[float, float]] = {}
        alive: set[int] = set()
        for agent in agents:
            aid = agent.state.agent_id
            alive.add(aid)
            goal = global_goals.get(aid)
            if goal is None:
                out[aid] = (0.0, 0.0)
                continue

            hxs, desired_v, theta = self._get_or_init_state(agent)

            robot_node = build_robot_node(agent, goal, theta)
            temporal_edge = build_temporal_edge(agent)
            spatial_edges = build_spatial_edges(agent, _TRAINING_HUMAN_NUM)

            rn_t = torch.from_numpy(robot_node).to(self._device).view(1, 1, 7)
            te_t = torch.from_numpy(temporal_edge).to(self._device).view(1, 1, 2)
            se_t = torch.from_numpy(spatial_edges).to(self._device).view(1, _TRAINING_HUMAN_NUM, 2)

            with torch.no_grad():
                action_t, new_hxs = self._model.act(rn_t, te_t, se_t, hxs)
            action = action_t.view(-1).cpu().numpy()
            self._hidden[aid] = new_hxs

            v_pref = agent.params.max_velocity
            delta_v = float(np.clip(action[0], -_DELTA_V_CLIP, _DELTA_V_CLIP))
            delta_theta = float(np.clip(action[1], -_DELTA_THETA_CLIP, _DELTA_THETA_CLIP))

            new_desired_v = float(np.clip(desired_v + delta_v, -v_pref, v_pref))
            new_theta = (theta + delta_theta) % (2.0 * math.pi)

            self._desired_v[aid] = new_desired_v
            self._theta[aid] = new_theta

            vx = new_desired_v * math.cos(new_theta)
            vy = new_desired_v * math.sin(new_theta)
            out[aid] = (vx, vy)

        for aid in list(self._hidden.keys()):
            if aid not in alive:
                del self._hidden[aid]
                self._desired_v.pop(aid, None)
                self._theta.pop(aid, None)
        return out


if __name__ == "__main__":
    import types
    from collections.abc import Sequence as _Seq
    from typing import Any as _Any

    class _FakePose:
        def __init__(self, x: float, y: float, theta: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.theta = theta

    class _FakeBelief:
        def __init__(self, observed: _Seq[_Any]) -> None:
            self.observed_agents = observed

    class _FakeObserved:
        def __init__(
            self,
            aid: int,
            x: float,
            y: float,
            vx: float = 0.0,
            vy: float = 0.0,
            r: float = 0.3,
        ) -> None:
            self.agent_id = aid
            self.pose = _FakePose(x, y)
            self.velocity = (vx, vy)
            self.radius = r

    def make_agent(aid: int, x: float, y: float, observed: _Seq[_Any]) -> types.SimpleNamespace:
        a = types.SimpleNamespace()
        a.state = types.SimpleNamespace(
            agent_id=aid,
            pose=_FakePose(x, y),
            velocity=(0.0, 0.0),
        )
        a.params = types.SimpleNamespace(
            agent_radius=0.3,
            desired_velocity=1.0,
            max_velocity=1.0,
        )
        a.belief = _FakeBelief(observed)
        return a

    obs1 = [_FakeObserved(10, 1.0, 0.5), _FakeObserved(11, -1.0, 1.5)]
    agents = [make_agent(1, 0.0, 0.0, obs1)]
    goals = {1: _FakePose(5.0, 0.0)}

    p1 = DSRNNPlanner(seed=42)
    v1 = p1.compute(agents, goals)  # type: ignore[arg-type]

    p2 = DSRNNPlanner(seed=42)
    v2 = p2.compute(agents, goals)  # type: ignore[arg-type]

    a1 = np.array(list(v1.values()))
    a2 = np.array(list(v2.values()))
    assert np.array_equal(a1, a2), f"mismatch: {a1} vs {a2}"
    print("DSRNN determinism OK")
