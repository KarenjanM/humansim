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
from .obs import build_observation

if TYPE_CHECKING:
    from .model import GA3CCADRLNet

_WEIGHTS_URL = "https://huggingface.co/arena-rosnav/cadrl/resolve/main/cadrl_iros18.pt"
_CONVERTED_NAME = "cadrl_iros18.pt"
_NUM_ACTIONS = 11
_MAX_OTHER_AGENTS = 10


def _build_actions_table() -> np.ndarray:
    a = np.mgrid[1.0:1.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 12].reshape(2, -1).T
    a = np.vstack([a, np.mgrid[0.5:0.6:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T])
    a = np.vstack([a, np.mgrid[0.0:0.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T])
    return a


_ACTIONS = _build_actions_table()


class CADRLPlanner(RobotPolicy):
    """GA3C-CADRL (IROS 2018) robot navigation policy.

    Loads the upstream IROS18 TF1 checkpoint, converts to PyTorch on first use,
    runs an RNN-over-other-agents head to pick a discrete (speed_frac, dheading)
    action, then emits world-frame (vx, vy)."""

    bypasses_kinematic_constraints: bool = True
    needs_global_subgoal: bool = True
    supports_pool: bool = False

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        max_other_agents: int = _MAX_OTHER_AGENTS,
        seed: int = 0,
    ):
        self._checkpoint_path = checkpoint_path or str(policy_cache_dir("cadrl") / _CONVERTED_NAME)
        self._device_str = device
        self._max_other_agents = int(max_other_agents)
        self._seed = int(seed)

        self._model: GA3CCADRLNet | None = None
        self._device = None
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
        if "max_other_agents" in blob:
            self._max_other_agents = int(blob["max_other_agents"])
            self._model = None
        if "seed" in blob:
            self._seed = int(blob["seed"])
            self._model = None

    def _ensure_checkpoint_on_disk(self) -> None:
        pt_path = Path(self._checkpoint_path)
        if pt_path.exists():
            return
        self._logger.info(f"Fetching CADRL weights {_WEIGHTS_URL} -> {pt_path}")
        fetch_to_disk(_WEIGHTS_URL, pt_path)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._seed_rngs(self._seed)
        import torch

        from .model import GA3CCADRLNet

        self._ensure_checkpoint_on_disk()
        self._torch = torch
        self._device = torch.device(self._device_str)

        try:
            ckpt = torch.load(self._checkpoint_path, map_location=self._device, weights_only=True)
        except (TypeError, RuntimeError):
            ckpt = torch.load(self._checkpoint_path, map_location=self._device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            num_actions = int(ckpt.get("num_actions", _NUM_ACTIONS))
            max_other = int(ckpt.get("max_other_agents", self._max_other_agents))
        else:
            state = ckpt
            num_actions = _NUM_ACTIONS
            max_other = self._max_other_agents
        self._max_other_agents = max_other

        model = GA3CCADRLNet(num_actions=num_actions, max_other_agents=max_other).to(self._device)
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._logger.info(f"Loaded CADRL checkpoint from {self._checkpoint_path} on {self._device_str}")

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            return {}

        self._ensure_model()
        torch = self._torch
        assert torch is not None and self._model is not None and self._device is not None

        host_list: list[np.ndarray] = []
        other_list: list[np.ndarray] = []
        seq_lens: list[int] = []
        keep_idx: list[int] = []
        out: dict[int, tuple[float, float]] = {}

        for i, agent in enumerate(agents):
            aid = agent.state.agent_id
            goal = global_goals.get(aid)
            if goal is None:
                out[aid] = (0.0, 0.0)
                continue
            host, other, n = build_observation(agent, goal, self._max_other_agents)
            host_list.append(host)
            other_list.append(other)
            seq_lens.append(n)
            keep_idx.append(i)

        if not host_list:
            return out

        host_t = torch.from_numpy(np.stack(host_list, axis=0)).to(self._device)
        other_t = torch.from_numpy(np.stack(other_list, axis=0)).to(self._device)
        seq_t = torch.tensor(seq_lens, dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits_p, _ = self._model(host_t, other_t, seq_t)
        action_idx = logits_p.argmax(dim=1).cpu().numpy()

        for k, agent_i in enumerate(keep_idx):
            agent = agents[agent_i]
            aid = agent.state.agent_id
            speed_frac, dheading = _ACTIONS[int(action_idx[k])]
            pref_speed = agent.params.desired_velocity
            speed = float(speed_frac) * pref_speed
            vx, vy = agent.state.velocity
            heading_global = math.atan2(vy, vx) if (vx * vx + vy * vy) > 1e-9 else _heading_to_goal(agent, global_goals[aid])
            new_heading = _wrap(heading_global + float(dheading))
            out[aid] = (speed * math.cos(new_heading), speed * math.sin(new_heading))
        return out


def _heading_to_goal(agent: BaseAgent, goal: Pose2D) -> float:
    return math.atan2(goal.y - agent.state.pose.y, goal.x - agent.state.pose.x)


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


if __name__ == "__main__":
    import types
    from collections.abc import Sequence as _Seq
    from typing import cast

    class _FakePose:
        def __init__(self, x: float, y: float, theta: float = 0.0):
            self.x = x
            self.y = y
            self.theta = theta

    class _FakeBelief:
        def __init__(self, observed: _Seq[object]) -> None:
            self.observed_agents = observed

    class _FakeObserved:
        def __init__(self, aid: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
            self.agent_id = aid
            self.pose = _FakePose(x, y)
            self.velocity = (vx, vy)

    def make_agent(aid: int, x: float, y: float, observed: _Seq[object]) -> BaseAgent:
        a = types.SimpleNamespace()
        a.state = types.SimpleNamespace(agent_id=aid, pose=_FakePose(x, y), velocity=(0.5, 0.0))
        a.params = types.SimpleNamespace(agent_radius=0.3, desired_velocity=1.0, max_velocity=1.0)
        a.belief = _FakeBelief(observed)
        return cast("BaseAgent", a)

    obs1 = [_FakeObserved(10, 1.0, 0.5, vx=-0.1), _FakeObserved(11, -1.0, 1.5)]
    agents = [make_agent(1, 0.0, 0.0, obs1)]
    goals = {1: Pose2D(x=5.0, y=0.0)}

    p1 = CADRLPlanner(seed=42)
    v1 = p1.compute(agents, goals)
    p2 = CADRLPlanner(seed=42)
    v2 = p2.compute(agents, goals)

    a1 = np.array(list(v1.values()))
    a2 = np.array(list(v2.values()))
    assert np.array_equal(a1, a2), f"mismatch: {a1} vs {a2}"
    print("CADRL determinism OK")
