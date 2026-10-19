"""Correspondence tests: a released driver against a public reference implementation.

    evaluate correspond [--drivers orca sfm straight] [--out_dir <dir>]

Each driver runs on held-out scenarios (none of them is a benchmark scenario) with
the same starts, goals, radii, preferred speeds and time step as its reference:

  orca      Python-RVO2 (the reference ORCA implementation), identical parameters
            (neighbor distance, max neighbours, time horizon, radius, max speed).
  sfm       pysocialforce (PedRepulsiveForce, the Helbing potential with the same
            v0 = A, sigma = B and tau) with the same relaxation time and radius.
            pysocialforce integrates the elliptical potential, the released SFM
            integrates the circular one, so this comparison measures the gap
            between the two published specifications as well as any adapter error.
  straight  closed form: constant preferred velocity toward the goal.

hsfm, nsp and socialgail have no public reference implementation that can be
driven step by step and are reported as `no reference`.

For every (driver, scenario) the per-agent trajectory Hausdorff distance and the
mean per-step velocity difference are recorded. The pass criterion is median
Hausdorff <= 0.35 m and mean velocity error <= 0.08 m/s.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import SampledParams, SampledPerception
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.utils.evaluation.metrics import pairwise_hausdorff
from arena_humansim.utils.types import AgentState, BeliefState, Pose2D

DT = 0.05
HORIZON_S = 25.0
GOAL_RADIUS = 0.15
RADIUS = 0.25
V_PREF = 1.1
V_MAX = 1.5
TAU = 0.5
A_REP = 2.1
B_REP = 0.3
ANISOTROPY = 0.5
NEIGHBOR_DIST = 5.0
MAX_NEIGHBORS = 10
TIME_HORIZON = 5.0
HAUSDORFF_PASS_M = 0.35
VEL_PASS_MPS = 0.08

Scenario = tuple[np.ndarray, np.ndarray]  # starts (n,2), goals (n,2)


def _circle(n: int, r: float) -> Scenario:
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    starts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    return starts, -starts


def _corridor_pass(n_per_side: int) -> Scenario:
    ys = np.linspace(-1.0, 1.0, n_per_side)
    left = np.stack([np.full(n_per_side, -6.0), ys], axis=1)
    right = np.stack([np.full(n_per_side, 6.0), ys[::-1]], axis=1)
    starts = np.concatenate([left, right])
    goals = np.concatenate([right, left])
    return starts, goals


def _crossing() -> Scenario:
    starts = np.array([[-5.0, 0.0], [5.0, 0.0], [0.0, -5.0], [0.0, 5.0], [-4.0, 1.5], [4.0, -1.5]])
    goals = -starts
    return starts, goals


def _sparse_random(n: int, seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    starts = rng.uniform(-6.0, 6.0, size=(n, 2))
    goals = rng.uniform(-6.0, 6.0, size=(n, 2))
    return starts, goals


def _perturbed(make: Callable[[], Scenario], seed: int) -> Callable[[], Scenario]:
    """Seeded +-5 cm start jitter against exactly symmetric deadlocks."""

    def build() -> Scenario:
        starts, goals = make()
        rng = np.random.default_rng(seed)
        return starts + rng.uniform(-0.05, 0.05, size=starts.shape), goals

    return build


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "circle_6": _perturbed(lambda: _circle(6, 4.0), 11),
    "circle_10": _perturbed(lambda: _circle(10, 5.0), 12),
    "corridor_pass_6": _perturbed(lambda: _corridor_pass(3), 13),
    "crossing_6": _perturbed(_crossing, 14),
    "sparse_random_10": lambda: _sparse_random(10, 3),
    "sparse_random_16": lambda: _sparse_random(16, 5),
}


def _params() -> SampledParams:
    return SampledParams(
        name="adult",
        desired_velocity=V_PREF,
        agent_radius=RADIUS,
        max_velocity=V_MAX,
        max_acceleration=1e9,
        max_deceleration=1e9,
        min_turning_radius=0.0,
        pivot_angular_velocity=1e9,
        reaction_time=0.0,
        personal_space_min=0.6,
        perception=SampledPerception(vision_range=1e3, vision_fov=360.0),
        local_planner_params={"relaxation_time": TAU, "repulsion_strength": A_REP, "repulsion_range": B_REP, "anisotropy": ANISOTROPY},
    )


def _pref_velocity(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    d = goal - pos
    dist = float(np.hypot(d[0], d[1]))
    if dist < GOAL_RADIUS:
        return np.zeros(2)
    return d / dist * V_PREF


def run_ours(driver: str, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """Trajectories (steps, n, 4) = x, y, vx, vy of the released driver."""
    planner = LocalPlanner.create(driver)
    planner.set_walls([])
    n = len(starts)
    agents: list[BaseAgent] = []
    for i in range(n):
        v0 = _pref_velocity(starts[i], goals[i])
        state = AgentState(agent_id=i + 1, pose=Pose2D(x=float(starts[i, 0]), y=float(starts[i, 1]), theta=float(math.atan2(v0[1], v0[0]))), velocity=(float(v0[0]), float(v0[1])), desired_velocity=V_PREF)
        agents.append(BaseAgent(state=state, params=_params(), global_planner=cast(Any, None), local_planner=cast(Any, planner), animation=cast(Any, None), belief=BeliefState(agent_id=i + 1)))
    goal_map = {i + 1: Pose2D(x=float(goals[i, 0]), y=float(goals[i, 1]), theta=0.0) for i in range(n)}
    steps = int(HORIZON_S / DT)
    out = np.zeros((steps, n, 4))
    for t in range(steps):
        for a in agents:
            a.belief = BeliefState(agent_id=a.state.agent_id, observed_agents=[b.state for b in agents if b is not a])
        vel = planner.compute(agents, goal_map, dt=DT)
        for i, a in enumerate(agents):
            out[t, i, 0], out[t, i, 1] = a.state.pose.x, a.state.pose.y
            vx, vy = vel.get(a.state.agent_id, (0.0, 0.0))
            if math.hypot(goals[i, 0] - a.state.pose.x, goals[i, 1] - a.state.pose.y) < GOAL_RADIUS:
                vx, vy = 0.0, 0.0
            out[t, i, 2], out[t, i, 3] = vx, vy
            a.state.velocity = (vx, vy)
            a.state.pose.x += vx * DT
            a.state.pose.y += vy * DT
            if abs(vx) + abs(vy) > 1e-9:
                a.state.pose.theta = math.atan2(vy, vx)
    return out


def run_rvo2(starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
    import rvo2  # pyright: ignore[reportMissingImports]

    sim = rvo2.PyRVOSimulator(DT, NEIGHBOR_DIST, MAX_NEIGHBORS, TIME_HORIZON, TIME_HORIZON, RADIUS, V_MAX)
    n = len(starts)
    ids = []
    for i in range(n):
        v0 = _pref_velocity(starts[i], goals[i])
        ids.append(sim.addAgent((float(starts[i, 0]), float(starts[i, 1])), NEIGHBOR_DIST, MAX_NEIGHBORS, TIME_HORIZON, TIME_HORIZON, RADIUS, V_MAX, (float(v0[0]), float(v0[1]))))
    steps = int(HORIZON_S / DT)
    out = np.zeros((steps, n, 4))
    for t in range(steps):
        for i, aid in enumerate(ids):
            p = np.array(sim.getAgentPosition(aid))
            pv = _pref_velocity(p, goals[i])
            sim.setAgentPrefVelocity(aid, (float(pv[0]), float(pv[1])))
            out[t, i, 0], out[t, i, 1] = p
        sim.doStep()
        for i, aid in enumerate(ids):
            out[t, i, 2], out[t, i, 3] = sim.getAgentVelocity(aid)
    return out


def run_pysocialforce(starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
    import pysocialforce as psf  # pyright: ignore[reportMissingImports]
    from pysocialforce import forces as psf_forces  # pyright: ignore[reportMissingImports]

    config = f"""title = "correspondence"
agent_radius = {RADIUS}
step_width = {DT}
max_speed_multiplier = {V_MAX / V_PREF}
tau = {TAU}
resolution = 10
[scene]
enable_group = false
[desired_force]
factor = 1.0
relaxation_time = {TAU}
goal_threshold = {GOAL_RADIUS}
[ped_repulsive_force]
factor = 1.0
v0 = {A_REP}
sigma = {B_REP}
fov_phi = 100.0
fov_factor = {ANISOTROPY}
[obstacle_force]
factor = 0.0
sigma = 0.2
threshold = 3
"""
    n = len(starts)
    state = np.zeros((n, 6))
    for i in range(n):
        v0 = _pref_velocity(starts[i], goals[i])
        state[i] = [starts[i, 0], starts[i, 1], v0[0], v0[1], goals[i, 0], goals[i, 1]]
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(config)
        cfg_path = f.name
    from pysocialforce.potentials import PedPedPotential  # pyright: ignore[reportMissingImports]
    from pysocialforce.utils import stateutils  # pyright: ignore[reportMissingImports]

    class _HelbingRepulsion(psf_forces.PedRepulsiveForce):
        """PedRepulsiveForce with upstream's `desired_directions` tuple bug fixed."""

        def _get_force(self) -> np.ndarray:
            assert self.peds is not None
            pot = PedPedPotential(self.peds.step_width, v0=self.config("v0"), sigma=self.config("sigma"))
            st = self.peds.state
            r_ab = pot.r_ab(st)
            speeds = stateutils.speeds(st)
            e, _ = stateutils.desired_directions(st)
            delta = 1e-3
            v = pot.value_r_ab(r_ab, speeds, e)
            dvdx = (pot.value_r_ab(r_ab + np.array([[[delta, 0.0]]]), speeds, e) - v) / delta
            dvdy = (pot.value_r_ab(r_ab + np.array([[[0.0, delta]]]), speeds, e) - v) / delta
            np.fill_diagonal(dvdx, 0.0)
            np.fill_diagonal(dvdy, 0.0)
            f_ab = -1.0 * np.stack((dvdx, dvdy), axis=-1)
            fov = psf_forces.FieldOfView(phi=self.config("fov_phi"), out_of_view_factor=self.config("fov_factor"))
            w = np.expand_dims(fov(e, -f_ab), -1)
            return np.sum(w * f_ab, axis=1) * self.factor

    sim = psf.Simulator(state, groups=None, obstacles=None, config_file=cfg_path)
    Path(cfg_path).unlink()
    repulsion = _HelbingRepulsion()
    force_list = [psf_forces.DesiredForce(), repulsion]
    for force in force_list:
        force.init(sim, sim.config)
    repulsion.config = sim.config.sub_config("ped_repulsive_force")
    repulsion.factor = repulsion.config("factor", 1.0)
    sim.forces = force_list
    steps = int(HORIZON_S / DT)
    out = np.zeros((steps, n, 4))
    for t in range(steps):
        st = sim.peds.state
        out[t, :, 0:2] = st[:, 0:2]
        out[t, :, 2:4] = st[:, 2:4]
        sim.step_once()
    return out


def run_straight_reference(starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
    n = len(starts)
    steps = int(HORIZON_S / DT)
    out = np.zeros((steps, n, 4))
    pos = starts.astype(float).copy()
    for t in range(steps):
        for i in range(n):
            v = _pref_velocity(pos[i], goals[i])
            out[t, i] = [pos[i, 0], pos[i, 1], v[0], v[1]]
            pos[i] += v * DT
    return out


REFERENCES: dict[str, tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray] | None]] = {
    "orca": ("Python-RVO2", run_rvo2),
    "sfm": ("pysocialforce PedRepulsiveForce", run_pysocialforce),
    "straight": ("closed form", run_straight_reference),
    "hsfm": ("no public reference", None),
    "nsp": ("no step-wise public reference", None),
    "socialgail": ("no public code release", None),
}


def compare(ours: np.ndarray, ref: np.ndarray) -> tuple[list[float], float]:
    """Per-agent Hausdorff distances and the mean per-step velocity difference."""
    n = ours.shape[1]
    hd = [pairwise_hausdorff(ours[:, i, 0:2], ref[:, i, 0:2]) for i in range(n)]
    dv = np.hypot(ours[:, :, 2] - ref[:, :, 2], ours[:, :, 3] - ref[:, :, 3])
    return hd, float(dv.mean())


def run_correspondence(drivers: list[str], out_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    for driver in drivers:
        label, reference = REFERENCES.get(driver, ("no public reference", None))
        if reference is None:
            rows.append({"driver": driver, "reference": label, "scenario": "-", "n_agents": 0, "median_hausdorff_m": np.nan, "max_hausdorff_m": np.nan, "mean_velocity_error_mps": np.nan, "passes": False})
            continue
        for name, make in SCENARIOS.items():
            starts, goals = make()
            ours = run_ours(driver, starts, goals)
            ref = reference(starts, goals)
            hd, dv = compare(ours, ref)
            rows.append(
                {
                    "driver": driver,
                    "reference": label,
                    "scenario": name,
                    "n_agents": len(starts),
                    "median_hausdorff_m": float(np.median(hd)),
                    "max_hausdorff_m": float(np.max(hd)),
                    "mean_velocity_error_mps": dv,
                    "passes": bool(np.median(hd) <= HAUSDORFF_PASS_M and dv <= VEL_PASS_MPS),
                }
            )
    table = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "correspondence.csv", index=False)
        summary = (
            table[table["scenario"] != "-"].groupby(["driver", "reference"]).agg(median_hausdorff_m=("median_hausdorff_m", "median"), max_hausdorff_m=("max_hausdorff_m", "max"), mean_velocity_error_mps=("mean_velocity_error_mps", "mean"), scenarios_passing=("passes", "sum"), scenarios=("passes", "size")).reset_index()
        )
        summary.to_csv(out_dir / "correspondence_summary.csv", index=False)
    return table
