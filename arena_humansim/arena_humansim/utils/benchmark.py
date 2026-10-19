"""Relative benchmark: compare two configurations at various agent counts."""

import argparse
import csv
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Any

import attrs
import numpy as np
import rclpy
import rclpy.executors
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import AddWalls, GetProfile, RemoveAgents, RemoveWalls, SpawnAgents
from geometry_msgs.msg import Point32, Pose2D, Vector3
from rclpy.node import Node


@attrs.define
class PhaseProfile:
    phase_names: list[str] = attrs.Factory(list)
    phase_means_ms: list[float] = attrs.Factory(list)
    phase_p95s_ms: list[float] = attrs.Factory(list)
    n_ticks: int = 0
    n_agents: int = 0


@attrs.define
class RunResult:
    n_agents: int = 0
    dt: float = 0.0
    tick_times_ms: list[float] = attrs.Factory(list)
    profile: PhaseProfile | None = None

    @property
    def median_ms(self) -> float:
        return float(np.median(self.tick_times_ms))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.tick_times_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.tick_times_ms, 99))

    @property
    def rtf(self) -> float:
        if self.profile and self.profile.phase_means_ms:
            total_ms = sum(self.profile.phase_means_ms)
        else:
            total_ms = float(np.mean(self.tick_times_ms))
        return (self.dt * 1000.0) / total_ms if total_ms > 0 else float("inf")


@attrs.define
class IncrementalResult:
    dt: float = 0.0
    spawn_interval: int = 0
    total_ticks: int = 0
    ticks: list[dict] = attrs.Factory(list)

    def by_agent_count(self) -> dict[int, RunResult]:
        groups: dict[int, list[float]] = {}
        for t in self.ticks:
            groups.setdefault(t["n_agents"], []).append(t["tick_ms"])
        return {n: RunResult(n_agents=n, dt=self.dt, tick_times_ms=times) for n, times in sorted(groups.items())}


@attrs.define
class SpawnRect:
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0


@attrs.define
class WallLayout:
    walls: list[list[list[float]]] = attrs.Factory(list)
    spawn_rect: SpawnRect | None = None


class BenchmarkDriver(Node):
    def __init__(self):
        super().__init__("benchmark_driver")
        self._spawn_cli = self.create_client(SpawnAgents, "spawn_agents")
        self._remove_cli = self.create_client(RemoveAgents, "remove_agents")
        self._add_walls_cli = self.create_client(AddWalls, "add_walls")
        self._remove_walls_cli = self.create_client(RemoveWalls, "remove_walls")
        self._get_profile_cli = self.create_client(GetProfile, "get_profile")

        self._tick_times: list[float] = []
        self._last_recv: float | None = None
        self._collecting = False
        self._target_ticks = 0
        self._done_event = threading.Event()

        self._sub = self.create_subscription(
            AgentFrameMsg,
            "agent_states",
            self._on_agent_states,
            10,
        )

    def _on_agent_states(self, msg: AgentFrameMsg):
        now = time.perf_counter()
        if not self._collecting:
            self._last_recv = now
            return
        if self._last_recv is not None:
            self._tick_times.append((now - self._last_recv) * 1000.0)
        self._last_recv = now
        if len(self._tick_times) >= self._target_ticks:
            self._collecting = False
            self._done_event.set()

    def wait_for_services(self, timeout: float = 10.0) -> None:
        for cli in (self._spawn_cli, self._remove_cli, self._add_walls_cli, self._remove_walls_cli, self._get_profile_cli):
            if not cli.wait_for_service(timeout_sec=timeout):
                raise TimeoutError(f"Service {cli.srv_name} not available")

    def add_walls(self, segments: list[list[list[float]]]) -> None:
        req = AddWalls.Request()
        for i, seg in enumerate(segments):
            req.names.append(f"bench_wall_{i}")  # pyright: ignore[reportAttributeAccessIssue]
            req.starts.append(Point32(x=float(seg[0][0]), y=float(seg[0][1]), z=0.0))  # pyright: ignore[reportAttributeAccessIssue]
            req.ends.append(Point32(x=float(seg[1][0]), y=float(seg[1][1]), z=0.0))  # pyright: ignore[reportAttributeAccessIssue]
        resp = self._call_sync(self._add_walls_cli, req, timeout=10.0)
        if resp is None or not resp.success:
            raise RuntimeError(f"add_walls failed: {resp}")

    def remove_walls(self) -> None:
        req = RemoveWalls.Request()
        self._call_sync(self._remove_walls_cli, req, timeout=10.0)

    def get_profile(self, reset: bool = True) -> PhaseProfile:
        req = GetProfile.Request()
        req.reset = reset
        resp = self._call_sync(self._get_profile_cli, req, timeout=10.0)
        return PhaseProfile(
            phase_names=list(resp.phase_names),
            phase_means_ms=list(resp.phase_means_ms),
            phase_p95s_ms=list(resp.phase_p95s_ms),
            n_ticks=resp.n_ticks,
            n_agents=resp.n_agents,
        )

    def spawn_random_agents(self, n: int, seed: int = 42, radius: float = 20.0, spawn_rect: SpawnRect | None = None) -> Iterable[int]:
        rng = np.random.default_rng(seed)
        req = SpawnAgents.Request()
        for _ in range(n):
            if spawn_rect:
                x = rng.uniform(spawn_rect.x_min, spawn_rect.x_max)
                y = rng.uniform(spawn_rect.y_min, spawn_rect.y_max)
                gx = rng.uniform(spawn_rect.x_min, spawn_rect.x_max)
                gy = rng.uniform(spawn_rect.y_min, spawn_rect.y_max)
            else:
                angle = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(0, radius)
                x, y = r * math.cos(angle), r * math.sin(angle)
                goal_angle = rng.uniform(0, 2 * math.pi)
                gx, gy = radius * math.cos(goal_angle), radius * math.sin(goal_angle)
            angle = rng.uniform(0, 2 * math.pi)

            msg = AgentStateMsg()
            msg.agent_id = 0
            msg.pose = Pose2D(x=x, y=y, theta=angle)
            msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
            msg.desired_velocity = float(rng.normal(1.3, 0.15))
            msg.radius = 0.0
            msg.agent_type = "adult"
            wp = WaypointMsg()
            wp.pose = Pose2D(x=gx, y=gy, theta=0.0)
            msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
            req.agents.append(msg)  # pyright: ignore[reportAttributeAccessIssue]

        resp = self._call_sync(self._spawn_cli, req, timeout=30.0)
        if resp is None or not resp.success:
            raise RuntimeError(f"Spawn failed: {resp}")
        return resp.spawned_ids

    def _call_sync(self, client: Any, request: Any, timeout: float = 10.0) -> Any:  # noqa: ANN401
        future = client.call_async(request)
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Service call to {client.srv_name} timed out")
        return future.result()

    def remove_all(self, ids: Iterable[int]) -> None:
        req = RemoveAgents.Request()
        req.agent_ids = list(ids)
        self._call_sync(self._remove_cli, req, timeout=10.0)

    def collect_ticks(self, n_ticks: int, warmup: int = 50) -> list[float]:
        self._tick_times = []
        self._last_recv = None
        self._target_ticks = warmup + n_ticks
        self._done_event.clear()
        self._collecting = True

        timeout = (warmup + n_ticks) * 1.0
        self._done_event.wait(timeout=timeout)

        if len(self._tick_times) < warmup + n_ticks:
            raise RuntimeError(f"Only collected {len(self._tick_times)}/{warmup + n_ticks} ticks")
        return self._tick_times[warmup:]

    def collect_incremental(
        self,
        total_ticks: int,
        spawn_interval: int,
        seed: int = 42,
        radius: float = 20.0,
        warmup: int = 20,
    ) -> tuple[list[int], IncrementalResult]:
        rng = np.random.default_rng(seed)
        all_ids: list[int] = []
        result = IncrementalResult(dt=0.0, spawn_interval=spawn_interval, total_ticks=total_ticks)

        self._tick_times = []
        self._last_recv = None
        self._target_ticks = warmup
        self._done_event.clear()
        self._collecting = True
        self._done_event.wait(timeout=warmup * 1.0)
        self._tick_times = []

        for tick in range(total_ticks):
            if tick > 0 and tick % spawn_interval == 0:
                ids = self._spawn_one_random(rng, radius)
                all_ids.extend(ids)

            self._tick_times = []
            self._last_recv = None
            self._target_ticks = 1
            self._done_event.clear()
            self._collecting = True
            self._done_event.wait(timeout=2.0)

            if self._tick_times:
                result.ticks.append({"tick": tick, "n_agents": len(all_ids), "tick_ms": self._tick_times[0]})

        return all_ids, result

    def _spawn_one_random(self, rng: np.random.Generator, radius: float = 20.0) -> list[int]:
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0, radius)
        x, y = r * math.cos(angle), r * math.sin(angle)
        goal_angle = rng.uniform(0, 2 * math.pi)
        gx, gy = radius * math.cos(goal_angle), radius * math.sin(goal_angle)

        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = Pose2D(x=x, y=y, theta=angle)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = float(rng.normal(1.3, 0.15))
        msg.radius = 0.0
        msg.agent_type = "adult"
        wp = WaypointMsg()
        wp.pose = Pose2D(x=gx, y=gy, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)

        req = SpawnAgents.Request()
        req.agents.append(msg)  # pyright: ignore[reportAttributeAccessIssue]
        resp = self._call_sync(self._spawn_cli, req, timeout=10.0)
        if resp is None or not resp.success:
            raise RuntimeError(f"Spawn failed: {resp}")
        return list(resp.spawned_ids)


def generate_random_walls(n: int, seed: int, radius: float = 20.0) -> WallLayout:
    rng = np.random.default_rng(seed)
    walls = []
    for _ in range(n):
        cx, cy = rng.uniform(-radius * 0.8, radius * 0.8, size=2)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(1.0, 6.0)
        dx, dy = length / 2 * math.cos(angle), length / 2 * math.sin(angle)
        walls.append([[cx - dx, cy - dy], [cx + dx, cy + dy]])
    return WallLayout(walls=walls)


def generate_maze(grid_size: int, seed: int, cell_size: float = 2.0) -> WallLayout:
    rng = np.random.default_rng(seed)
    rows, cols = grid_size, grid_size
    visited = np.zeros((rows, cols), dtype=bool)
    h_walls = np.ones((rows + 1, cols), dtype=bool)
    v_walls = np.ones((rows, cols + 1), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                neighbors.append((nr, nc, dr, dc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc, dr, dc = neighbors[rng.integers(len(neighbors))]
        if dr == -1:
            h_walls[r, c] = False
        elif dr == 1:
            h_walls[r + 1, c] = False
        elif dc == -1:
            v_walls[r, c] = False
        elif dc == 1:
            v_walls[r, c + 1] = False
        visited[nr, nc] = True
        stack.append((nr, nc))

    walls = []
    for r in range(rows + 1):
        for c in range(cols):
            if h_walls[r, c]:
                x0, x1 = c * cell_size, (c + 1) * cell_size
                y = r * cell_size
                walls.append([[x0, y], [x1, y]])
    for r in range(rows):
        for c in range(cols + 1):
            if v_walls[r, c]:
                y0, y1 = r * cell_size, (r + 1) * cell_size
                x = c * cell_size
                walls.append([[x, y0], [x, y1]])

    margin = cell_size * 0.3
    rect = SpawnRect(
        x_min=margin,
        x_max=cols * cell_size - margin,
        y_min=margin,
        y_max=rows * cell_size - margin,
    )
    return WallLayout(walls=walls, spawn_rect=rect)


def _merge_params_file(params_file: str, extra_params: dict[str, Any], dt: float) -> str:
    with open(params_file) as f:
        data = yaml.safe_load(f) or {}
    node_key = next(iter(data), "arena_humansim")
    ros_params = data.setdefault(node_key, {}).setdefault("ros__parameters", {})
    ros_params["mode"] = "master"
    ros_params["rtf"] = 0.0
    ros_params["publish_markers"] = 0
    ros_params["dt"] = dt
    ros_params["profile_phases"] = True
    ros_params.update(extra_params)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(data, tmp)
    tmp.close()
    return tmp.name


def launch_humansim(
    params_file: str | None,
    extra_params: dict[str, Any],
    dt: float,
    profile: bool = False,
    profile_interval: int = 100,
) -> subprocess.Popen:
    cmd = [
        "ros2",
        "run",
        "arena_humansim",
        "arena_humansim_node",
    ]
    if profile:
        cmd.extend(["--profile", "--profile-interval", str(profile_interval)])
    cmd.append("--ros-args")
    if params_file:
        merged = _merge_params_file(params_file, extra_params, dt)
        cmd.extend(["--params-file", merged])
    else:
        cmd.extend(["-p", "mode:=master", "-p", "rtf:=0.0", "-p", "publish_markers:=0", "-p", f"dt:={dt}", "-p", "profile_phases:=true"])
        for k, v in extra_params.items():
            cmd.extend(["-p", f"{k}:={v}"])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL if not profile else None,
        stderr=subprocess.PIPE if not profile else None,
        start_new_session=True,
    )
    return proc


def kill_proc(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()


def run_single(
    driver: BenchmarkDriver,
    params_file: str | None,
    extra_params: dict[str, Any],
    n_agents: int,
    dt: float,
    n_ticks: int,
    warmup: int,
    seed: int,
    executor: rclpy.executors.Executor,
    walls: list[list[list[float]]] | None = None,
    spawn_rect: SpawnRect | None = None,
    profile: bool = False,
    profile_interval: int = 100,
) -> RunResult:
    proc = launch_humansim(params_file, extra_params, dt, profile=profile, profile_interval=profile_interval)
    try:
        time.sleep(1.5)
        driver.wait_for_services(timeout=10.0)
        if walls:
            driver.add_walls(walls)
        ids = driver.spawn_random_agents(n_agents, seed=seed, spawn_rect=spawn_rect)
        tick_times = driver.collect_ticks(n_ticks, warmup=warmup)
        profile_data = driver.get_profile(reset=True)
        result = RunResult(n_agents=n_agents, dt=dt, tick_times_ms=tick_times, profile=profile_data)
        driver.remove_all(ids)
    finally:
        kill_proc(proc)
    time.sleep(0.5)
    return result


def run_incremental(
    driver: BenchmarkDriver,
    params_file: str | None,
    extra_params: dict[str, Any],
    dt: float,
    total_ticks: int,
    spawn_interval: int,
    warmup: int,
    seed: int,
    executor: rclpy.executors.Executor,
    profile: bool = False,
    profile_interval: int = 100,
) -> IncrementalResult:
    proc = launch_humansim(params_file, extra_params, dt, profile=profile, profile_interval=profile_interval)
    try:
        time.sleep(1.5)
        driver.wait_for_services(timeout=10.0)
        ids, result = driver.collect_incremental(
            total_ticks=total_ticks,
            spawn_interval=spawn_interval,
            seed=seed,
            warmup=warmup,
        )
        result.dt = dt
        driver.remove_all(ids)
    finally:
        kill_proc(proc)
    time.sleep(0.5)
    return result


def print_incremental_results(label: str, result: IncrementalResult):
    by_count = result.by_agent_count()
    if not by_count:
        print(f"  {label}: no data collected")
        return

    def ms(v: float) -> str:
        return f"{v:.2f}ms"

    W = 10
    counts = list(by_count.keys())
    header = f"{'agents':>12}" + "".join(f"{n:>{W}}" for n in counts)

    print(f"  {label} (incremental: 1 agent every {result.spawn_interval} ticks, {result.total_ticks} total)")
    print(header)
    print(f"{'median':>12}" + "".join(f"{ms(r.median_ms):>{W}}" for r in by_count.values()))
    print(f"{'p95':>12}" + "".join(f"{ms(r.p95_ms):>{W}}" for r in by_count.values()))
    print(f"{'p99':>12}" + "".join(f"{ms(r.p99_ms):>{W}}" for r in by_count.values()))
    print(f"{'rtf':>12}" + "".join(f"{r.rtf:>{W}.1f}" for r in by_count.values()))
    print(f"{'samples':>12}" + "".join(f"{len(r.tick_times_ms):>{W}}" for r in by_count.values()))
    print()


def best_round(round_results: Sequence[RunResult]) -> RunResult:
    return min(round_results, key=lambda r: np.mean(r.tick_times_ms))


@attrs.define
class Stage:
    agents: list[int] = attrs.Factory(list)
    walls: list[list[list[float]]] | None = None
    spawn_rect: SpawnRect | None = None
    label: str = ""

    @property
    def wall_count(self) -> int:
        return len(self.walls) if self.walls else 0


@attrs.define
class StageResults:
    stage: Stage = attrs.Factory(Stage)
    cand_results: list[RunResult] = attrs.Factory(list)
    ref_results: list[RunResult] = attrs.Factory(list)


def _col_label(n_agents: int, wall_count: int) -> str:
    if wall_count:
        return f"{n_agents}[{wall_count}w]"
    return str(n_agents)


def print_results(
    cand_label: str,
    ref_label: str,
    all_stages: Sequence[StageResults],
):
    def ms(v: float) -> str:
        return f"{v:.2f}ms"

    W = 12
    PW = 16
    col_labels = []
    all_cand = []
    all_ref = []
    for sr in all_stages:
        wc = sr.stage.wall_count
        for cr, rr in zip(sr.cand_results, sr.ref_results, strict=True):
            col_labels.append(_col_label(cr.n_agents, wc))
            all_cand.append(cr)
            all_ref.append(rr)

    header = f"{'':>{PW}}" + "".join(f"{c:>{W}}" for c in col_labels)

    phases = set()
    for r in all_cand + all_ref:
        if r.profile:
            phases.update(r.profile.phase_names)
    if phases:
        print("  phase breakdown (proportion of tick)")
        print(header)
        for label, results in [(cand_label, all_cand), (ref_label, all_ref)]:
            print(f"  {label}")
            for phase in sorted(phases):
                vals = []
                for r in results:
                    if r.profile and phase in r.profile.phase_names:
                        idx = r.profile.phase_names.index(phase)
                        total = sum(r.profile.phase_means_ms)
                        prop = r.profile.phase_means_ms[idx] / total if total > 0 else 0
                        vals.append(f"{prop:.4f}")
                    else:
                        vals.append("-")
                print(f"{phase:>{PW}}" + "".join(f"{v:>{W}}" for v in vals))
            print(f"{'TOTAL':>{PW}}" + "".join(f"{'1.0000':>{W}}" for _ in results))
            print()

    for label, results in [(cand_label, all_cand), (ref_label, all_ref)]:
        print()
        print(f"  {label}")
        print(header)
        print(f"{'median':>{PW}}" + "".join(f"{ms(r.median_ms):>{W}}" for r in results))
        print(f"{'p95':>{PW}}" + "".join(f"{ms(r.p95_ms):>{W}}" for r in results))
        print(f"{'p99':>{PW}}" + "".join(f"{ms(r.p99_ms):>{W}}" for r in results))
        print(f"{'rtf':>{PW}}" + "".join(f"{r.rtf:>{W}.1f}" for r in results))

    print()
    print("  comparison (candidate vs reference)")
    print(header)

    def _total_ms(r: RunResult) -> float:
        if r.profile and r.profile.phase_means_ms:
            return sum(r.profile.phase_means_ms)
        return float(np.mean(r.tick_times_ms))

    ratios = []
    speedups = []
    for rc, rr in zip(all_cand, all_ref, strict=True):
        ct, rt = _total_ms(rc), _total_ms(rr)
        ratios.append(ct / rt if rt > 0 else float("inf"))
        speedups.append((rt - ct) / rt * 100 if rt > 0 else 0)
    print(f"{'ratio':>{PW}}" + "".join(f"{r:>{W}.3f}" for r in ratios))

    def _speedup_label(s: float) -> str:
        sign = "+" if s >= 0 else ""
        return f"{sign}{s:.1f}%"

    print(f"{'speedup':>{PW}}" + "".join(f"{_speedup_label(s):>{W}}" for s in speedups))
    print()


def parse_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    params = {}
    for pair in raw.split(","):
        k, v = pair.split("=", 1)
        params[k.strip()] = v.strip()
    return params


def _parse_walls_spec(spec: int | str | list[list[list[float]]] | None, seed: int, stage_idx: int) -> WallLayout | None:
    if spec is None:
        return None
    if isinstance(spec, int):
        return generate_random_walls(spec, seed=seed + stage_idx)
    if isinstance(spec, str):
        import re

        m = re.match(r"maze\((\d+)\)", spec)
        if m:
            return generate_maze(int(m.group(1)), seed=seed + stage_idx)
        raise ValueError(f"Unknown walls spec: {spec!r}")
    if isinstance(spec, list):
        return WallLayout(walls=spec)
    raise ValueError(f"Unknown walls spec type: {type(spec)}")


def parse_stages(cfg: dict[str, Any], seed: int) -> list[Stage]:
    if "stages" in cfg:
        stages = []
        for i, s in enumerate(cfg["stages"]):
            agents = s.get("agents", [10])
            if isinstance(agents, int):
                agents = [agents]
            layout = _parse_walls_spec(s.get("walls"), seed, i)
            walls = layout.walls if layout else None
            spawn_rect = layout.spawn_rect if layout else None
            if walls:
                default_label = f"stage {i} ({len(walls)} walls)"
            else:
                default_label = f"stage {i}"
            label = s.get("label", default_label)
            stages.append(Stage(agents=agents, walls=walls, spawn_rect=spawn_rect, label=label))
        return stages
    agents = cfg.get("agents", [10, 25, 50, 100, 200])
    return [Stage(agents=agents)]


def load_benchmark_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config_path(path: str | None) -> str | None:
    if path is None:
        return None
    if path.startswith("pkg://"):
        rest = path[len("pkg://") :]
        pkg, _, rel = rest.partition("/")
        return get_package_share_directory(pkg) + "/" + rel
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Relative benchmark: compare two arena_humansim configurations",
    )
    parser.add_argument("config", nargs="?", default=None, metavar="BENCHMARK_YAML", help="Benchmark config YAML (default: config/benchmark/default.yaml)")
    parser.add_argument("--ref", default=None, metavar="FILE", help="Reference params YAML file")
    parser.add_argument("--candidate", default=None, metavar="FILE", help="Candidate params YAML file")
    parser.add_argument("--ref-params", default=None, help="Extra reference params as key=val,key=val")
    parser.add_argument("--candidate-params", default=None, help="Extra candidate params as key=val,key=val")
    parser.add_argument("--ref-label", default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--agents", default=None, help="Comma-separated agent counts")
    parser.add_argument("--dt", type=float, default=None, help="Simulation timestep in seconds (default: 0.05)")
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None, help="Rounds per (config, agent_count) pair (default: 3)")
    parser.add_argument("--incremental", action="store_true", help="Spawn 1 agent every --spawn-interval ticks for --ticks total")
    parser.add_argument("--spawn-interval", type=int, default=None, help="Ticks between spawns in incremental mode (default: 5)")
    parser.add_argument("--profile", action="store_true", help="Enable per-phase tick profiling inside the sim node")
    parser.add_argument("--profile-interval", type=int, default=0, help="Ticks between profile log dumps (0 = flush at shutdown only)")
    parser.add_argument("--csv", default=None, metavar="FILE", help="Save raw per-tick data to CSV file")
    args = parser.parse_args()

    cfg = {}
    config_path = args.config
    if config_path is None and args.ref is None and args.candidate is None:
        pkg_share = get_package_share_directory("arena_humansim")
        config_path = pkg_share + "/config/benchmark/default.yaml"
    elif config_path is not None and "/" not in config_path and not config_path.endswith(".yaml"):
        pkg_share = get_package_share_directory("arena_humansim")
        config_path = pkg_share + "/config/benchmark/" + config_path + ".yaml"
    if config_path is not None:
        cfg = load_benchmark_config(config_path)

    ref_label = args.ref_label or cfg.get("reference", {}).get("label", "reference")
    cand_label = args.candidate_label or cfg.get("candidate", {}).get("label", "candidate")
    ref_file = resolve_config_path(args.ref or cfg.get("reference", {}).get("params_file"))
    cand_file = resolve_config_path(args.candidate or cfg.get("candidate", {}).get("params_file"))
    ref_extra = parse_params(args.ref_params) or cfg.get("reference", {}).get("params", {})
    cand_extra = parse_params(args.candidate_params) or cfg.get("candidate", {}).get("params", {})
    dt = args.dt or cfg.get("dt", 0.05)
    n_ticks = args.ticks or cfg.get("ticks", 200)
    warmup = args.warmup or cfg.get("warmup", 50)
    rounds = args.rounds or cfg.get("rounds", 3)
    spawn_interval = args.spawn_interval or cfg.get("spawn_interval", 5)
    cfg_seed = cfg.get("seed")
    if args.seed is not None:
        seed = args.seed
    elif cfg_seed is not None:
        seed = cfg_seed
    else:
        seed = int.from_bytes(os.urandom(4), "little")

    stages = parse_stages(cfg, seed=seed)
    if args.agents:
        agent_counts = [int(x) for x in args.agents.split(",")]
        stages = [Stage(agents=agent_counts, walls=stages[0].walls if stages else None)]

    print(f"Benchmark: {cand_label} vs {ref_label}")
    print(f"  stages: {len(stages)}  dt: {dt}  ticks: {n_ticks}  warmup: {warmup}")
    print(f"  rounds: {rounds}  seed: {seed}")
    print()

    rclpy.init()
    driver = BenchmarkDriver()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(driver)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if args.incremental:
            max_agents = n_ticks // spawn_interval
            print(f"Incremental mode: 1 agent every {spawn_interval} ticks, {n_ticks} total ticks, up to {max_agents} agents")
            print()
            for label, params_file, extra in [
                (cand_label, cand_file, cand_extra),
                (ref_label, ref_file, ref_extra),
            ]:
                print(f"{label}: running incremental...", flush=True)
                result = run_incremental(
                    driver,
                    params_file,
                    extra,
                    dt,
                    n_ticks,
                    spawn_interval,
                    warmup,
                    seed,
                    executor,
                    profile=args.profile,
                    profile_interval=args.profile_interval,
                )
                print_incremental_results(label, result)
            executor.shutdown()
            driver.destroy_node()
            rclpy.shutdown()
            return

        all_stage_results = []
        csv_file = None
        csv_writer = None
        if args.csv:
            csv_file = open(args.csv, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["config", "agents", "walls", "round", "tick", "tick_ms"])
        configs = [
            (cand_label, cand_file, cand_extra),
            (ref_label, ref_file, ref_extra),
        ]
        for stage in stages:
            if len(stages) > 1:
                print(f"--- {stage.label} ---")
            cand_results = []
            ref_results = []
            wc = stage.wall_count
            for n_agents in stage.agents:
                cand_rounds = []
                ref_rounds = []
                round_lists = [cand_rounds, ref_rounds]
                for r in range(rounds):
                    for (label, params_file, extra), round_list in zip(configs, round_lists, strict=True):
                        print(f"{label}: {n_agents} agents [{r + 1}/{rounds}] running...", end="", flush=True)
                        result = run_single(
                            driver,
                            params_file,
                            extra,
                            n_agents,
                            dt,
                            n_ticks,
                            warmup,
                            seed,
                            executor,
                            walls=stage.walls,
                            spawn_rect=stage.spawn_rect,
                            profile=args.profile,
                            profile_interval=args.profile_interval,
                        )
                        round_list.append(result)
                        if csv_writer is not None and csv_file is not None:
                            for tick_i, tick_ms in enumerate(result.tick_times_ms):
                                csv_writer.writerow((label, n_agents, wc, r + 1, tick_i, tick_ms))
                            csv_file.flush()
                        print(
                            f"\r{label}: {n_agents} agents [{r + 1}/{rounds}] median={result.median_ms:.2f}ms  p95={result.p95_ms:.2f}ms  rtf={result.rtf:.2f}",
                            flush=True,
                        )
                cand_results.append(best_round(cand_rounds))
                ref_results.append(best_round(ref_rounds))
            all_stage_results.append(StageResults(stage=stage, cand_results=cand_results, ref_results=ref_results))
        print_results(cand_label, ref_label, all_stage_results)
        if csv_file:
            csv_file.close()
            print(f"Raw data saved to {args.csv}")
    finally:
        executor.shutdown()
        driver.destroy_node()
        rclpy.shutdown()
