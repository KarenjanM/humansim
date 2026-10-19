from __future__ import annotations

from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim.utils.types import AgentLifetime
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import GetProfile, SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3
from rclpy.parameter import Parameter


def _spawn_one(mgr: AgentManager, x: float = 0.0, y: float = 0.0, gx: float = 5.0, gy: float = 0.0) -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = RosPose2D(x=x, y=y, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.0
    msg.agent_type = "adult"
    wp = WaypointMsg()
    wp.pose = RosPose2D(x=gx, y=gy, theta=0.0)
    msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return list(resp.spawned_ids)[0]


def test_tick_100_ticks_no_errors(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick")
    for _ in range(100):
        mgr.tick()
    assert mgr._tick_count == 100


def test_tick_noop_collision_beside_noop_animation(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    assert minimal_scenario.modules.animation == "noop"
    mgr = manager_factory(minimal_scenario, node_name="test_tick_noop_collision", extra_params=[Parameter("collision", Parameter.Type.STRING, "noop")])
    _spawn_one(mgr)
    for _ in range(3):
        mgr.tick()
    assert mgr._tick_count == 3


def test_tick_agent_count_stays_zero_with_no_spawns(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick_empty")
    for _ in range(20):
        mgr.tick()
        assert len(mgr._agents) == 0
        assert mgr._pool.n == 0


def test_tick_phase_profiling_populates(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick_profile")
    for _ in range(5):
        mgr.tick()
    assert mgr._tick_phases, "expected _tick_phases dict populated after tick"
    for name in ("despawn", "spawn", "sense", "local_plan", "kinematics", "animation", "integrate", "collision", "publish"):
        assert name in mgr._tick_phases, f"missing phase {name!r}"
        assert mgr._tick_phases[name] >= 0.0
    assert mgr._phase_accum, "expected _phase_accum populated when profile_phases=True"


def test_tick_phase_accum_advances_per_tick(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick_phase_accum")
    n_ticks = 7
    for _ in range(n_ticks):
        mgr.tick()
    assert mgr._profile_interval == 0, "precondition: interval=0 so accum is not flushed"
    every_tick_phases = ("despawn", "spawn", "sense", "local_plan", "kinematics", "animation", "integrate", "collision", "publish")
    for name in every_tick_phases:
        times = mgr._phase_accum[name]
        assert len(times) == n_ticks, f"phase {name!r} accum len={len(times)} expected {n_ticks}"
    bt_gated_phases = ("decide", "global_plan")
    expected_bt = sum(1 for t in range(n_ticks) if t % mgr._bt_tick_interval == 0)
    for name in bt_gated_phases:
        times = mgr._phase_accum[name]
        assert len(times) == expected_bt, f"phase {name!r} accum len={len(times)} expected {expected_bt}"


def test_tick_count_monotonic(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick_tc_mono")
    prev = mgr._tick_count
    for i in range(1, 25):
        mgr.tick()
        assert mgr._tick_count == prev + 1, f"tick_count did not advance at iter {i}"
        prev = mgr._tick_count


def test_tick_last_spawned_ids_clears_with_no_spawn(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick_spawn_ids")
    for _ in range(3):
        mgr.tick()
        assert mgr._last_spawned_ids == []
    _spawn_one(mgr)
    assert mgr._last_spawned_ids == [], "direct callback spawn does not populate _last_spawned_ids (only scheduler-driven spawns do)"
    mgr.tick()
    assert mgr._last_spawned_ids == [], "no scheduler source -> list stays empty on next tick"


def test_tick_last_despawned_ids_populates_on_ttl(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick_despawn_ids")
    aid = _spawn_one(mgr)
    mgr._despawn_monitor.register(aid, AgentLifetime(agent_id=aid, spawn_tick=mgr._tick_count - 1, max_lifetime_s=mgr._dt / 2))
    mgr.tick()
    assert aid in mgr._last_despawned_ids
    mgr.tick()
    assert mgr._last_despawned_ids == [], "list resets at top of next tick"


def test_tick_get_profile_callback_shape(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_tick_profile_cb")
    for _ in range(4):
        mgr.tick()
    req = GetProfile.Request()
    req.reset = False
    resp = GetProfile.Response()
    out = mgr._get_profile_callback(req, resp)
    assert out is resp
    assert len(resp.phase_names) > 0
    assert len(resp.phase_names) == len(resp.phase_means_ms) == len(resp.phase_p95s_ms)
    for m in resp.phase_means_ms:
        assert m >= 0.0
    for p in resp.phase_p95s_ms:
        assert p >= 0.0
    assert resp.n_agents == mgr._pool.n
    assert resp.n_ticks >= 1
    assert mgr._phase_accum, "reset=False should not clear accum"

    req.reset = True
    mgr._get_profile_callback(req, GetProfile.Response())
    assert mgr._phase_accum == {}
