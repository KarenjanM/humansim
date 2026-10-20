from __future__ import annotations

from collections.abc import Callable

import attrs
import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT
from arena_humansim.utils.scenario import (
    AgentTemplateModel,
    FlowScenarioConfig,
    ModuleConfig,
    RateKeyframeModel,
    ScenarioConfig,
    ShapeModel,
    SimulationParams,
    SinkAffinityModel,
    SinkScenarioConfig,
    SourceScenarioConfig,
)
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from rclpy.executors import SingleThreadedExecutor

from ._capture import LATCHED, settle, spin_until, subscribe

TICKS = 300
ROBOT_Y = 200.0
EXTERNAL_ROBOT_ID = 0x2345678

Population = dict[int, tuple[int, float, float, float]]
Tracks = list[dict[int, tuple[float, float, float]]]
Attribution = dict[int, tuple[str, int]]
Releases = dict[tuple[str, int], tuple[int, bool, float, float, float]]


def _source(x: float, sink_idx: int, max_concurrent: int) -> SourceScenarioConfig:
    return SourceScenarioConfig(
        pose=Pose2D(x=x, y=0.0, theta=0.0),
        shape=ShapeModel(type="circle", radius=0.8),
        rate_profile=[RateKeyframeModel(t=0.0, rate=1.5)],
        max_concurrent=max_concurrent,
        agent_template=AgentTemplateModel(desired_velocity_min=1.0, desired_velocity_max=1.3, agent_radius=0.25, sink_affinity=[SinkAffinityModel(sink_idx=sink_idx, weight=1.0)]),
    )


def _crossing(local_planner: str, west_cap: int = 3) -> ScenarioConfig:
    sink = ShapeModel(type="circle", radius=1.0)
    return ScenarioConfig(
        name="robot_counterfactual",
        simulation=SimulationParams(seed=5, dt=0.05, max_ticks=TICKS),
        modules=ModuleConfig(local_planner=local_planner),
        flow=FlowScenarioConfig(
            sources=[_source(-6.0, 0, west_cap), _source(6.0, 1, -1)],
            sinks=[
                SinkScenarioConfig(pose=Pose2D(x=7.0, y=0.0, theta=0.0), shape=sink, absorption_radius=1.0),
                SinkScenarioConfig(pose=Pose2D(x=-7.0, y=0.0, theta=0.0), shape=sink, absorption_radius=1.0),
            ],
        ),
    )


def _loop(cx: float, cy: float) -> WaypointsMsg:
    corners = ((cx, cy), (cx + 2.0, cy), (cx + 1.0, cy + 2.0))
    return WaypointsMsg(points=[WaypointMsg(pose=RosPose2D(x=x, y=y, theta=0.0)) for x, y in corners], mode=WaypointsMsg.MODE_RANDOM)


def _agent_msg(agent_id: int, kind: int, x: float, y: float) -> AgentStateMsg:
    msg = AgentStateMsg()
    msg.agent_id = agent_id
    msg.kind = kind
    msg.pose = RosPose2D(x=x, y=y, theta=0.0)
    msg.radius = 0.3
    msg.desired_velocity = 1.0
    msg.agent_type = "adult"
    return msg


def _spawn(mgr: AgentManager, robot: str) -> None:
    req = SpawnAgents.Request()
    if robot == "internal":
        msg = _agent_msg(0, KIND_ROBOT, 0.0, ROBOT_Y)
        msg.policy = "straight"
        msg.waypoints = _loop(0.0, ROBOT_Y)
        req.agents.append(msg)
    walker = _agent_msg(0, KIND_HUMAN, 0.0, 10.0)
    walker.waypoints = _loop(0.0, 10.0)
    req.agents.append(walker)
    assert mgr._spawn_agents_callback(req, SpawnAgents.Response()).success


def _feed_external_robot(mgr: AgentManager) -> None:
    states = AgentStatesMsg()
    states.header.stamp.sec = mgr._tick_count // 20
    states.header.stamp.nanosec = (mgr._tick_count % 20) * 50_000_000
    msg = _agent_msg(EXTERNAL_ROBOT_ID, KIND_ROBOT, -10.0 + 0.05 * mgr._tick_count, ROBOT_Y)
    msg.name = "robot"
    states.agents.append(msg)
    mgr._world_state_callback(states)


def _run(mgr: AgentManager, robot: str) -> tuple[Population, Tracks]:
    _spawn(mgr, robot)
    population: Population = {}
    tracks: Tracks = []
    for _ in range(TICKS):
        if robot == "external":
            _feed_external_robot(mgr)
        mgr.tick()
        for aid in mgr._last_spawned_ids:
            params = mgr._agents[aid].params
            population[aid] = (mgr._tick_count, float(params.desired_velocity), float(params.agent_radius), float(params.max_velocity))
        pool = mgr._pool
        tracks.append({int(pool.agent_ids[i]): (float(pool.pos[i, 0]), float(pool.pos[i, 1]), float(pool.theta[i])) for i in range(pool.n) if pool.kind[i] == KIND_HUMAN})
    return population, tracks


@pytest.mark.parametrize("local_planner", ["sfm", "orca"])
@pytest.mark.parametrize("robot", ["internal", "external"])
def test_far_robot_leaves_pedestrian_spawns_and_tracks_unchanged(manager_factory: Callable[..., AgentManager], local_planner: str, robot: str) -> None:
    scenario = _crossing(local_planner)
    alone = _run(manager_factory(scenario, node_name=f"cf_alone_{robot}_{local_planner}"), "none")
    mgr = manager_factory(scenario, node_name=f"cf_robot_{robot}_{local_planner}")
    with_robot = _run(mgr, robot)
    robots = [int(mgr._pool.agent_ids[i]) for i in range(mgr._pool.n) if mgr._pool.kind[i] == KIND_ROBOT]
    assert robots == [-1]
    assert len(alone[0]) > 10
    assert with_robot[0] == alone[0]
    for tick, (a, b) in enumerate(zip(with_robot[1], alone[1], strict=True)):
        assert a == b, f"tick {tick}: pedestrian tracks differ"


def test_capped_source_leaves_other_source_draws_unchanged(manager_factory: Callable[..., AgentManager]) -> None:
    released = {}
    for west_cap in (3, 1):
        mgr = manager_factory(_crossing("sfm", west_cap), node_name=f"cf_cap_{west_cap}")
        east = []
        for _ in range(TICKS):
            mgr.tick()
            for aid in mgr._last_spawned_ids:
                agent = mgr._agents[aid]
                if agent.state.pose.x > 0.0:
                    east.append((mgr._tick_count, float(agent.params.desired_velocity), float(agent.params.agent_radius), float(agent.params.max_velocity)))
        released[west_cap] = east
    assert len(released[3]) > 10
    assert released[1] == released[3]


def test_robot_ids_count_down_outside_pedestrian_ids(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(attrs.evolve(_crossing("sfm"), flow=None), node_name="cf_robot_ids")
    req = SpawnAgents.Request()
    req.agents.extend([_agent_msg(-1, KIND_ROBOT, 0.0, 0.0), _agent_msg(0, KIND_ROBOT, 2.0, 0.0), _agent_msg(0, KIND_ROBOT, 4.0, 0.0), _agent_msg(0, KIND_HUMAN, 6.0, 0.0)])
    resp = mgr._spawn_agents_callback(req, SpawnAgents.Response())
    assert list(resp.spawned_ids) == [-1, -2, -3, 1]


def _published_releases(mgr: AgentManager, robot: str) -> tuple[Attribution, Releases]:
    """Source and release index each agent carries on the published agent_meta, and the spawn of every released agent by that pair."""
    metas = subscribe(mgr, AgentMetaMsg, "agent_meta", LATCHED)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)
    _spawn(mgr, robot)
    attribution: Attribution = {}
    releases: Releases = {}
    for _ in range(TICKS):
        published = mgr._meta_published
        mgr.tick()
        if mgr._meta_published is not published:
            count = len(metas) + 1
            spin_until(executor, lambda count=count: len(metas) == count)
            meta = metas[-1]
            for aid, name, seq in zip(meta.agent_id, meta.source_name, meta.source_seq, strict=True):
                assert attribution.setdefault(int(aid), (name, int(seq))) == (name, int(seq))
        for aid in mgr._last_spawned_ids:
            agent = mgr._agents[aid]
            if attribution[aid][0]:
                releases[attribution[aid]] = (mgr._tick_count, agent.state.pose.x > 0.0, float(agent.params.desired_velocity), float(agent.params.agent_radius), float(agent.params.max_velocity))
    executor.remove_node(mgr)
    mgr.destroy_node()
    return attribution, releases


def test_agent_meta_pairs_released_agents_with_source_and_release_index(manager_factory: Callable[..., AgentManager]) -> None:
    attribution, releases = _published_releases(manager_factory(_crossing("sfm"), node_name="attr_robot"), "internal")
    scripted = {aid: pair for aid, pair in attribution.items() if pair[0] == ""}
    assert scripted == {-1: ("", 0), 1: ("", 0)}
    for source, east in (("source_0", False), ("source_1", True)):
        spawns = {seq: spawn for (name, seq), spawn in releases.items() if name == source}
        assert len(spawns) >= 3
        assert sorted(spawns) == list(range(len(spawns)))
        assert {spawn[1] for spawn in spawns.values()} == {east}


def test_far_robot_keeps_every_source_release_pair(manager_factory: Callable[..., AgentManager]) -> None:
    _, alone = _published_releases(manager_factory(_crossing("sfm"), node_name="attr_alone"), "none")
    _, with_robot = _published_releases(manager_factory(_crossing("sfm"), node_name="attr_far_robot"), "internal")
    assert len(alone) > 10
    assert with_robot == alone


def test_capped_source_keeps_the_other_source_pairs_under_new_ids(manager_factory: Callable[..., AgentManager]) -> None:
    east: dict[int, Releases] = {}
    ids: dict[int, list[int]] = {}
    for west_cap in (3, 1):
        attribution, releases = _published_releases(manager_factory(_crossing("sfm", west_cap), node_name=f"attr_cap_{west_cap}"), "none")
        east[west_cap] = {pair: spawn for pair, spawn in releases.items() if pair[0] == "source_1"}
        ids[west_cap] = sorted(aid for aid, pair in attribution.items() if pair[0] == "source_1")
    assert len(east[3]) > 10
    assert east[1] == east[3]
    assert ids[1] != ids[3]
