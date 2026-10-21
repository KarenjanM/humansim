from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import attrs
import py_trees
import pytest
import yaml
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.elevators import WAIT_RADIUS, Phase
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.scenario import (
    AgentConfig,
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
    WallConfig,
    load_scenario,
)
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import Elevator
from arena_humansim_msgs.srv import ResetSimulation, SetElevatorDoors, SetElevators, TeleportAgents
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.parameter import Parameter

ROOM = 6.0
ROOM_B_X = 20.0
SINK = Pose2D(x=ROOM_B_X + 1.0, y=3.0)
PLANNERS = {
    "navmesh": [],
    "dijkstra": [Parameter("global_planner.resolution", Parameter.Type.DOUBLE, 0.1)],
}


def _room(name: str, x0: float) -> list[WallConfig]:
    corners = [(x0, 0.0), (x0 + ROOM, 0.0), (x0 + ROOM, ROOM), (x0, ROOM)]
    return [WallConfig(name=f"{name}{i}", start=Pose2D(x=ax, y=ay), end=Pose2D(x=bx, y=by)) for i, ((ax, ay), (bx, by)) in enumerate(zip(corners, corners[1:] + corners[:1], strict=True))]


def _scenario(name: str, planner: str, riders: int = 1) -> ScenarioConfig:
    flow = FlowScenarioConfig(
        sources=[
            SourceScenarioConfig(
                pose=Pose2D(x=1.0, y=3.0),
                shape=ShapeModel(type="circle", radius=0.1),
                type="poisson",
                rate_profile=[RateKeyframeModel(t=0.0, rate=20.0)],
                max_concurrent=riders,
                max_total=riders,
                agent_template=AgentTemplateModel(desired_velocity_min=1.2, desired_velocity_max=1.2, agent_radius=0.25, sink_affinity=[SinkAffinityModel(sink_idx=0, weight=1.0)]),
            ),
        ],
        sinks=[SinkScenarioConfig(pose=SINK, shape=ShapeModel(type="circle", radius=0.5), absorption_radius=0.5)],
    )
    return ScenarioConfig(
        name=name,
        simulation=SimulationParams(seed=3, dt=0.05),
        modules=ModuleConfig(global_planner=planner),
        walls=_room("a", 0.0) + _room("b", ROOM_B_X),
        flow=flow,
    )


def _cabin(name: str, destination: str, x0: float, y: float, door_open: bool = False, half: float = 0.6) -> Elevator:
    e = Elevator()
    e.name = name
    e.destination = destination
    e.boarding = Pose2DMsg(x=x0 + 4.6 + half, y=y, theta=0.0)
    e.doorway = Pose2DMsg(x=x0 + 4.0, y=y, theta=0.0)
    e.queue = Pose2DMsg(x=x0 + 3.4, y=y, theta=0.0)
    e.footprint = [Point32(x=x0 + px, y=y + py, z=0.0) for px, py in ((4.6, -half), (4.6 + 2 * half, -half), (4.6 + 2 * half, half), (4.6, half))]
    e.door_open = door_open
    return e


def _set_elevators(mgr: AgentManager, cabins: list[Elevator]) -> SetElevators.Response:
    return mgr._set_elevators_callback(SetElevators.Request(elevators=cabins), SetElevators.Response())


def _set_door(mgr: AgentManager, name: str, door_open: bool) -> SetElevatorDoors.Response:
    return mgr._set_elevator_doors_callback(SetElevatorDoors.Request(name=[name], open=[door_open]), SetElevatorDoors.Response())


def _teleport(mgr: AgentManager, ids: list[int], poses: list[tuple[float, float]]) -> TeleportAgents.Response:
    request = TeleportAgents.Request(agent_id=ids, pose=[Pose2DMsg(x=x, y=y, theta=0.0) for x, y in poses])
    return mgr._teleport_agents_callback(request, TeleportAgents.Response())


def _tick_until(mgr: AgentManager, done: Callable[[], bool], limit: int) -> int:
    for n in range(limit):
        if done():
            return n
        mgr.tick()
    raise AssertionError(f"condition not met in {limit} ticks")


def _rider(mgr: AgentManager) -> int:
    _tick_until(mgr, lambda: bool(mgr._agents), 40)
    return next(iter(mgr._agents))


def _ride_cabins(name: str) -> list[Elevator]:
    return [_cabin(f"{name}_a", f"{name}_b", 0.0, 4.6), _cabin(f"{name}_b", f"{name}_a", ROOM_B_X, 4.6)]


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_pedestrian_waits_at_closed_door_boards_rides_and_reaches_sink_in_other_room(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    mgr = manager_factory(_scenario(f"elevator_ride_{planner}", planner), node_name=f"test_elevator_ride_{planner}", extra_params=PLANNERS[planner])
    assert _set_elevators(mgr, _ride_cabins("lift")).success
    aid = _rider(mgr)
    pool = mgr._pool

    _tick_until(mgr, lambda: aid in mgr._elevators.rides and mgr._elevators.rides[aid].phase == Phase.WAIT, 300)
    assert mgr._elevators.rides[aid].cabin == "lift_a"
    for _ in range(60):
        mgr.tick()
    i = pool.idx(aid)
    assert mgr._elevators.rides[aid].phase == Phase.WAIT
    assert math.hypot(pool.pos[i, 0] - 3.4, pool.pos[i, 1] - 4.6) <= WAIT_RADIUS + 0.3
    assert math.hypot(*pool.vel[i]) < 0.05

    assert _set_door(mgr, "lift_a", True).success
    _tick_until(mgr, lambda: mgr._elevators.rides[aid].phase == Phase.HOLD, 200)
    for _ in range(40):
        mgr.tick()
    i = pool.idx(aid)
    assert mgr._elevators.rides[aid].phase == Phase.HOLD
    assert math.hypot(*pool.vel[i]) < 0.05

    x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
    assert _teleport(mgr, [aid], [(x + ROOM_B_X, y)]).success
    assert aid not in mgr._elevators.rides
    i = pool.idx(aid)
    assert (pool.pos[i, 0], pool.pos[i, 1]) == (x + ROOM_B_X, y)
    assert not pool.vel[i].any()

    last: list[tuple[float, float]] = []

    def absorbed() -> bool:
        if aid not in mgr._agents:
            return True
        j = pool.idx(aid)
        last[:] = [(float(pool.pos[j, 0]), float(pool.pos[j, 1]))]
        assert aid not in mgr._elevators.rides
        return False

    _tick_until(mgr, absorbed, 400)
    assert math.hypot(last[0][0] - SINK.x, last[0][1] - SINK.y) < 1.0


def _riders(mgr: AgentManager, n: int) -> list[int]:
    _tick_until(mgr, lambda: len(mgr._agents) >= n, 200)
    return sorted(mgr._agents)[:n]


def test_second_rider_queues_clear_of_the_door_while_the_only_spot_is_taken(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("elevator_queue", "navmesh", riders=2), node_name="test_elevator_queue")
    assert _set_elevators(mgr, _ride_cabins("lift")).success
    assert len(mgr._elevators.cabins["lift_a"].slots) == 1
    first, second = _riders(mgr, 2)
    rides = mgr._elevators.rides

    _tick_until(mgr, lambda: all(a in rides and rides[a].phase == Phase.WAIT for a in (first, second)), 600)
    assert _set_door(mgr, "lift_a", True).success
    _tick_until(mgr, lambda: any(rides[a].phase == Phase.HOLD for a in (first, second)), 300)
    holder, queued = (first, second) if rides[first].phase == Phase.HOLD else (second, first)
    for _ in range(40):
        mgr.tick()
    assert rides[queued].phase == Phase.WAIT
    j = mgr._pool.idx(queued)
    assert math.hypot(mgr._pool.pos[j, 0] - 3.4, mgr._pool.pos[j, 1] - 4.6) <= WAIT_RADIUS + 0.3

    i = mgr._pool.idx(holder)
    assert _teleport(mgr, [holder], [(float(mgr._pool.pos[i, 0]) + ROOM_B_X, float(mgr._pool.pos[i, 1]))]).success
    _tick_until(mgr, lambda: rides[queued].phase == Phase.HOLD, 300)


def test_riders_of_one_cabin_hold_on_distinct_spots(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("elevator_spots", "navmesh", riders=2), node_name="test_elevator_spots")
    cabins = [_cabin("big_a", "big_b", 0.0, 3.0, half=1.0), _cabin("big_b", "big_a", ROOM_B_X, 3.0, half=1.0)]
    assert _set_elevators(mgr, cabins).success
    assert len(mgr._elevators.cabins["big_a"].slots) == 4
    riders = _riders(mgr, 2)
    rides = mgr._elevators.rides

    _tick_until(mgr, lambda: all(a in rides and rides[a].phase == Phase.WAIT for a in riders), 600)
    assert _set_door(mgr, "big_a", True).success
    _tick_until(mgr, lambda: all(rides[a].phase == Phase.HOLD for a in riders), 400)
    assert rides[riders[0]].slot != rides[riders[1]].slot


def test_holding_rider_pushed_out_of_an_open_cabin_boards_again(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("elevator_reboard", "navmesh"), node_name="test_elevator_reboard")
    assert _set_elevators(mgr, _ride_cabins("lift")).success
    aid = _rider(mgr)
    rides = mgr._elevators.rides
    _tick_until(mgr, lambda: aid in rides and rides[aid].phase == Phase.WAIT, 300)
    assert _set_door(mgr, "lift_a", True).success
    _tick_until(mgr, lambda: rides[aid].phase == Phase.HOLD, 200)

    i = mgr._pool.idx(aid)
    mgr._pool.pos[i] = (3.9, 4.6)
    mgr.tick()
    assert rides[aid].phase == Phase.BOARD
    _tick_until(mgr, lambda: rides[aid].phase == Phase.HOLD, 200)


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_ride_timeout_releases_waiting_rider_to_the_next_cabin(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    params = [*PLANNERS[planner], Parameter("elevator_ride_timeout", Parameter.Type.DOUBLE, 1.0)]
    mgr = manager_factory(_scenario(f"elevator_timeout_{planner}", planner), node_name=f"test_elevator_timeout_{planner}", extra_params=params)
    cabins = _ride_cabins("a") + [_cabin("b_a", "b_b", 0.0, 1.4), _cabin("b_b", "b_a", ROOM_B_X, 1.4)]
    assert _set_elevators(mgr, cabins).success
    aid = _rider(mgr)

    _tick_until(mgr, lambda: aid in mgr._elevators.rides and mgr._elevators.rides[aid].phase == Phase.WAIT, 300)
    assert mgr._elevators.rides[aid].cabin == "a_a"
    _tick_until(mgr, lambda: aid in mgr._elevators.rides and mgr._elevators.rides[aid].cabin == "b_a", 60)
    _tick_until(mgr, lambda: mgr._elevators.rides[aid].phase == Phase.WAIT, 300)
    i = mgr._pool.idx(aid)
    assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 1.4) <= WAIT_RADIUS + 0.3
    _tick_until(mgr, lambda: aid not in mgr._elevators.rides, 60)
    for _ in range(20):
        mgr.tick()
    assert aid not in mgr._elevators.rides


def test_set_elevators_replaces_and_clears_the_cabin_set_and_survives_soft_reset(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("elevator_set", "navmesh"), node_name="test_elevator_set")
    assert _set_elevators(mgr, _ride_cabins("x") + [_cabin("solo", "", 0.0, 1.4, door_open=True)]).success
    assert sorted(mgr._elevators.cabins) == ["solo", "x_a", "x_b"]
    assert mgr._elevators.open == {"x_a": False, "x_b": False, "solo": True}

    assert _set_elevators(mgr, _ride_cabins("y")).success
    assert sorted(mgr._elevators.cabins) == ["y_a", "y_b"]
    assert not _set_door(mgr, "x_a", True).success
    assert _set_door(mgr, "y_b", True).success
    assert mgr._elevators.open["y_b"]

    rejected = _set_elevators(mgr, [_cabin("z", "nowhere", 0.0, 1.4), _cabin("z", "", 0.0, 1.4)])
    assert not rejected.success
    assert "nowhere" in rejected.message and "'z'" in rejected.message
    assert sorted(mgr._elevators.cabins) == ["y_a", "y_b"]

    assert mgr._reset_callback(ResetSimulation.Request(soft=True), ResetSimulation.Response()).success
    assert sorted(mgr._elevators.cabins) == ["y_a", "y_b"]

    assert _set_elevators(mgr, []).success
    assert mgr._elevators.cabins == {}

    assert _set_elevators(mgr, _ride_cabins("y")).success
    assert mgr._reset_callback(ResetSimulation.Request(soft=False), ResetSimulation.Response()).success
    assert mgr._elevators.cabins == {}


def test_teleport_agents_rejects_unknown_ids_and_moves_nobody(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("elevator_teleport", "navmesh"), node_name="test_elevator_teleport")
    aid = _rider(mgr)
    i = mgr._pool.idx(aid)
    before = mgr._pool.pos[i].copy()

    response = _teleport(mgr, [aid, 999], [(2.0, 2.0), (3.0, 3.0)])
    assert not response.success
    assert "999" in response.message
    assert (mgr._pool.pos[i] == before).all()
    assert not _teleport(mgr, [aid], []).success
    assert _teleport(mgr, [aid], [(2.0, 2.0)]).success
    assert tuple(mgr._pool.pos[mgr._pool.idx(aid)]) == (2.0, 2.0)


def _go_to_scenario(tmp_path: Path, planner: str) -> ScenarioConfig:
    walls = [{"name": w.name, "start": {"x": w.start.x, "y": w.start.y}, "end": {"x": w.end.x, "y": w.end.y}} for w in _room("a", 0.0) + _room("b", ROOM_B_X)]
    raw = {
        "name": f"elevator_go_to_{planner}",
        "simulation": {"seed": 42, "dt": 0.05, "bt_tick_interval": 1},
        "modules": {"global_planner": planner},
        "walls": walls,
        "agent_types": {
            "go_to_walker": {
                "extends": "adult",
                "mode": "behavior_tree",
                "initial_sequence": "walk",
                "sequences": {"walk": {"steps": {"to_other_room": {"kind": "go_to", "target_pose": {"x": SINK.x, "y": SINK.y}, "patience": {"mean": 120.0, "std": 0.0, "clip_low": 120.0, "clip_high": 120.0}}}}},
            },
        },
        "agents": [{"agent_id": 1, "agent_type": "go_to_walker", "spawn_pose": {"x": 1.0, "y": 3.0, "theta": 0.0}, "desired_velocity": 1.2}],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_scenario(str(path))


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_behavior_tree_go_to_rider_rides_and_reaches_target_in_other_room(manager_factory: Callable[..., AgentManager], tmp_path: Path, planner: str) -> None:
    mgr = manager_factory(_go_to_scenario(tmp_path, planner), node_name=f"test_elevator_go_to_{planner}", extra_params=PLANNERS[planner])
    assert _set_elevators(mgr, _ride_cabins("lift")).success
    bt = mgr._behavior_trees[1]
    pool = mgr._pool

    _tick_until(mgr, lambda: 1 in mgr._elevators.rides and mgr._elevators.rides[1].phase == Phase.WAIT, 300)
    assert mgr._elevators.rides[1].cabin == "lift_a"
    assert _set_door(mgr, "lift_a", True).success
    _tick_until(mgr, lambda: mgr._elevators.rides[1].phase == Phase.HOLD, 200)
    i = pool.idx(1)
    assert _teleport(mgr, [1], [(float(pool.pos[i, 0]) + ROOM_B_X, float(pool.pos[i, 1]))]).success

    _tick_until(mgr, lambda: bt.root.status == py_trees.common.Status.SUCCESS, 600)
    assert 1 not in mgr._elevators.rides
    pose = mgr._agents[1].state.pose
    assert math.hypot(pose.x - SINK.x, pose.y - SINK.y) <= DISTANCE_TOLERANCE


def test_robots_with_a_goal_in_the_other_room_never_ride(manager_factory: Callable[..., AgentManager]) -> None:
    goal = [Pose2D(x=SINK.x, y=SINK.y)]
    agents = [
        AgentConfig(agent_id=-1, kind=1, policy="sfm", agent_type="robot", spawn_pose=Pose2D(x=1.0, y=1.5), goal_sequence=goal, desired_velocity=1.0),
        AgentConfig(agent_id=-2, kind=1, policy="", agent_type="robot", spawn_pose=Pose2D(x=1.0, y=4.5), goal_sequence=goal, desired_velocity=1.0),
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=goal, desired_velocity=1.2),
    ]
    scenario = attrs.evolve(_scenario("elevator_robots", "navmesh"), flow=None, agents=agents)
    mgr = manager_factory(scenario, node_name="test_elevator_robots")
    assert _set_elevators(mgr, _ride_cabins("lift")).success
    assert {-1, -2, 1} <= set(mgr._agents)

    phases: set[Phase] = set()
    for _ in range(300):
        mgr.tick()
        assert not {-1, -2} & mgr._elevators.rides.keys()
        assert all(mgr._high_level_cmds[r].target_pose.x == SINK.x for r in (-1, -2))
        if 1 in mgr._elevators.rides:
            phases.add(mgr._elevators.rides[1].phase)
    assert Phase.WAIT in phases
