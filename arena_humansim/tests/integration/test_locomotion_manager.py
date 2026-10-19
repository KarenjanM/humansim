from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import attrs
import numpy as np
import pytest
from arena_humansim.core import interaction_classes
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.locomotion import MODE_NORMAL, MODE_REVERSE
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.local_planner.hsfm import HSFMPlanner
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
    WallConfig,
)
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import ResetSimulation, SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

from ._capture import LATCHED, STREAM, settle, spin_until, subscribe

DT = 0.05
TICKS = 100
WHEELCHAIR = str(Path(__file__).resolve().parents[2] / "config" / "agent_types" / "wheelchair_manual.yaml")
LIMP = str(Path(__file__).resolve().parents[2] / "config" / "agent_types" / "adult_limp_right.yaml")


def _corridor(name: str, local_planner: str = "hsfm") -> ScenarioConfig:
    walls = [
        WallConfig(name="north", start=Pose2D(x=-8.0, y=1.5), end=Pose2D(x=8.0, y=1.5)),
        WallConfig(name="south", start=Pose2D(x=-8.0, y=-1.5), end=Pose2D(x=8.0, y=-1.5)),
    ]
    return ScenarioConfig(name=name, simulation=SimulationParams(seed=7, dt=DT, max_ticks=TICKS), modules=ModuleConfig(local_planner=local_planner), walls=walls)


def _spawn(mgr: AgentManager, agent_type: str, x: float, theta: float, goal_x: float, kind: int = 0, y: float = 0.0) -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.pose = RosPose2D(x=x, y=y, theta=theta)
    msg.desired_velocity = 0.9
    msg.agent_type = agent_type
    msg.kind = kind
    wp = WaypointMsg()
    wp.pose = RosPose2D(x=goal_x, y=y, theta=0.0)
    msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    assert resp.success, resp.message
    return resp.spawned_ids[0]


def _turn_step(pool: AgentPool, idx: int, speed: float) -> float:
    return float(max(pool.pivot_angular_velocity[idx], speed / pool.min_turning_radius[idx])) * DT


def test_wheelchair_manual_drives_along_its_heading_and_publishes_gait(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_corridor("wheelchair_corridor"), node_name="test_wheelchair_corridor")
    metas = subscribe(mgr, AgentMetaMsg, "agent_meta", LATCHED)
    frames = subscribe(mgr, AgentFrameMsg, "agent_states", STREAM)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)

    chair = _spawn(mgr, WHEELCHAIR, -5.0, 0.0, 6.0, y=-0.6)
    walker = _spawn(mgr, "adult", 5.0, math.pi, -6.0, y=0.8)
    pool = mgr._pool
    agent = mgr._agents[chair]
    assert agent.params.name == "wheelchair_manual"
    assert agent.params.local_planner == "hsfm"
    assert mgr._policy_names[int(pool.policy_idx[pool.idx(chair)])] == "hsfm"
    assert agent.params.locomotion.active
    assert agent.params.locomotion.kinematics == 1
    assert pool.axial_offset[pool.idx(chair)] == pytest.approx(0.55 - 0.325)
    assert interaction_classes.name(int(pool.interaction_class[pool.idx(chair)])) == "wheelchair"
    assert pool.interaction_class[pool.idx(walker)] == interaction_classes.HUMAN
    assert mgr._locomotion.active[pool.idx(chair)]
    assert not mgr._locomotion.active[pool.idx(walker)]

    lateral = []
    forward = []
    planned = []
    start_x = float(pool.pos[pool.idx(chair), 0])
    for _ in range(TICKS):
        mgr.tick()
        i = pool.idx(chair)
        vx, vy = pool.vel[i]
        theta = float(pool.theta[i])
        lateral.append(abs(vx * math.sin(theta) - vy * math.cos(theta)))
        forward.append(vx * math.cos(theta) + vy * math.sin(theta))
        planned.append(math.hypot(*pool.prev_vel[i]))
    assert max(lateral) < 1e-9
    assert min(forward) >= 0.0
    assert max(forward) > 0.5
    assert float(pool.pos[pool.idx(chair), 0]) > start_x + 2.0
    published = np.asarray(forward[50:90])
    state = np.asarray(planned[50:90])
    assert np.ptp(published) > 0.1
    assert np.ptp(state) < 0.5 * np.ptp(published)

    spin_until(executor, lambda: len(frames) >= TICKS and metas and chair in list(metas[-1].agent_id))
    frame = frames[-1]
    ids = list(frame.agent_id)
    chair_row = ids.index(chair)
    walker_row = ids.index(walker)
    assert frame.gait_phase[chair_row] != 0.0
    assert frame.gait_cadence[chair_row] >= 0.6
    assert frame.gait_phase[walker_row] == 0.0
    assert frame.gait_cadence[walker_row] == 0.0
    phases = [f.gait_phase[list(f.agent_id).index(chair)] for f in frames[-10:]]
    assert all(b > a for a, b in zip(phases[:-1], phases[1:], strict=True))

    meta = metas[-1]
    assert list(meta.agent_id) == [chair, walker]
    assert list(meta.agent_type) == ["wheelchair_manual", "adult"]
    executor.remove_node(mgr)


def test_robot_kind_rows_take_the_robot_interaction_class(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_corridor("robot_class_corridor"), node_name="test_robot_class_corridor")
    robot = _spawn(mgr, "robot", -5.0, 0.0, 6.0, kind=KIND_ROBOT)
    pool = mgr._pool
    assert pool.kind[pool.idx(robot)] == KIND_ROBOT
    assert pool.interaction_class[pool.idx(robot)] == interaction_classes.ROBOT
    ext = AgentStateMsg(agent_id=0, name="ext_robot", pose=RosPose2D(x=2.0, y=0.0, theta=0.0), radius=0.3)
    aid = mgr._spawn_external_agent(ext)
    assert pool.interaction_class[pool.idx(aid)] == interaction_classes.ROBOT
    assert np.all(mgr._locomotion.frame_arrays(np.arange(pool.n))[0] == 0.0)


def test_frame_gait_arrays_and_meta_agent_type_match_the_agent_count(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_corridor("frame_lengths_corridor"), node_name="test_frame_lengths_corridor")
    metas = subscribe(mgr, AgentMetaMsg, "agent_meta", LATCHED)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)

    first = _spawn(mgr, "adult", -5.0, 0.0, 6.0, y=0.8)
    second = _spawn(mgr, "elder", 5.0, math.pi, -6.0, y=0.8)
    mgr.tick()
    frame = mgr._build_agent_frame()
    assert list(frame.agent_id) == [first, second]
    assert list(frame.gait_phase) == [0.0, 0.0]
    assert list(frame.gait_cadence) == [0.0, 0.0]
    spin_until(executor, lambda: bool(metas) and list(metas[-1].agent_id) == [first, second])
    assert list(metas[-1].agent_type) == ["adult", "elder"]

    world = AgentStatesMsg()
    world.agents.append(AgentStateMsg(agent_id=900, name="ext_robot", pose=RosPose2D(x=0.0, y=0.0, theta=0.0), radius=0.3))
    mgr._world_state_callback(world)
    mgr.tick()
    mirror = mgr._external_entities[900].agent_id
    chair = _spawn(mgr, WHEELCHAIR, -5.0, 0.0, 6.0, y=-0.6)
    pool = mgr._pool
    assert pool.idx(mirror) < pool.idx(chair)
    assert pool.interaction_class[pool.idx(mirror)] == interaction_classes.ROBOT
    mgr.tick()
    frame = mgr._build_agent_frame()
    assert list(frame.agent_id) == [first, second, chair]
    assert len(frame.gait_phase) == len(frame.gait_cadence) == 3
    assert list(frame.gait_phase[:2]) == [0.0, 0.0]
    assert frame.gait_phase[2] != 0.0
    assert frame.gait_cadence[2] >= 0.6
    spin_until(executor, lambda: list(metas[-1].agent_id) == [first, second, chair])
    assert list(metas[-1].agent_type) == ["adult", "elder", "wheelchair_manual"]
    executor.remove_node(mgr)


def test_integrator_leaves_the_heading_of_extension_owned_rows_to_the_extension(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    cart_yaml = tmp_path / "cart.yaml"
    cart_yaml.write_text("name: cart\nextends: adult\nlocomotion:\n  kinematics: along_heading\n")
    glider_yaml = tmp_path / "glider.yaml"
    glider_yaml.write_text("name: glider\nextends: adult\nlocomotion:\n  recovery: {stall_after_s: 1.0, reverse_m: 0.3}\n")
    mgr = manager_factory(_corridor("integrator_heading_corridor", local_planner="sfm"), node_name="test_integrator_heading_corridor")
    cart = _spawn(mgr, str(cart_yaml), -5.0, 0.0, 6.0, y=-0.8)
    glider = _spawn(mgr, str(glider_yaml), -3.0, 0.0, 6.0, y=0.0)
    walker = _spawn(mgr, "adult", -1.0, 0.0, 6.0, y=0.8)
    pool = mgr._pool
    ext = mgr._locomotion
    rows = [pool.idx(cart), pool.idx(glider), pool.idx(walker)]
    assert not mgr._provides_heading_mask(pool).any()

    pool.theta[rows] = 0.0
    pool.vel[rows] = (-0.3, 0.0)
    ext.mode[pool.idx(glider)] = MODE_REVERSE
    mgr._integrate_state_vectorized(pool)
    assert pool.theta[pool.idx(cart)] == 0.0
    assert pool.theta[pool.idx(glider)] == 0.0
    assert abs(pool.theta[pool.idx(walker)]) == pytest.approx(_turn_step(pool, pool.idx(walker), 0.3))

    ext.mode[pool.idx(glider)] = MODE_NORMAL
    pool.theta[rows] = 0.0
    pool.vel[rows] = (0.0, 0.3)
    mgr._integrate_state_vectorized(pool)
    assert pool.theta[pool.idx(cart)] == 0.0
    assert pool.theta[pool.idx(glider)] == pytest.approx(_turn_step(pool, pool.idx(glider), 0.3))
    assert pool.theta[pool.idx(walker)] == pytest.approx(_turn_step(pool, pool.idx(walker), 0.3))

    pool.theta[rows] = 0.0
    pool.goal_theta[pool.idx(cart)] = 1.0
    pool.has_goal_theta[pool.idx(cart)] = True
    mgr._integrate_state_vectorized(pool)
    assert pool.theta[pool.idx(cart)] > 0.0


def test_type_pinned_planner_drives_its_agent_under_a_different_default(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_corridor("pinned_planner_corridor", local_planner="sfm"), node_name="test_pinned_planner_corridor")
    walker = _spawn(mgr, "adult", 5.0, math.pi, -6.0, y=0.8)
    chair = _spawn(mgr, WHEELCHAIR, -5.0, 0.0, 6.0, y=-0.6)
    pool = mgr._pool
    assert mgr._policy_names == ["sfm", "hsfm"]
    assert mgr._policy_names[int(pool.policy_idx[pool.idx(walker)])] == "sfm"
    assert mgr._policy_names[int(pool.policy_idx[pool.idx(chair)])] == "hsfm"
    hsfm = mgr._policies[int(pool.policy_idx[pool.idx(chair)])]
    assert isinstance(hsfm, HSFMPlanner)
    assert mgr._agents[chair].local_planner is hsfm
    assert mgr._module_pool["local_planner"]["hsfm"] is hsfm
    assert mgr._agents[walker].local_planner is mgr._local_planner
    row = pool.idx(chair)
    lp = mgr._agents[chair].params.local_planner_params
    assert hsfm._heading_source[row] == 1.0
    assert hsfm._lateral_gain[row] == 0.0
    assert hsfm._angular_gain[row] == lp["angular_gain"] > 0.0
    assert hsfm._relaxation_time[row] == lp["relaxation_time"] > 0.0
    assert hsfm._relaxation_time[pool.idx(walker)] > 0.0

    for _ in range(TICKS):
        mgr.tick()
    assert pool.pos[pool.idx(chair), 0] > -5.0 + 2.5
    assert pool.pos[pool.idx(walker), 0] < 5.0 - 2.5
    assert math.hypot(*pool.prev_vel[pool.idx(chair)]) > 0.5

    assert mgr.set_parameters_atomically([Parameter("local_planner", value="helbing")]).successful
    mgr._reset_callback(ResetSimulation.Request(soft=True), ResetSimulation.Response())
    assert sorted(mgr._policy_names) == ["helbing", "hsfm"]
    assert mgr._policy_names[int(pool.policy_idx[pool.idx(chair)])] == "hsfm"
    assert mgr._policy_names[int(pool.policy_idx[pool.idx(walker)])] == "helbing"
    assert mgr._policies[int(pool.policy_idx[pool.idx(chair)])] is hsfm
    assert mgr._agents[chair].local_planner is hsfm
    assert mgr._module_pool["local_planner"]["hsfm"] is hsfm
    assert hsfm._heading_source[pool.idx(chair)] == 1.0
    before = float(pool.pos[pool.idx(chair), 0])
    for _ in range(20):
        mgr.tick()
    assert pool.pos[pool.idx(chair), 0] > before + 0.5

    late = _spawn(mgr, WHEELCHAIR, -7.0, 0.0, 6.0, y=0.0)
    assert mgr._policy_names[int(pool.policy_idx[pool.idx(late)])] == "hsfm"
    assert mgr._agents[late].local_planner is hsfm


@pytest.mark.parametrize(("default", "pinned"), [("sfm", "hsfm"), ("hsfm", "sfm")])
def test_every_pool_planner_plans_from_the_pre_plan_velocities(manager_factory: Callable[..., AgentManager], tmp_path: Path, default: str, pinned: str) -> None:
    pinned_yaml = tmp_path / f"adult_{pinned}.yaml"
    pinned_yaml.write_text(f"name: adult_{pinned}\nextends: adult\nlocal_planner: {pinned}\n")
    mgr = manager_factory(_corridor(f"mixed_{default}_{pinned}_corridor", local_planner=default), node_name=f"test_mixed_{default}_{pinned}_corridor")
    first = _spawn(mgr, "adult", -6.0, 0.0, 7.0, y=0.9)
    second = _spawn(mgr, "adult", -6.0, 0.0, 7.0, y=0.0)
    other = _spawn(mgr, str(pinned_yaml), -6.0, 0.0, 7.0, y=-0.9)
    pool = mgr._pool
    assert mgr._policy_names == [default, pinned]
    assert [mgr._policy_names[int(pool.policy_idx[pool.idx(a)])] for a in (first, second, other)] == [default, default, pinned]
    top = dict.fromkeys((first, second, other), 0.0)
    for _ in range(TICKS):
        mgr.tick()
        for aid in top:
            top[aid] = max(top[aid], math.hypot(*pool.prev_vel[pool.idx(aid)]))
    for aid, speed in top.items():
        assert speed >= 0.8 * pool.desired_vel[pool.idx(aid)], (mgr._agents[aid].params.name, speed)
    assert pool.desired_vel[pool.idx(other)] == 0.9


def test_scheduler_spawns_take_the_planner_their_type_pins(manager_factory: Callable[..., AgentManager]) -> None:
    flow = FlowScenarioConfig(
        sources=[
            SourceScenarioConfig(
                pose=Pose2D(x=-6.0, y=0.0, theta=0.0),
                shape=ShapeModel(type="circle", radius=0.2),
                type="poisson",
                rate_profile=[RateKeyframeModel(t=0.0, rate=5.0)],
                max_concurrent=1,
                agent_template=AgentTemplateModel(agent_type=WHEELCHAIR, desired_velocity_min=0.9, desired_velocity_max=0.9, sink_affinity=[SinkAffinityModel(sink_idx=0, weight=1.0)]),
            ),
        ],
        sinks=[SinkScenarioConfig(pose=Pose2D(x=6.0, y=0.0, theta=0.0), shape=ShapeModel(type="circle", radius=0.5), absorption_radius=0.5)],
    )
    scenario = attrs.evolve(_corridor("pinned_flow_corridor", local_planner="sfm"), flow=flow)
    mgr = manager_factory(scenario, node_name="test_pinned_flow_corridor")
    pool = mgr._pool
    for _ in range(TICKS):
        mgr.tick()
        if pool.n:
            break
    assert pool.n == 1
    assert mgr._agents[int(pool.agent_ids[0])].params.name == "wheelchair_manual"
    assert mgr._policy_names[int(pool.policy_idx[0])] == "hsfm"
    start = float(pool.pos[0, 0])
    for _ in range(60):
        mgr.tick()
    assert pool.pos[0, 0] > start + 1.0


def test_adult_limp_right_surges_and_lurches_once_per_cycle_while_an_adult_walks_steadily(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_corridor("limp_corridor", local_planner="sfm"), node_name="test_limp_corridor")
    limp = _spawn(mgr, LIMP, -7.0, 0.0, 7.5, y=-0.7)
    adult = _spawn(mgr, "adult", -7.0, 0.0, 7.5, y=0.7)
    pool = mgr._pool
    li, ai = pool.idx(limp), pool.idx(adult)
    assert mgr._agents[limp].params.name == "adult_limp_right"
    speed = np.zeros((200, 2))
    lateral = np.zeros((200, 2))
    phase = np.zeros(200)
    for i in range(260):
        mgr.tick()
        if i >= 60:
            vel = pool.vel[[li, ai]]
            theta = pool.theta[[li, ai]]
            speed[i - 60] = np.hypot(vel[:, 0], vel[:, 1])
            lateral[i - 60] = -np.sin(theta) * vel[:, 0] + np.cos(theta) * vel[:, 1]
            phase[i - 60] = mgr._locomotion.phase[li]
    assert np.all(np.diff(phase) > 0.0)
    cycles = (phase[-1] - phase[0]) / (2.0 * math.pi)
    assert 5.0 < cycles < 12.0
    assert np.ptp(speed[:, 0]) > 0.08 * speed[:, 0].mean()
    assert np.ptp(lateral[:, 0]) > 0.03 * speed[:, 0].mean()
    assert np.ptp(speed[:, 1]) < 0.02 * speed[:, 1].mean()
    gait_phase, gait_cadence = mgr._locomotion.frame_arrays(np.array([li, ai]))
    assert gait_phase[0] > 0.0 and gait_cadence[0] > 0.0
    assert gait_phase[1] == 0.0 and gait_cadence[1] == 0.0
