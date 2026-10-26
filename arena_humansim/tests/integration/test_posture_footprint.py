from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import PRONE_RADIUS, SEAT_RISE_S, AgentManager
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.world_knowledge import WorldObject
from arena_humansim.utils.types import BehaviorTreeMovement, InteractionOutcome, Pose2D, SeekSpec
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D


def test_prone_posture_widens_the_footprint_and_restores(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(name="posture", simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    mgr = manager_factory(scenario, node_name="test_posture")
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.name = "ped"
    msg.kind = AgentStateMsg.KIND_HUMAN
    msg.pose = RosPose2D(x=0.0, y=0.0, theta=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.3
    msg.agent_type = "adult"
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    aid = resp.spawned_ids[0]
    agent = mgr._agents[aid]
    agents = [mgr._agents[aid] for aid in mgr._pool_agent_ids]
    idx = agents.index(agent)

    mgr._apply_postures(agents, mgr._pool)
    assert mgr._pool.agent_radius[idx] == pytest.approx(agent.params.agent_radius)
    agent.movement = BehaviorTreeMovement(posture="prone")
    mgr._apply_postures(agents, mgr._pool)
    assert mgr._pool.agent_radius[idx] == pytest.approx(PRONE_RADIUS)
    agent.movement = BehaviorTreeMovement()
    mgr._apply_postures(agents, mgr._pool)
    assert mgr._pool.agent_radius[idx] == pytest.approx(agent.params.agent_radius)


def test_released_seat_returns_the_agent_to_where_it_walked_up(manager_factory: Callable[..., AgentManager]) -> None:
    """A released agent goes back to where it walked up from and stays there while it gets up."""
    scenario = ScenarioConfig(name="unpark", simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    mgr = manager_factory(scenario, node_name="test_unpark")
    seat = Pose2D(x=0.0, y=0.0, theta=1.0)
    mgr._world_knowledge.add_object(WorldObject(object_id="chair", type="chair", pose=Pose2D(x=0.0, y=0.0, theta=0.0), seats=[seat]))

    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.name = "ped"
    msg.kind = AgentStateMsg.KIND_HUMAN
    msg.pose = RosPose2D(x=0.6, y=0.0, theta=0.0)
    msg.desired_velocity = 1.0
    msg.radius = 0.3
    msg.agent_type = "adult"
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    aid = resp.spawned_ids[0]
    pool = mgr._pool
    idx = pool._id_to_idx[aid]

    im = mgr._interaction_manager
    iid = im._create_interaction(creator_id=aid, spec=SeekSpec(interaction_type=InteractionType.SIT_ON, target="chair")).id
    im.interactions[iid].outcome = InteractionOutcome.ACTIVE
    im.update({}, dt=0.05)
    mgr._park_seated(pool)
    assert (pool.pos[idx, 0], pool.pos[idx, 1]) == pytest.approx((seat.x, seat.y))

    im.force_stop(aid)
    mgr._park_seated(pool)
    assert (pool.pos[idx, 0], pool.pos[idx, 1]) == pytest.approx((0.6, 0.0))
    assert aid not in mgr._parked_from

    pool.pos[idx] = (1.0, 0.0)
    mgr._tick_count += round(SEAT_RISE_S / 0.05)
    mgr._park_seated(pool)
    assert (pool.pos[idx, 0], pool.pos[idx, 1]) == pytest.approx((0.6, 0.0))
    pool.pos[idx] = (1.0, 0.0)
    mgr._tick_count += 1
    mgr._park_seated(pool)
    assert (pool.pos[idx, 0], pool.pos[idx, 1]) == pytest.approx((1.0, 0.0))
