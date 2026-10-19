import numpy as np
from rclpy.logging import get_logger

from arena_humansim.core.agents import BaseAgent, ParamDist
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.world_knowledge import WorldObject
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import (
    BehaviorTreeMovement,
    CommandType,
    HighLevelCommand,
    InteractionOutcome,
    Pose2D,
    SeekSpec,
    pose_distance,
)

_bt_logger = get_logger("behavior_tree")


def _sample_param_dist(dist: ParamDist, rng: np.random.Generator) -> float:
    value = rng.normal(dist.mean, dist.std) if dist.std > 0 else dist.mean
    return float(np.clip(value, dist.clip_low, dist.clip_high))


def _nav_command(agent: BaseAgent, target_pose: Pose2D) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent.state.agent_id,
        type=CommandType.NAVIGATE,
        target_pose=target_pose,
        desired_velocity=agent.state.desired_velocity,
    )


def _set_command(agent: BaseAgent, cmd: HighLevelCommand | None) -> None:
    mv = agent.movement
    if isinstance(mv, BehaviorTreeMovement):
        mv.command = cmd


def _seek_command(agent: BaseAgent, spec: SeekSpec) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent.state.agent_id,
        type=CommandType.SEEK,
        desired_velocity=agent.state.desired_velocity,
        spec=spec,
    )


def _cancel_command(agent: BaseAgent, interaction_id: int | None = None) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent.state.agent_id,
        type=CommandType.STOP,
        interaction_target=interaction_id if interaction_id is not None else -1,
        reason=InteractionOutcome.CANCELED,
    )


def _at_target(agent: BaseAgent, target_pose: Pose2D, tolerance: float = DISTANCE_TOLERANCE) -> bool:
    return pose_distance(agent.state.pose, target_pose) < tolerance


def _resolve_interaction_radius(obj: WorldObject, step_override: float | None, interaction_name: str | None) -> float:
    if step_override is not None:
        return step_override
    if obj.interaction_radius is not None:
        return float(obj.interaction_radius)
    if interaction_name is not None:
        return InteractionType[interaction_name].kind.interaction_radius
    if obj.formation is not None:
        params = obj.formation.params or {}
        base = float(params.get("base_radius", params.get("base_step", 0.0)))
        if base > 0.0:
            return base + DISTANCE_TOLERANCE
    return DISTANCE_TOLERANCE
