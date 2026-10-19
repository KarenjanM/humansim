from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from rclpy.parameter import Parameter

from ._helpers import build_manager


@pytest.fixture
def manager_factory(rclpy_context: object) -> Iterator[Callable[..., AgentManager]]:  # noqa: ARG001
    created: list[AgentManager] = []

    def make(scenario: ScenarioConfig, node_name: str = "test_manager", extra_params: Iterable[Parameter] | None = None) -> AgentManager:
        mgr = build_manager(scenario, node_name=node_name, extra_params=extra_params)
        created.append(mgr)
        return mgr

    yield make

    for mgr in created:
        try:
            mgr.destroy_node()
        except Exception:
            pass
