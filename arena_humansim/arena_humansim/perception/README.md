# Perception

Builds each agent's belief about its neighbors - which other agents it currently sees, and at what bearing. Runs once per tick, before decision and planning.

## Available

| Name | Class | Notes |
|---|---|---|
| `default` | `DefaultPerception` | KDTree neighbor lookup + per-agent range/FOV filtering. Populates `pool.neighbor_csr` for the whole simulation in one call. |

## Contract

```python
class Perception(Loggable, ABC):
    supports_pool: bool = False

    @abstractmethod
    def compute(self, agent, all_agents, world_state, belief) -> BeliefState: ...

    def compute_pool(self, pool) -> None: ...
```

- `compute` (per-agent) is the fallback used in tests and off the hot path.
- `supports_pool=True` + `compute_pool(pool)` is the hot path. It writes a CSR (`indptr`, `indices`) into `pool.set_neighbor_csr`. `AgentManager` consumes this directly.
- The CSR encodes directional visibility: `indptr[i]:indptr[i+1]` lists the agent_ids that *agent i sees*, in agent id order. It is not symmetric - observer-A may see B without B seeing A.

## `proximity_sense` bypasses FOV

On top of the range + FOV gate, every agent always perceives neighbors within its `proximity_sense` radius (default ~1 m, Hall's personal-zone upper bound) regardless of bearing. The gate is `(range_ok & fov_ok) | prox_ok` in both paths. Without this, a co-located peer standing behind an agent at a gathering spot would be invisible, and the interaction manager would split a single GROUP_CONVERSATION into two 1p FORMING interactions that never merge. The `query_ball_point` / `sparse_distance_matrix` call uses `max(vision_range, proximity_sense)` so short-range neighbors still reach the per-row filter.

## Dense vs KDTree paths

`DefaultPerception` switches strategies at `_SMALL_N_THRESHOLD = 64`:

- **Dense** (N <= 64) - pairwise distance matrix, broadcast FOV test. Faster for small N, avoids tree-build overhead.
- **KDTree** (N > 64) - `scipy.spatial.cKDTree.sparse_distance_matrix` at the max vision range in the pool, then per-row range + FOV pruning.

Both paths must produce **bitwise-identical** CSRs for the same state. The contract test `tests/contracts/test_perception_contract.py` enforces this. Note the bearing convention: target-observer (`dx = target.x - observer.x`). The dense path had this inverted and was fixed on 2026-04-15 - do not re-invert without updating both paths.

## Adding a perception module

1. Subclass `Perception` in a new file under `perception/`.
2. Implement `compute`. Optionally set `supports_pool=True` and implement `compute_pool`.
3. Register in `perception/__init__.py` via a `_load_<name>` lazy loader.
4. Contract coverage in `tests/contracts/test_perception_contract.py` - the CSR-symmetry, range-gate, and FOV-gate tests apply to every module.

## Downstream consumers

- **InteractionManager** - rule 4 of `_try_bind` resolves visibility through `pool.visible_agent_ids`, which reads the CSR.
- **SFM** - neighbor repulsion terms iterate the CSR directly.
- **Markers** - `publish_markers=2` draws FOV cones and neighbor edges.
