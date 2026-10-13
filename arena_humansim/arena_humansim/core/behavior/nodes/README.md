# nodes

Composable `py_trees.Behaviour` primitives the compiler assembles into per-agent trees. See [../README.md](../README.md) for cross-node invariants (patience phases, seek/cancel semantics).

## Layout

| Module | Contents |
|---|---|
| `helpers.py` | `_nav_command`, `_seek_command`, `_cancel_command`, `_at_target`, `_resolve_interaction_radius`, `_sample_param_dist`, `_bt_logger`. |
| `utility.py` | Utility scoring: `check_condition`, `preconditions_met`, `score_actions`. |
| `primitives.py` | Leaf nodes that don't resolve world state: `ClearOutcomeNode`, `PatienceWatchdogNode`, `HoldNode`, `SatisfyNode`, `NeedsDecayNode`, `TuneNode` (holds a step's `tune:`), `ConditionalTunesNode` (holds the type's `tunes:` entries while their `when` holds). |
| `navigation.py` | `ResolveObjectNode`, `GoToNode` - target resolution and approach. |
| `attention.py` | `AttentionNode` - drives every channel of one `attention:` block: resolves each entry per tick (`partner`/`partners` via `ctx.im`, `target` via `StepContext`, `goal` via `movement.command`, `robot:<name>`, object id, agent name, object type, agent id, literal xyz, relative azimuth/elevation), walks the list (`dwell` or `unreachable` cycling with the reach hysteresis from `../reach.py`), faces the winning channel's entry when `face` allows and the ped is neither walking (`walking` predicate) nor bound (`FACE_ENTER_RAD`, `FACE_KEEP_RAD`, `FACE_TIMEOUT_S`), and publishes the shown channels as `GestureIntent`s on `movement.gestures`. `bare=True` halts, owns duration and returns SUCCESS or FAILURE (unresolved past `RESOLVE_TIMEOUT_S`, facing timeout). Riders never finish and FAIL only when `required`. `suspend()` lowers the channels without losing list position. `terminate()` clears `heading_goal` and removes `hold: release` channels. `SequenceRiderNode` ticks a sequence's rider next to its steps. |
| `interaction.py` | `SeekNode` (universal find-or-create-or-join), `CancelNode` (explicit teardown), `BlockNode` (pursuit+SEEK for BLOCK, pursuit speed boost held as a tune). |
| `autonomous.py` | `AutonomousNode` - utility-based action selection inside a step. |
| `state_machine.py` | `SequenceStateMachine` - runs one compiled sequence with `transitions` / `then` / `on_failure` edges and holds the sequence's `tune:`. |

Public API is re-exported from `__init__.py`. Import `from arena_humansim.core.behavior.nodes import ...`; submodules are internal and can be reorganized freely.

## Authoring rules

- **One agent per node.** Never emit a `HighLevelCommand` for anyone but `self._agent`.
- **Re-emit nav every tick.** Formations overwrite `movement.command`, so `GoToNode` re-emits NAVIGATE unconditionally. `HoldNode` re-emits NAVIGATE-to-self (current pose, `desired_velocity=0.0`) - it halts motion without tearing down live interactions. `SeekNode` re-emits SEEK only while **not yet bound**; once bound it short-circuits to SUCCESS and leaves `movement.command` alone so the formation-driven NAVIGATE (written by IM after each tick) isn't clobbered on the next BT tick.
- **Commands vs signals.** NAVIGATE and STOP are movement commands - the emitter owns the agent's motion. SEEK is a signal that mutates IM-side state (creates/joins/queues) and does not by itself move the agent; the formation emitter or a sibling `GoToNode` is what moves them. The motion pipeline filters `movement.command` by type: only NAVIGATE feeds the global planner and pool terminals.
- **Arrival latch is per-step.** Once `GoToNode` enters, it resets `pool.latched` for the agent so a stale latch from a previous `go_to` step can't short-circuit this one to SUCCESS. The latch also releases whenever `pool.has_terminal` goes False (i.e. the step emitted a STOP or SEEK), preventing spurious arrivals on the next NAVIGATE.
- **Explicit teardown.** Interactions end via `CancelNode` (STOP with `reason=CANCELED`) or contract duration expiring (COMPLETED). `SeekNode.terminate()` is a no-op - the interaction persists after the node exits. Teardown is always explicit.
- **Intrinsics go through `retune`.** A node that needs other params or another desired speed while it runs takes a key with `retune(agent, self.name, tune)` in `initialise` or `update` and releases it with `retune(agent, self.name, None)` in `terminate`. Never write `agent.params` or `state.desired_velocity` directly. See [../README.md](../README.md#tunes).
- **Composite teardown.** Use `stop(INVALID)` to cascade `terminate()` into children; `terminate()` alone is a leaf-only hook. See `SequenceStateMachine._goto`.
- **Signals on `BehaviorTreeMovement`.** IM writes these; nodes read them.
  - `interaction_id` - primary interaction the agent is in (FORMING or ACTIVE), or `None`. Cleared when the agent leaves.
  - `last_outcome` - `COMPLETED` / `CANCELED` / `INTERRUPTED` set on interaction end. Nodes return SUCCESS on `COMPLETED` / `CANCELED`, FAILURE on `INTERRUPTED`. `INTERRUPTED` wins same-tick collisions against a still-bound check.
  - `ClearOutcomeNode` at the head of every compiled step resets `last_outcome` and `interaction_id` so stale signals from a previous step don't short-circuit the next one.
  - "Is the agent bound?" is not stored on the movement - use `im.is_bound(aid)` (threaded into `StepContext.is_bound_lookup`).
- **`gestures` and `heading_goal` on `BehaviorTreeMovement` are BT-owned signals:** written by `AttentionNode` (`heading_goal` cleared on exit, `gestures` entries removed on exit unless `hold: keep`, replaced by slot when a later node takes the slot over) and wiped by `SequenceStateMachine` when it leaves a sequence, never by IM. `agent_manager` merges `heading_goal` into `pool.set_heading_goals` (formation headings win) and publishes `gestures` on `AgentState.gestures`.
- **Watchdogs never succeed.** `PatienceWatchdogNode` returns only RUNNING or FAILURE. It sits in a `Parallel(SuccessOnOne)` where the sibling sequence signals success.
- **Context over args.** Multi-node step chains (resolve -> go-to -> seek) thread `StepContext` rather than passing resolved poses through constructor args - initialisation order in `Sequence` isn't guaranteed to match authoring order.

## BT flow idioms

Every interaction is driven by one node: `SeekNode`. It emits SEEK; IM finds a matching open interaction and joins, or (if creation is allowed for this `InteractionKind`) creates one. SUCCESS when the interaction reaches ACTIVE (or when `last_outcome` arrives as `COMPLETED`/`CANCELED`). `CancelNode` ends it.

### Symmetric peer interaction (TALK_TO, GROUP_CONVERSATION, WAVE_AT)

Both peers have the same step; whichever ticks first creates a FORMING interaction, the other joins and promotes it to ACTIVE.

```yaml
chat: {interaction: TALK_TO, duration: {mean: 8.0}}
```

Compiled: `ClearOutcome -> Seek -> Satisfy?`. `duration:` threads to the contract's IM timer -> `COMPLETED`.

### Object-anchored (SIT_ON, LIE_ON, USE, QUEUE_USE)

Seeker names the object (id or type); IM find-or-creates on that object. The compiler walks the agent to the object first.

```yaml
rest: {target: bench, interaction: SIT_ON, duration: {mean: 5.0}}
```

Compiled: `ClearOutcome -> Resolve -> GoTo -> Seek -> Satisfy?`. Cluster formation pins the agent; IM timer ends the sit.

### Asymmetric service provider

Creates the interaction and waits for seekers. `offer: true` is only valid on SERVICE.

```yaml
offer_ride:
  interaction: SERVICE
  target: escort_ride        # service tag (str)
  offer: true
  min_participants: 2        # provider + one seeker
  max_participants: 2
  queueable: false
  formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}
```

### Asymmetric service seeker

Polls for a visible provider of the matching tag. Pure seeker - no `offer`.

```yaml
hail: {interaction: SERVICE, target: escort_ride}
```

### Mobile provider (escort / shuttle)

Offer, walk, cancel. The interaction stays live across the walk; the provider-anchored line formation trails the provider's moving pose.

```yaml
offer: {interaction: SERVICE, target: escort_ride, offer: true,
        min_participants: 2, max_participants: 2, queueable: false,
        formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}}
ride:  {kind: go_to, target_pose: {x: 7.0, y: 0.0}}
drop:  {cancel: true}
```

Slot-0 of the line formation is the provider itself; the formation emitter skips it so the provider's own NAVIGATE (from `GoToNode`) isn't overwritten.

### BT-controlled teardown

When the scenario needs the BT to decide end-of-interaction (e.g. need-driven break-off), chain seek -> hold -> cancel:

```yaml
talk:   {interaction: TALK_TO}
wait:   {duration: {mean: 5.0}}      # pure-wait hold; formation still pins
close:  {cancel: true}
```

### Cancel mid-sequence

`{cancel: true}` emits STOP on `movement.interaction_id` with `reason=CANCELED`. If `interaction_id` is `None`, falls back to force_stop (target=-1); IM no-ops if the agent isn't in any interaction.

## Adding a node

1. Pick the module whose concern matches; add a new module and import from `__init__.py` if none fits.
2. Subclass `py_trees.behaviour.Behaviour`. Keep state under `self._` attributes and reset in `initialise()` - nodes are constructed once and re-ticked.
3. If the node emits a `HighLevelCommand`, pick once-vs-every-tick per the rules above. Movement-owning nodes (`GoToNode`, `HoldNode`, `BlockNode`, bare `AttentionNode`) re-emit every tick and restore on `terminate()`. Signal nodes (`SeekNode`) re-emit until bound; teardown is done by a separate `CancelNode`, not `terminate()`.
4. Wire it into [../compiler.py](../compiler.py) - most primitives are constructed inside the step-builder helpers.
5. Cover it in [tests/unit/test_node_primitives.py](../../../../tests/unit/test_node_primitives.py) or a sibling unit test module.
