# Behavior tree nodes

Cross-node invariants enforced by the compiler live here. For the package layout and per-node authoring rules, see [nodes/README.md](nodes/README.md).

## One primitive for interactions: `SeekNode`

There are no `ADVERTISE` / `accept` phases anymore - `AdvertiseInteractionNode` and `AcceptInteractionNode` are gone, and so is `InteractionType.FOLLOW` (collapsed into `SERVICE` with `offer: true` plus a line formation anchored on the provider).

Every interaction is driven by one BT node: `SeekNode`. It emits `SEEK`; the interaction manager (via `_handle_seek` and the per-`HandleKind` strategy in [../../core/interaction_kinds.py](../../core/interaction_kinds.py)) either joins a matching open interaction or, if creation is allowed for this `InteractionKind`, creates one. `CancelNode` tears an interaction down (STOP with `reason=CANCELED`).

### Symmetric seek migration (`HandleKind.NONE`)

For symmetric types (peer-to-peer, no service tag or object anchor - e.g. `GROUP_CONVERSATION`, `TALK_TO`), a solo 1p FORMING owner does **not** short-circuit on its own interaction. `seek()` still runs the strategy's `find()` each tick so a better-populated peer can take over. If a peer's matching interaction is discovered, the agent is silently removed from its own FORMING via `_detach_quiet` and `accept`ed into the peer's - no `INTERRUPTED` is emitted on `BehaviorTreeMovement.last_outcome`, so `SeekNode` stays `RUNNING` through the handoff instead of failing. If the detached interaction ends up empty with no queue, it collapses to `INTERRUPTED` as bookkeeping but never reaches the BT. Asymmetric types (`TAG` / `AGENT` / `OBJECT`) short-circuit on the agent's own FORMING - that interaction is their offer or reservation, not a placeholder to swap out of.

## Compiler dispatch

| Flags on the step | Compiled inner sequence |
|---|---|
| `cancel: true` | `ClearOutcome -> Cancel` |
| `interaction:` one of `USE` / `SIT_ON` / `LIE_ON` / `QUEUE_USE` with `target:` | `ClearOutcome -> Resolve -> GoTo -> Seek -> Satisfy?` |
| `interaction:` (any other, or object-bound without `target:`) | `ClearOutcome -> Seek -> Satisfy?` |
| `interaction: BLOCK` with `target: <int>` | `ClearOutcome -> Block` (pursuit + SEEK on arrival) |
| `kind: go_to, target_pose:` | unchanged - nav to pose |
| `kind: go_to, target:` | `ClearOutcome -> Resolve -> GoTo` |
| `kind: attention` (or `attention:` alone) | `ClearOutcome -> Attention -> Satisfy?` (bare: halt, face, drive every channel) |
| any of the above plus `attention:` | same inner sequence, plus an `AttentionNode` rider as a third `Parallel` child |
| only `duration:` (pure wait) | `ClearOutcome -> Hold` |
| `autonomous: true` | `AutonomousNode` that scores `actions:` while idle and runs the winner to its end, each action compiled like the step it stands for (a symmetric interaction with `target:` walks there first) |

Every compiled step is wrapped in a `Parallel(SuccessOnOne)` with a `PatienceWatchdogNode` - step-level patience spans every phase below. The parallel also stops its still-running children (watchdog, rider) when the inner sequence ends, so a rider releases its `hold: release` channels at step end while `hold: keep` channels stay on `movement.gestures` until a later node takes the slot or `SequenceStateMachine` leaves the sequence. A sequence with `attention:` is wrapped in a `SequenceRiderNode` that ticks the sequence rider next to the current step, suspending it while the step has its own `attention:` or is autonomous. `attention:` is rejected on autonomous steps and inside `actions:`.

## Patience phases

For an interaction step, patience covers:

1. **Nav to within `interaction_radius`** (object-bound steps only - the `Resolve -> GoTo` prefix).
2. **Seek emit** - `SeekNode` issues a SEEK command; IM either joins an existing interaction or creates a new one.
3. **Wait for `ACTIVE`** - `SeekNode` stays `RUNNING` until `is_bound(aid)` is true (contract reached `min_participants`).
4. **Hold** (optional) - if the step has `duration:` without an interaction, a `HoldNode` pins the agent at the current pose via NAVIGATE-to-self. With `interaction:` + `duration:`, the duration threads to the contract and IM tears the interaction down with outcome `COMPLETED`.
5. **Cancel** - either BT-controlled via a follow-up `{cancel: true}` step, or contract-timed by IM.

A step with `patience: {mean: 5.0}` and a target 20 m away may fail during phase 1 before ever emitting SEEK. Non-interaction steps are symmetric: patience caps nav + duration.

## `interaction_radius`

Cascade: step override > object override > per-`InteractionKind` default (see [../../core/interaction_kinds.py](../../core/interaction_kinds.py)) > `DISTANCE_TOLERANCE`. Full table and authoring examples in [../../../config/scenarios/README.md](../../../config/scenarios/README.md).

## Tunes

[../agents/tune.py](../agents/tune.py) holds the overlay. `agent.tunes` keeps the spawn params as base plus `{key: (level, TuneDef)}`. `retune(agent, key, tune | None, level)` folds the active tunes per field over the base in level order (`CONDITION`, `SEQUENCE`, `STEP`, `NODE`, then the order taken) into `agent.params` and `state.desired_velocity` and marks `agent.tunes.dirty`, which the manager answers in the decide phase with `AgentPool.update_agent` (pool rows rewritten, `on_pool_update` on the pool extensions). It is idempotent on unchanged content. Owners: `SequenceStateMachine` holds the sequence's `tune:` under the key `sequence` across every `_goto` and releases it in `terminate()`, a `TuneNode` sibling in the step parallel holds a step's `tune:`, `ConditionalTunesNode` at the root holds each `tunes:` entry of the type while its `when` holds on the needs, and `BlockNode` holds its pursuit boost under its own name. Nodes never write `agent.params` or `state.desired_velocity` directly, they take and release a key, and code that replaces the spawn params at runtime (`_reseat_agents`) goes through `rebase`. Schema in [../../../config/agent_types/README.md](../../../config/agent_types/README.md#sequences).

## Signals on `BehaviorTreeMovement`

IM writes `interaction_id` and `last_outcome`; BT nodes read them. `ClearOutcomeNode` at the head of every compiled step resets them so stale signals don't short-circuit a re-entry. Full rules and outcome encoding in [nodes/README.md](nodes/README.md#authoring-rules).

For compile-shape examples (symmetric peer, object-anchored, mobile provider, BT-controlled teardown, mid-sequence cancel), see [nodes/README.md](nodes/README.md#bt-flow-idioms).
