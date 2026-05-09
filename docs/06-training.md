# Training loop (wake / sleep)

## The cycle

```
loop:
    wake_phase()              # solve as many tasks as possible
    abstraction_sleep()       # mine the corpus, grow the library
    dream_sleep()             # train the recognition network on replays + dreams
    evaluate()                # measure on held-out tasks
    checkpoint()
```

Each iteration touches every subsystem. The phases are intentionally
separated so each can be developed and tested independently.

## Wake phase

```
for each task in current_batch:
    result ← Search.solve(task, lib, neural)
    if result.solved:
        replay_buffer.push((task, result.program, result.trajectory))
    metrics.record(task, result)
```

- Tasks come from the current curriculum: a mix of training pools across
  enabled families, plus periodically the hold-out for evaluation only.
- `Search.solve` is the function from `03-search.md`, fully wired with the
  current library and neural weights.
- Time budgets are per-task. Failed tasks are not punished beyond their lack
  of contribution to the replay buffer; we'll retry them next iteration when
  the library and network have improved.

The wake phase is the only phase where the library and network are read but
not written. It's also the only embarrassingly-parallel phase.

## Abstraction sleep

```
new_lib, rewritten_corpus, audit ← Library.compress(replay_buffer.corpus())
replay_buffer.update_corpus(rewritten_corpus)
publish(new_lib)            # signals other components to invalidate caches
```

Triggers:
- After every wake phase, **if** the replay buffer has changed materially
  (e.g. ≥ N new programs since last sleep). Otherwise skip.
- Always once at the start (run on the seed library to no-op, exercising
  the path).

The audit log is logged and ideally reviewed by a human during early
training. It's our best leading indicator that the system is doing
something sensible. ("Today's dreams added `flip`, `compose-three`, and
`sum-where`.")

## Dream sleep

Two sub-phases that share an optimiser step:

1. **Replay training.** Each `(task, program, trajectory)` in the buffer
   gives a sequence of `(state, candidates, action_taken,
   on-winning-path?)` tuples. Policy target: one-hot on the action taken
   at each step on the winning path; the same step on losing branches is
   not used as a positive policy example but does feed the value target.
   Value target: 1 on the winning path, 0 elsewhere.
2. **Dream training.** Sample a program ρ from the prior under the
   current library; synthesise inputs and run ρ to produce a fake task.
   Construct ρ in canonical bottom-up order (a topological sort of its
   DAG nodes); each prefix is a state, the next node is the action.
   Targets are derived the same way as replays. Dreams are essential
   because they give the network value-encoding signal in tasks the wake
   phase didn't reach.

The two sub-phases are mixed in a 1:1 (or 1:N — tunable) ratio inside each
mini-batch so the network doesn't drift toward either.

```
for epoch in 1..dream_epochs:
    for batch in mix(replay_loader, dream_loader, ratio):
        loss ← policy_xent(batch) + value_mse(batch) + reg
        loss.backward(); opt.step()
```

After dream sleep:
- `Neural.publish_weights(new_version)`
- `EmbeddingCache.invalidate_all_for_old_version()`

## Evaluation phase

Run search on the hold-out tasks of every enabled family with **no replay
recording and no time-budget bonus**. Report:

- Pass rate per family (fraction solved).
- Average solution size.
- Average wall time to solve.
- Library size, average primitive arity, fraction of new entries used in
  this iteration's solutions.

These are the dashboards we actually look at to decide whether the system
is learning. Ablations to drive into the dashboard early:

- **No-network**: search with prior only, no neural guidance. Establishes
  the baseline.
- **No-library-growth**: freeze the library at the seed, see how much the
  network alone buys us.
- **No-dreams**: train only on replays, see if dreams matter.

## Checkpointing and reproducibility

Each iteration emits a checkpoint that captures *everything* needed to
resume:

```rust
pub struct Checkpoint {
    pub iteration: u32,
    pub library: Library,
    pub model_weights: Vec<u8>,         // serialised tch tensors
    pub replay_buffer: ReplayBuffer,
    pub rng_state: RngSnapshot,
    pub metrics_history: Vec<IterationMetrics>,
}
```

Determinism is non-negotiable for research velocity: with a fixed seed and
a fixed config, the system runs end-to-end identically. This means:

- All RNGs are seeded explicitly.
- Search uses a deterministic priority queue tiebreaker (e.g. structural
  hash).
- Floating-point reductions in NN training are accepted as nondeterministic
  but tracked separately (we have a "strict-determinism" flag for tests
  that disables NN training).

## Compute and scaling notes

The MVP runs end-to-end on a workstation with a single GPU:

- Initial library: ~30 primitives.
- Replay buffer: a few hundred programs.
- Dreams per epoch: ~10k samples.
- Network: small (`d=128`, a few MLPs) — under a million parameters.
- Wake phase: a few minutes per iteration on list/string tasks.
- Dream phase: ~10 minutes per iteration.

When we scale up: dreams are the easiest scaling lever (parallelise the
program sampling and evaluation across CPU cores; the network training on
GPU consumes them as a stream). Search throughput is the next lever:
batched policy inference is the bottleneck and benefits from bigger batches.
