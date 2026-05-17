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

1. **Dream training.** Sample a program ρ from the prior under the
   current library; synthesise inputs and run ρ to produce a fake task.
   Walk ρ in canonical bottom-up order (a topological sort of its DAG
   nodes); each prefix is a state, the next node is the action.
   Dreams keep the network calibrated to the prior under the current
   library, and are the sole training signal until the wake phase
   starts feeding the replay buffer.
2. **Replay training**, once the wake phase produces solved-search
   trajectories. These give higher-quality (task, program, trajectory)
   tuples than dreams — the search's actual exploration was hard, so
   the training signal is sharp. Mixed with dreams in a tunable ratio.

### Training signal from one dream

For a dream `D` with bottom-up trajectory `[S_1, …, S_T]` (the topo
sort of `D`'s DAG, restricted to non-seed nodes), at each step `t`:

- `pool_t` = seeds ∪ {S_1, …, S_{t-1}}
- `S_t = App(f_t, a_t)` for some `f_t`, `a_t` already in `pool_t`.
- **Positive pair**: `(f_t, a_t)`.
- **Candidate set `C_t`**: the positive plus `K` sampled negatives from
  `pool_t × pool_t \ {(f_t, a_t)}`.

The loss is softmax cross-entropy over `C_t`'s `q` logits with the
positive pair as the target:

```
loss_t = -log( exp(q(f_t, a_t)) / Σ_{(f, a) ∈ C_t} exp(q(f, a)) )
```

See [`02-neural.md`](./02-neural.md) for the negative-sampling policy
(uniform / hard mix, default `K = 64`).

Backprop flows through every leaf embedding, `app_net`, the per-example
projection `phi`, the cross-attention parameters, and the `q_head` MLP.
Optimizer: Adam.

### Loop structure

```
for epoch in 1..dream_epochs:
    for batch in dream_loader:           # plus replay_loader, once wake feeds it
        net.zero_grad()
        for dream D in batch:
            for step t in D's bottom-up trajectory:
                sample C_t, accumulate loss_t
        total_loss.backward()
        opt.step()
```

After each training round:
- `Neural.publish_weights(new_version)`
- The structural cache is dropped wholesale (model-version key changed)
- Per-task value / target / `q` caches are dropped at task end anyway

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
    pub model_weights: Vec<u8>,         // serialised network state
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
