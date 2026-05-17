# Roadmap

A milestone plan ordered so each step yields a runnable, testable artifact.
The point is to delay the most uncertain pieces (network, ARC) until
prerequisites are solid, and to have an end-to-end demo as early as possible
even if it's tiny.

## Repository layout

A Cargo workspace, replacing the current `graph-seek/` crate:

```
puzzles/
├── crates/
│   ├── lang/        # IR, types, evaluator
│   ├── neural/      # embedding net, heads, training
│   ├── search/      # best-first + MCTS option
│   ├── library/     # compression
│   ├── tasks/       # task trait + per-family adapters
│   ├── training/    # wake/sleep loop, replays, dreams
│   └── cli/         # binaries (graph-seek-train, graph-seek-inspect)
├── docs/            # this directory
├── ARC-AGI/         # submodule
├── solutions/       # existing Python ARC solutions, kept for reference
└── benches/
```

## Milestone 0 — workspace skeleton (½ day)

- Convert `graph-seek` into a workspace; create empty crates with `lib.rs`
  stubs and integration crate boundaries.
- One end-to-end CI workflow that runs `cargo check` and `cargo test`.
- Migrate the existing parser experiment into `crates/lang/src/parser.rs`
  for now (we'll likely throw it away).

Acceptance: `cargo test -p lang` passes the empty test.

## Milestone 1 — `lang` end-to-end (1 week)

- IR, hash-cons arena, constructors.
- Strict evaluator with fuel and `Value` type.
- Initial built-ins (numeric, list, conditional, higher-order list
  ops, `K` and `B` combinators).
- Property tests + a handwritten suite of small programs.

Acceptance: every program in the test suite evaluates correctly;
round-trip serialisation works.

## Milestone 2 — `tasks` and a no-NN search (1 week)

- `Task` trait + `ListExamplesTask` (programmatic generator).
- Bottom-up size-iterative enumeration. No neural guidance, no
  value-based pruning beyond hash-cons identity.

At this point we have a vanilla un-guided program synthesiser. The
trivial list bench (`cargo bench --bench trivial_list`) characterises
its speed limits; size-7 programs land quickly, size-11 and beyond
take orders of magnitude longer. Closing that gap is the M4 neural
prior's job — search-time speed is not an M2 acceptance criterion.

Acceptance: `cargo test` passes (the search pipeline runs end-to-end
on identity); the bench produces useful numbers as a baseline.

## Milestone 3 — `neural` skeleton + cache (1.5 weeks)

- Framework: pure-Rust hand-rolled MLP + Adam.
- Implement the embedding network (`app_net`, leaf tables,
  `embed_value`), the per-example projection, the attention pooler, and
  the `q_head`.
- Implement the embedding caches + invalidation hook on weight-version bump.
- Wire `q(f, a)` into search; run with random network weights.

Acceptance: search runs end-to-end with random network weights, no
correctness regressions vs Milestone 2 (the random network just adds
useless guidance, but doesn't break anything); per-step neural cost is
constant in pool size given a warm cache.

## Milestone 4 — `training` + dreams (2 weeks)

- Dream sampler (PCFG over the seed library).
- Bottom-up trajectory extractor (canonical topo order of a dream's DAG).
- Curriculum that ramps dream program size 1 → 13.
- Best-first search wired through the trained `q(f, a)`.
- Per-iteration evaluation harness against
  `crates/search/benches/trivial_list.rs`.

This is the milestone where the system first *learns*. With dreams as the
sole training signal (no wake phase yet — wake plus library land in M5),
the network's job is to score candidate next-nodes well enough to navigate
the size-13 action-space cliff that un-guided enumeration hits.

Acceptance: a guided search using the trained network solves every program
in the trivial-list bench (identity, sum, head, length, add-one-to-each)
within its current per-task time budget — including `add-one-to-each` at
size 13, which un-guided enumeration cannot reach.

## Milestone 5 — `library` + abstraction sleep (1 week)

- Pattern mining, anti-unification, beam compression.
- Audit log + serialisation.
- Test against synthetic planted-fragment corpora.
- Wire a one-shot abstraction sleep onto the M4 replay buffer.

Acceptance: on a hand-built corpus where the optimal new primitive is
known, the algorithm finds it; the M4 search then solves a strict
superset of tasks because the library is bigger.

## Milestone 6 — string editing tasks (1 week)

- Char/string primitives in `lang`.
- A string-task generator inspired by the FlashFill / DreamCoder text suite.
- Iterate; observe what primitives the system extracts.

Acceptance: pass rate ≥ 50% on a hand-curated 50-task string suite within
10 iterations.

## Milestone 7 — ARC primitives + first ARC pass (open-ended)

- `Grid` type, grid primitives derived from `solutions/`.
- ARC task adapter.
- Run the system on ARC training tasks; measure.

ARC is open-ended; expectations are low for v0. The win condition for this
milestone is *infrastructural*: ARC tasks load, run end-to-end through the
loop, and at least a few solve.

## Beyond v1

Items deliberately deferred:

- Version-space algebras for compression. Replace the bounded-fragment miner.
- MCTS variant of search; ablation study against best-first.
- Float literal optimisation via gradient descent through differentiable
  primitives.
- Distributed training (multi-GPU dream throughput).
- A small "interpreter visualisation" UI that draws DAGs as you edit them.

## Risk register

The areas most likely to surprise us, with mitigations:

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Cache invalidation bugs in `neural` | High | Property test + always provide a `--no-cache` flag for differential tests |
| Library extraction grows pathologically | Medium | Audit log + GC; cap arity; reject equivalent-to-existing primitives |
| Float / numerical instability in eval | Medium | Defer floats until after milestone 6 |
| Hand-rolled NN gradient bugs | Medium | Numeric-gradient unit tests on every layer; small enough fan-in to spot-check |
| ARC turns out to need totally new abstractions | High | Plan ARC as v2; v1 success criterion is list/string |
| Replay buffer not large enough to drive learning | Medium | Dreams compensate; tune ratio. With reordered M4, dreams are the *only* training signal until M5+ wakes — extra mitigation: curriculum ramp |
| Action space without static types is too large | High | M2 baseline already shows the cost; M3 neural prior is the planned mitigation |
