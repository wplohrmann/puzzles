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
- Initial built-ins (numeric, list, conditional, higher-order list ops,
  `K` and `B` combinators).
- Property tests at Layer 1.
- Reference Python evaluator for differential tests.
- A handwritten test suite of 10–20 small programs (sum, reverse, sort, …)
  that all evaluate correctly.

Acceptance: any program in the test suite evaluates correctly; round-trip
serialisation works.

**Note on the type system.** The original M1 plan included Hindley-Milner
type inference. We shipped that, then stripped it out at the start of M2
when it became clear that the static-type machinery was both costly to
maintain and *too restrictive* to support tasks that legitimately route
through `Bool` or `Pair` intermediates. Runtime type errors now surface
as `Value::Bottom`. See `docs/decisions/m2-strip-static-types.md`.

## Milestone 2 — `tasks` and a no-NN search (1 week)

- `Task` trait + at least one generator (`ListExamplesTask`, programmatic).
- BUSTLE-style bottom-up size-iterative enumeration. No static types,
  no neural guidance, uniform priors.
- Observational-equivalence dedup over runtime values, with a
  probe-based extension for closure-typed entries.

At this point we have a vanilla untyped-enumeration program synthesiser.
Empirically (release build, single-machine):

| Task               | Size | Solve time         |
|--------------------|------|--------------------|
| `identity`         | 1    | < 0.1 ms           |
| `sum`              | 7    | ~ 30 ms            |
| `head`             | 7    | ~ 30 ms            |
| `length`           | 11   | ~ 15 s             |
| `add-one-to-each`  | 13   | does not solve in 60 s |

Acceptance: 4 of the 5 list tasks solve under their budgets;
`add-one-to-each` is run as a *characterisation* test only — its
13-node program is past the boundary of un-guided enumeration on this
primitive set, and it goes on the M5 evaluation suite as a target the
wake/sleep loop should crack once neural guidance and library growth
are in place.

The roadmap originally promised "5 of 5 within 10 s" with type-driven
pruning carrying the load. After stripping static types we honestly
can't deliver size-13 inside 10s without a smarter prior, and trying
to hard-wire one would re-introduce exactly the brittleness that
motivated removing types. M5 is the right place for the speedup.

## Milestone 3 — `library` + abstraction sleep (1 week)

- Pattern mining, anti-unification, beam compression.
- Audit log + serialisation.
- Test against synthetic planted-fragment corpora.
- Wire a one-shot abstraction sleep onto the corpus from Milestone 2.

Acceptance: on a hand-built corpus where the optimal new primitive is
known, the algorithm finds it; the search in Milestone 2 then solves a
strict superset of tasks because the library is bigger.

## Milestone 4 — `neural` skeleton + cache (1.5 weeks)

- Choose framework (probably `tch`); set up the `Tensor` trait wall.
- Implement the embedding network and heads.
- Implement the embedding cache + invalidation.
- Property test the cache equals no-cache.
- Wire the policy head into search; run with random network weights.

Acceptance: search runs end-to-end with random network weights, no
correctness regressions vs Milestone 2 (the random network just adds
useless guidance, but doesn't break anything); cache hit rate ≥ 90% on a
representative search.

## Milestone 5 — `training` + dreams (2 weeks)

- Replay buffer, dream sampler, training loop.
- Per-iteration evaluation harness.
- The full wake/sleep loop, runnable end-to-end.
- Metrics dashboard (jsonl + a tiny plotting script).

This is the first iteration where the system can *learn*. Expected first
result: on a list-task pool, pass rate climbs over the first 5 iterations,
library grows, network policy stops being uniform.

Acceptance: after 5 iterations on a 200-task list pool, pass rate is at
least double the no-NN baseline.

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
| `tch` ergonomic friction | Medium | Trait-bounded `Tensor` so we can swap to `burn` |
| ARC turns out to need totally new abstractions | High | Plan ARC as v2; v1 success criterion is list/string |
| Replay buffer not large enough to drive learning | Medium | Dreams compensate; tune ratio |
| Action space without static types is too large | High | M2 baseline already shows the cost; M4 neural prior is the planned mitigation |
