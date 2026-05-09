# Testing strategy

Testing is the lever that lets a system this big actually work. The plan is
not "lots of unit tests" but a layered strategy where every subsystem has at
least one test type that *cannot pass* unless the subsystem is correct.

## Layer 1 — Property tests (the cheap, deep net)

Use `proptest` (Rust) to fuzz invariants that should hold for *any* input.

Per subsystem:

- **Language**
  - Generate random typed programs. Check: round-trip serialise/deserialise
    yields the same program; hash-cons inserts are idempotent; type-checking
    of every constructed program succeeds; alpha-equivalent terms have the
    same `NodeId`.
  - Generate random programs and random inputs. Check: evaluation is
    deterministic (same fuel → same result); evaluation respects sharing
    (a program with shared subterms produces the same value as the
    fully-tree version).
  - Generate random programs of total functions. Check: evaluation
    terminates within fuel proportional to size.

- **Library**
  - Generate a synthetic corpus where the optimal compression is known by
    construction (e.g. plant a recurring fragment in 30 programs). Check:
    the algorithm finds it within budget.
  - Generate a corpus with no compressible structure. Check: the algorithm
    refuses to grow the library.
  - Generate `(corpus, libA)` and `(corpus, libB)` where `libB ⊃ libA`.
    Check: total score under `libB` ≤ score under `libA` *only if* `libB`'s
    extras are actually used.

- **Search**
  - Generate small tasks with a known minimal solution (the planted-program
    test). Check: search finds an equivalent program of size ≤ minimal +
    epsilon, given enough budget.
  - With prior set to uniform, check that search reduces to vanilla typed
    enumeration and finds the planted program.
  - With the network's policy mocked to always emit the right action,
    check that search finds the planted program in N steps where N = size.

- **Neural**
  - The cache invariant: for any DAG, `embed_with_cache(D) == embed_without_cache(D)`
    bitwise (in eval mode). Property test over random DAGs.
  - Permutation invariance of the task encoder: for any (input, output) set,
    permuting the order doesn't change the encoding.

## Layer 2 — Differential / golden tests

For each task family we maintain a small set of (task, expected canonical
program) pairs. These are tests, not training data.

- The interpreter is differentially tested against a hand-rolled *reference*
  evaluator written in pure Python (or Rust, whatever, just unrelated code
  paths). For each test program in the suite, both implementations agree on
  outputs. This is the single best defence against subtle interpreter bugs
  that would silently corrupt training.
- For library extraction we have golden corpora — small bundles of programs
  with the *expected* output library (curated by inspection). Test that the
  algorithm produces it (or something with equal or better total score).
- For search we have a "smallest program for this task" oracle for a
  handful of trivial tasks (`identity`, `successor`, `reverse-2-element-list`)
  and check the algorithm returns programs of correct size given budget.

## Layer 3 — Integration tests

Run the full wake-sleep loop on a pre-canned tiny task family with a fixed
seed and assert end-state metrics:

- After 5 iterations on `tiny_list_tasks`, pass rate ≥ X.
- After 1 iteration the library has grown by ≥ 1 primitive.
- The replay buffer contains ≥ K programs of size ≤ S.

This is slow (~minutes), so run on CI but not pre-commit. The seed is fixed
and we have a recorded "expected metrics" file; deviation triggers
investigation, not automatic failure (it's a noisy test by nature, so it
fails *bands*, not exact values).

## Layer 4 — Benchmark suite

A standing benchmark report, run nightly:

- Pass rate on each enabled family's hold-out set.
- Wall-clock per iteration.
- Average solution size.
- Library evolution (size, churn).

Track these in a flat-file metrics database (`metrics.jsonl` per
iteration) so we can plot trends across many runs. This is research
infrastructure, not a test, but we treat it like one — surprise regressions
get investigated.

## What we do not test

- We do not unit-test individual neural-network ops. The framework
  (`tch`/`burn`) is responsible for those. We test the integration only.
- We do not test specific library entries (their identity is not
  deterministic across seeds). We test invariants about *what the library
  achieves*.
- We do not test exact training trajectories. We test bands.

## Tooling

- `cargo test --workspace` runs Layer 1 + Layer 2 (under a minute).
- `cargo test --workspace --features integration` runs Layer 3 (minutes).
- `cargo bench` and `tools/run_benchmarks.rs` run Layer 4 (longer).
- `criterion` for microbenchmarks of hot paths (interpreter, hash-cons, NN
  cache hit-rate).
- A debug CLI: `graph-seek inspect <ckpt>` prints the library, replay
  highlights, and recent audits — this is the day-to-day debugging
  interface and worth investing in early.

## Reproducibility checklist

Every test that exercises non-deterministic components (search, training)
must:

1. Take a seed parameter.
2. Set every RNG (proptest, rand, NN init, search tiebreaker).
3. Be runnable single-threaded if `--strict-determinism` is set.
4. Save the input task pool, library, and network weights as artifacts
   when it fails, so we can re-run exactly.
