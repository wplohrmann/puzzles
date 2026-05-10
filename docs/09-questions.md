# Design choices

The non-default calls underpinning the rest of the architecture, and
the things deliberately left out of v0.

## Core choices

### Bottom-up construction with value-aware embeddings

The search builds a pool of fully-formed evaluable nodes from the
leaves up — one node per step (literal, primitive ref, or `App` of two
pool nodes), with the network (M4 onwards) seeing each node's actual
values on the task's examples. This is the BUSTLE / DeepCoder framing.

Trade-offs:
- Larger per-step action space than top-down. Hash-cons identity is
  the only structural prune; value-based pruning is the M4 prior's
  job.
- Embedding caches split: a structural part is cacheable across
  tasks, a value part is cacheable only within a task. Per-edit cost
  is O(1) in graph size, O(n) in example count.

### Combinator-style by default; lambdas optional in the IR

DAG sharing of arbitrary lambda terms is subtle (free variables move
scope when shared). The default is combinator style: no free
variables in the top-level program, higher-order arguments are passed
by reference (`PrimRef` or library entry). Explicit `Lambda` nodes
exist in the IR with de Bruijn indices and the evaluator handles
them, but the search doesn't propose bare lambdas as actions.

### No static type system

Nodes carry only structural information. Type errors surface as
`Value::Bottom` at evaluation. Library extraction (M3) constrains
anti-unification holes by runtime `Value` variants observed at hole
positions. See [`01-language.md`](./01-language.md).

### Two-component embeddings

Each node has a structural embedding (kind + child embeddings, plus
PrimId for primitives) and a value embedding (the node's runtime
values on the task's examples). They combine into the final node
embedding; policy and value heads see both. See
[`02-neural.md`](./02-neural.md).

### Best-first beam, MCTS later

Best-first beam guided by the policy prior is the default search.
MCTS is the planned alternative for harder tasks where the value head
can prune large branches; it's a benchmark comparison, not the
default.

### Total/strict semantics; no IO; finite fuel

Strict evaluation with a fuel counter and total primitives (`fold`,
`unfold` for recursion). General recursion via Y is foreclosed; in
return the search is well-behaved.

### `Bottom` instead of `Maybe` everywhere

When primitives fail (`head []`, `div 1 0`, `add Bool Int`), the
program returns `Bottom` and the task scores it as not-solved. This
avoids forcing every list-touching program to thread `Maybe` types,
which would balloon node counts at the cost of a tiny amount of
expressiveness.

## Risks to know about

- **Cache invalidation across library updates.** Every `LibraryRef`
  embedding becomes stale when a primitive's body changes. We track
  dependencies; getting it wrong silently degrades training.
- **MDL counting subtlety.** Counting library bodies into the score
  lets the system trade program-savings against library-cost — but
  only if novel primitives count once globally, not once per use. We
  compute total-corpus size including library, not per-program.
- **Dreams over a young library** bias the network, which biases the
  next wake's search, which biases the next replay buffer. Audit
  logging and library GC are the levers.
- **ARC-AGI ambition vs. v0 reality.** ARC is unusually hard for this
  approach. List/string is the v0 success criterion; ARC is v1.

## Standing decisions

1. **Neural framework: `tch` on Apple Silicon (MPS).** Trait-bounded
   `Tensor` so we can swap to `burn`/`candle` later without churn.
2. **Search: best-first beam first; MCTS later** as a benchmark
   comparison. Both share the same bottom-up state and action space.
3. **Lambdas: combinator-style by default; explicit `Lambda` nodes in
   the IR.** Search doesn't propose bare lambdas in v0.
4. **Floats and gradient-based literal optimisation: deferred** until
   after the list/string milestones.
5. **ARC is the real goal.** List/string is the warm-up.
6. **Hardware: M5 Pro, 48 GB, single-machine.** No distributed
   training in v0.
7. **Online vs offline: tunable.** The training loop runs as long as
   you tell it to; checkpoints + a "freeze-and-export" path support
   either mode behind a flag.
8. **Determinism: nondeterministic if it's significantly faster.**
   `--strict-determinism` flag for tests; not the default.
9. **Replay buffer: unbounded for now**, with a `Filter` hook to drop
   tasks (by age, family, size) later without restructuring.
10. **No static type system.** Nodes carry no type field; mismatches
    surface as `Value::Bottom`.
11. **Runtime value variants in v0: `Int, Bool, Float, Char, Pair,
    List, Closure, Bottom`.** Trees, dicts, sets are encoded
    (`Tree<T> ≡ Pair<T, List<Tree<T>>>`); sum types and ADTs
    deferred. Recursion is via `fold`/`unfold`, not user-defined
    fixed points.

## Out of scope for v0

- Distributed/multi-machine training.
- Theorem proving / dependent types.
- An interactive UI.
- Probabilistic programs / stochastic primitives.
- Tasks that interact with an environment.
- Multi-language program output.
