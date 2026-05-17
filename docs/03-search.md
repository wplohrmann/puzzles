# Search

## Goal and framing

Given a `Task` (an oracle scoring programs), find a node whose values
on the training examples match the task's targets, with as small a
program as possible, within a fixed compute budget.

We construct programs **bottom-up**: each step grows a pool by one
node, formed either from a literal, a primitive reference, or an
`App(f, a)` of two pool nodes. Every node, the moment it joins the
pool, has a concrete value per training example, which the network
uses to score candidates. Solution detection is a
free side-effect of evaluating each candidate: the moment its runtime
values match the task's expected outputs on every example, we're done.
`Value::PartialEq` is variant-strict, so value-equality alone is
sound.

There is no partial program with holes. The pool is a set of
fully-formed evaluable nodes, and a "solution" is any pool node whose
values match the task's targets.

## State

```rust
pub struct Pool {
    entries: Vec<Entry>,
    by_node: FxHashMap<NodeId, usize>,   // hash-cons-canonical dedup
    by_size: Vec<Vec<usize>>,            // entries indexed by program size
}

pub struct Entry {
    node:   NodeId,
    size:   u32,
    values: Vec<Value>,                  // one per task example
}
```

The pool is a single growing structure shared across all enumeration
steps; it is not cloned per beam-state.

## Actions

Three action shapes seed and grow the pool:

- **Literal** — a small fixed seed set (`0`, `1`, plus anything the
  caller passes via `SearchConfig::literal_seeds`). Future work will
  add value-copy proposals from the task's examples.
- **PrimRef** — every primitive in the library, added once at size 1.
- **App** — every `(f, a)` pair from the pool. Without a static type
  system, construction always succeeds; mismatched runtime types
  surface as `Value::Bottom` after evaluation.

Two filters reject candidates before they reach the pool:

- **Hash-cons identity.** Any `App(f, a)` whose `NodeId` already
  exists in the pool is filtered out at zero cost.
- **Bottom values.** Any candidate whose runtime values contain
  `Bottom` on at least one example is dropped. `apply` propagates
  `Bottom` strictly on either side, so a single `Bottom` at any
  example position taints every composition through that entry; and
  `values_match` rejects any `Bottom` in the candidate's values. Such
  an entry is therefore incapable of being on a solution path —
  filtering it out is a logical implication of the pipeline, not a
  speed heuristic.

Beyond these two structural filters there is no value-based pruning;
search speed comes from the neural prior `q(f, a)`.

## Algorithm — guided

Best-first expansion of a priority queue of candidate App-pairs, scored
by the neural network's `q(f, a)`. See [`02-neural.md`](./02-neural.md)
for how `q` is computed.

```
seed pool with Param(0), literal seeds, every PrimRef
for each seed S: compute h_struct(S), h_value(S, ·) and cache

for each (f, a) in pool × pool:
    if intern(App(f, a)) already in pool: drop
    score = q(f, a)                          # one MLP forward, cached
    push (score, App(f, a)) to frontier

while frontier not empty and budget not exhausted:
    pop (score, App(f, a)) — highest score first
    if pool already contains this canonical NodeId: skip
    values ← apply(f.values, a.values) per example
    if values contain Bottom on any example: skip (dropped)
    if values match expected: return as solution
    add (App(f, a), values) to pool
    compute h_struct, h_value for the new node and cache
    for each existing entry e in pool:
        if intern(App(new, e)) not in pool: push (q(new, e), App(new, e))
        if intern(App(e, new)) not in pool: push (q(e, new), App(e, new))
return None
```

Two design properties:

1. **`q(f, a)` depends only on `(f, a, task)`** — not on the rest of
   the pool. It is computed once at enqueue time, cached, and never
   recomputed as the pool grows. The priority queue's older entries
   keep their scores valid.
2. **Lazy evaluation.** The candidate's runtime result is computed only
   when the pair is popped — the same `apply` the unguided baseline
   does on admission, no extra work. `q` ranks pairs from `f` and `a`'s
   runtime state alone, without running the application first.

The per-pool-add work is `O(pool)` `q`-forwards (one per direction per
existing entry). The single `apply` per pop is unchanged from the
unguided baseline. Both are parallelisable across pairs and across
examples.

Time budget, frontier-size cap, and pool-size cap are checked
periodically in the loop; any of them triggers a graceful early return.

### Tie-breaking

When two candidates have equal score (rare with `f32` scores but
possible), prefer smaller `size`, then break by `NodeId` for
determinism.

## Algorithm — unguided baseline

Size-iterative enumeration without any neural prior. Used as the
ablation against which the guided search is measured.

```
seed pool with Param(0), literal seeds, every PrimRef
for size in 2..=max_program_size:
    for (k_f, k_a) with k_f + k_a + 1 == size:
        for each f in pool entries of size k_f:
            for each a in pool entries of size k_a:
                node ← arena.intern(App(f, a))
                if node already in pool: skip
                values ← apply(f.values, a.values, fuel) per example
                if values match expected: return node
                add (node, size, values) to pool
return None
```

The time budget and `max_pool_size` are checked periodically inside
the inner loop; either triggers a graceful early return.

## Solution check

`values_match(values, expected)` is the only test:

- lengths match,
- no `Bottom` in values,
- `values == expected` element-wise via `Value::PartialEq`.

There's no separate type check; `PartialEq` is variant-strict so
`Int(0)` ≠ `List([])` etc.

## Lazy `if` and incremental apply

The pool's values for an `App(f, a)` candidate are computed by
`apply(f.values[i], a.values[i], …)` per example, rather than by a
fresh `eval(arena, lib, candidate, …)`. This is O(1) per example per
add but does not preserve the lazy-`if` short-circuit that `eval`
applies when an entire `if cond then else` chain sits at the apex of
three `App`s. A candidate of that exact shape with one `Bottom`-valued
branch will Bottom-propagate in the pool view; the solution-validator
path uses `eval` and gets it right.

## Sharing

The arena is hash-consed, so two semantically-identical candidates
collapse to a single `NodeId`. The pool's `by_node` index makes the
"is it already here?" check O(1). Library entries (built-ins now,
learned primitives later) plug in identically: a primitive ref is just
a node in the pool, used via `App` like any other.

## Outputs

```rust
pub struct SearchResult {
    pub program:         Option<NodeId>,
    pub solved:          bool,
    pub size:            u32,
    pub elapsed:         Duration,
    pub final_pool_size: usize,
    pub stats:           SearchStats,
}
```

`stats` carries counts of `apps_attempted`, `apps_dup_node`,
`apps_added`, `eval_errors`, and `pool_by_size` for diagnostic use.

## What lands later

- **Trajectory recording** for the wake phase: the ordered sequence of
  (state, action, on-winning-path?) tuples generated by the guided
  search becomes higher-quality training data than the dream-only
  signal.
- **PUCT MCTS** as a benchmark alternative to best-first. `q(f, a)` is
  a natural prior for PUCT's exploration bonus.
- **Parallelism**: candidate evaluation and `q`-scoring are
  embarrassingly parallel within a search; tasks are independent
  across searches. Requires moving `Value` from `Rc` to `Arc` first.

## Performance

The trivial list bench (`cargo bench --bench trivial_list`) is the
canonical timing reference; correctness lives in `cargo test`.
