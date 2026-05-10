# Milestone 2 design decisions

> **⚠️ Partially superseded.** Several decisions in this doc were
> overturned mid-M2 by the deletion of the static type system. See
> [m2-strip-static-types.md](./m2-strip-static-types.md) for the
> details. Specifically: §5 (tycon filter), §8 (root-tycon
> pre-filter), §3 (in part — apply construction no longer fails),
> and the type-aware extended obs-eq formulation in §4 are all
> obsolete. The §1 (acceptance task list) and §6 (reverse iteration)
> material still applies; the §10 performance numbers were measured
> under the now-deleted typed regime.

A log of every non-obvious decision made while implementing the `tasks`
crate, the `search` crate, and the M1 patches that M2 depended on. Each
is reversible — call any of these out and we'll change them.

## 1. Re-aligned the M2 acceptance task list (and added combinators)

The roadmap's M2 acceptance bar was *"5 of 5 trivial list tasks
(`identity, head, length, add-one-to-each`) within 10s each"*. Two
problems with this:

- The seed library shipped in M1 had `fold/unfold/cons/nil/...` but **no
  `K, B, C` combinators and no `head, length, map`**. Combined with the
  resolved decision (`09-questions.md` #3) that the search **doesn't
  propose lambdas as actions**, three of the four named tasks are
  *unreachable* with M1's seed library: any callback that doesn't fit a
  primitive's curried shape requires a lambda.
- M1's `fold` was implemented with non-canonical "left-fold-with-args-
  swapped" semantics, which made `head` and `map` infeasible even with
  the right combinators.

We resolved both with two small M1 patches (`m1-lang.md` §16 and §17):

- Added `K` and `B` to the seed library.
- Switched `fold` to standard right-fold semantics.

With those in place, the roadmap's task set becomes expressible:

| Task              | Body                              | Size |
|-------------------|-----------------------------------|------|
| `identity`        | `P`                               | 1    |
| `head`            | `fold k 0 P`                      | 7    |
| `length`          | `fold (k (add 1)) 0 P`            | 11   |
| `add-one-to-each` | `fold (b cons (add 1)) nil P`     | 13   |
| `sum`             | `fold add 0 P`                    | 7    |

We chose `head` over `last` because it's the roadmap's named task and
both have identical size and difficulty under right-fold semantics.
We added `sum` to round out the five. (Dropped `last` to avoid
redundancy with `head`.)

## 2. `Task` trait shipped as the M2 subset only

`docs/05-tasks.md` proposes a richer trait with `target_type, solves,
score, encoding, id` and `Send + Sync` bounds. We shipped just
`target_type, solves (default impl on score), score, id`. Reasons:

- **`encoding` deferred to M4** — it's the hook for the neural
  recogniser. Adding it now means writing dead code or a stub that
  panics. YAGNI.
- **`Send + Sync` dropped** — `Value` holds `Rc`s so it's `!Send`.
  Parallel-task evaluation requires switching `Rc → Arc` first. Filed
  as a follow-up.

`solve` is currently typed against the concrete `ListExamplesTask`
rather than `&dyn Task` because the search's solution-detection path
needs the example inputs/outputs (which the trait doesn't expose). The
abstraction is fine for adding new task families later — they'll
implement `Task` for their own scoring + provide their own ListExamples-
analogue for the search to consume.

## 3. Search is BUSTLE-style size-iterative, not a priority-queue beam

`docs/03-search.md` describes a best-first beam with a priority queue,
priority `= log_prior - α·size`. We shipped instead:

- A single, monotonically-growing **pool** (not per-state).
- **Iterate by program size N from 1 upward**; at each N, enumerate
  all `App(f, a)` with `size(f) + size(a) + 1 == N`.

The two are mathematically equivalent under uniform-typed-prior (every
admissible action gets the same prior, so all states at size N have
equal priority — ranking degenerates to "smaller first"). The
priority-queue beam earns its complexity only when priors are
non-uniform, which is M4.

Per-state pools cost more than per-search pools because each beam-state
clones a HashSet per child. With the monotonic pool the structure is
just `(Vec<Entry>, FxHashMap<NodeId, idx>, by_size, by_obs)` — no
cloning per action.

## 4. Observational equivalence pruning included from day one

The search doc lists obs-eq as something to "layer in once basic
version works". We shipped it from the start because without it the
five trivial tasks won't reliably solve in 10s once `fold` is in the
action set — add-one-to-each in particular needs aggressive dedup at
size 13.

Two flavours:

- **Plain obs-eq for non-function-typed entries**. Hash key is
  `(canonical_type, value_tuple)`. Closures aren't covered: the
  `Value::Closure` case in our hash function returns a single
  discriminant byte, which means *all* closure values would collide.
  Skipping function types is the right call.
- **Extended (probe-based) obs-eq for function-typed entries** whose
  type has shape `goal_arg_ty → R` with `R` concrete (`pool::applied_obs_key`).
  We probe the closure with each example input and dedup on the
  resulting `R` values.

### Soundness caveat for extended obs-eq

Two closures C1 and C2 of type `T_arg → R` that produce identical R
values on every example input *might* differ on intermediate values
fed to them inside higher-order primitives (e.g. when used as the
callback to `fold`, which applies the closure to many distinct
intermediate accumulator values). Probe-equivalence at the goal level
does not guarantee behavioural equivalence in arbitrary contexts.

For M2 trivial tasks every solution applies its goal-shape closure
*directly* to `Param(0)` (not as a fold callback), so the prune is
sound. For broader task families this could prune a valid program.
The flag `SearchConfig::extended_obs_eq` defaults to `true`; disable
to fall back to the safe plain version.

## 5. Tycon filter, auto-derived from the goal type

`SearchConfig::restrict_to_goal_tycons` (default `true`) computes the
set of type constructors mentioned in the task's `target_type` (`Fn` is
always allowed; free vars are always allowed). Any candidate type
mentioning a forbidden tycon is dropped before adding to the pool.

Effect on the 5 trivial tasks: cuts the seed primitive set from 19
down to ~9 (drops `not, and, or, if, eq, lt, pair, fst, snd, unfold`),
and prunes the App search space proportionally. Without this the pool
saturates on Bool/Pair-typed intermediate closures that can't
contribute to a `List<Int> → R` solution.

### Caveat: drops `if`-based programs

A task whose solution legitimately needs `if cond x y` for `Int`-valued
output won't typecheck through this filter (`Bool` would be forbidden).
The 5 trivial tasks don't need `if`. Disable
`restrict_to_goal_tycons` for tasks that route via `Bool` or `Pair`.

## 6. Reverse iteration of `(k_f, k_a)` splits at each size

For each size N, splits are enumerated `k_f = N-2 … 1` (reversed from
the natural `1 … N-2`). The reason is empirical:

- Splits with **large** `k_f` (small `k_a`) tend to *complete* an
  `App(closure, simple-arg)` chain and produce concrete-typed results
  that are likely to match the goal.
- Splits with **small** `k_f` (large `k_a`) tend to *extend* a
  composition with another closure factor, producing yet more
  function-typed intermediates that explode the pool.

For add-one-to-each (size 13, solution = `App(F11, P)`), the natural
order hit pool-cap at ~2M entries before reaching split (11, 1).
Reversed order found the solution in 2.0s with pool ≈ 770K.

This is a heuristic — it could underperform for tasks where the
solution apex isn't a `(size-2, 1)` shape. Keep it under review when
new task families come online.

## 7. Incremental `apply` for App values, not full `eval`

When a new `App(f, a)` candidate is added, its values per example are
computed by `apply(f.values[i], a.values[i], …)` rather than
`eval(arena, lib, candidate, &[input], …)`.

- **Pro**: O(1) per example per add (closure manipulation), instead of
  O(node-size).
- **Con**: doesn't preserve the lazy-`if` optimisation. A candidate of
  shape `if cond x y` with one of `x`/`y` evaluating to `Bottom` will
  see `Bottom` propagate through `apply` regardless of `cond`'s value.
  `eval` (used by the `Task::score` solution-validator) gets this
  right, but candidates added to the pool see the eager-apply view.

For the 5 trivial tasks, `if` is forbidden by the tycon filter (Bool
is excluded from the goal tycons), so this divergence is moot in M2.
A future task that needs `if` should either:

- Trigger a fall-back to `eval` for App candidates whose top-level
  matches the `if` syntactic pattern, or
- Disable the tycon filter and accept the slight unsoundness on
  lazy-`if` candidates as a "false negative" in pool-mode (the
  solution check still uses `eval`, so the candidate could be
  detected as a solution at the moment it's constructed even if
  pool-mode marks it as Bottom).

## 8. Root-tycon pre-filter before invoking unification

Hot loop sees ~5–10× more `App` attempts than fit in the time budget
without a fast-path filter. Before calling `construct::app` (which
does instantiation + full unification), we compare the root
constructor of `f`'s first-arg type with the root constructor of
`a`'s type. Free variables (`Wildcard`) match anything; concrete
constructors must match exactly.

Effect: typecheck failures are still counted in stats, but the cost
of detecting them drops from "instantiate two polytypes + unify" to
"two pointer derefs + enum compare". Length went from 1.1s to 0.2s
with this change in place.

## 9. `drop_all_bottom` for non-function entries

Default-on. A candidate whose values are all `Value::Bottom` and whose
type is non-functional is skipped. It can't be a solution (expected
outputs are non-Bottom for the trivial tasks) and propagating Bottom
through subsequent applications produces only more Bottom.

Marginal win — saves ~300 entries on add-one-to-each — but cheap
to apply.

## 10. `max_pool_size` is calibrated to the hardest M2 task

Default `2_000_000`. Add-one-to-each peaks at pool ≈ 770K entries; the
2× headroom is for slop and for tasks slightly larger than the M2 set.
At ~500 bytes/entry this is ~400 MB of resident memory at peak, which
is reasonable for an M5 Pro / 48 GB target. Expect to revisit when
M5 (training loop) batches multiple searches concurrently — there
isn't headroom for many parallel searches at this pool size.

## 11. Default `eval_fuel = 200_000`

Per-evaluation, not per-search. Picked empirically: the deepest
trivial-list task (add-one-to-each) evaluates a 13-node program over
4 example inputs of length up to 4 — well under 1000 reductions per
example. 200K is generous enough that real bugs (an infinite loop in
`unfold`) hit the unfold soft-cap (100K elements) first and return
`Bottom`.

## 12. Test arena is fresh per-task

The acceptance tests build the ground-truth program in one arena and
run the search in a separate arena. This means the search has zero
information leakage from the ground-truth (which is the right
abstraction for a search test). Hash-cons across arenas is unrelated;
the search has to rediscover everything from primitives.

## 13. No priority field, no `log_prior` accumulation

The design doc proposes `log_prior` accumulated through the search
trajectory and a per-state priority. Under uniform priors these would
all be equal at a given size, so we don't compute them. When a real
policy is wired in (M4) we'll need them; the data-model change is
small (add a `priority: f32` to `Entry`).

The trajectory recording (described in `03-search.md` for training
ingestion) is also deferred to M4 since the trainer doesn't exist yet.

---

## Things deliberately not done in M2

- **No priority queue / non-uniform priors.** M4 work — meaningful
  only with a neural policy.
- **No type reachability analysis.** Would let us prune candidates
  whose result type can't reach the goal type within the remaining
  size budget. Non-trivial to implement (Var instantiation makes the
  reachability set type-dependent). The tycon filter + root-tag
  pre-filter cover most of the win for our tasks.
- **No MCTS.** Listed in the design doc as option B; defer to a
  benchmark-time decision after M4–5.
- **No multi-arg tasks.** `ListExamplesTask` assumes a single `Param(0)`.
  Multi-arg requires either explicit lambda construction in the search
  or an "uncurry" convention. Defer.
- **No parallel search.** `Value` holds `Rc`s; need `Arc` first.
- **No CLI binary.** Roadmap puts CLI in M5 with metrics/dashboard.
  M2's runnable artifact is the integration test `crates/search/tests/trivial_list.rs`.

## Things to flag for review

If any of these calls seem wrong, the most consequential ones to
revisit are:

1. **Reversed split order** is a heuristic that happens to match the
   structure of the 5 trivial tasks. New task families should be
   profiled before assuming the same ordering wins.
2. **Tycon filter as default-on**. Convenient for trivial tasks but
   silently hides any task whose solution routes through Bool/Pair.
   When we hit the first such task, we should turn the default off
   and rely on the heuristic only when explicitly opted-in.
3. **Extended obs-eq's soundness caveat**. We accepted it for M2 and
   it's necessary to fit add-one-to-each in 10s. The risk grows with
   richer task families that pass closures into HO contexts other than
   the outermost `App(_, P)`.
4. **Incremental apply (lazy-`if` divergence)**. Sufficient for M2
   (Bool is filtered) but a real correctness issue for any future
   task that needs `if` short-circuiting in the pool (vs. just at
   solution-check time, which uses `eval`).

## Performance reference (current main, release)

```
identity        :  ~10 µs   pool=1
sum             :  ~4 ms    pool=363
head            :  ~4 ms    pool=363
length          :  ~200 ms  pool=66K
add-one-to-each :  ~2 s     pool=770K
total           :  ~2.2 s
```

Acceptance bar is 10s per task; we're roughly 5× under on the worst
case and 2500× under on the easiest.
