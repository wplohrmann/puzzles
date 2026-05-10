# M2 mid-flight decision: strip the static type system

A log of the decision to remove all static type machinery from the
language, and the resulting cascade of simplifications and
performance shifts.

## Why

Three reasons converged:

1. **The auto-derived tycon filter was over-pruning.** During M2 we
   added a heuristic that restricted candidate types to those whose
   tycons appear in the task's target type. This made `add-one-to-each`
   solvable in 10 s, but at the cost of disallowing intermediate types
   like `Bool` and `Pair`. Many real tasks (especially ARC) route
   through those intermediates. The heuristic was fundamentally
   incompatible with the eventual goal.

2. **The neural recogniser doesn't need types.** The architecture in
   `02-neural.md` embeds nodes from `(structural | value)` features.
   The structural side included tycons; the value side carries the
   actual data. With enough training, the network learns
   type-equivalents implicitly — runtime values are a richer signal
   than static types anyway (they reflect distribution, range, shape).
   Hand-wired type rules duplicate work the network would do.

3. **Code cost.** The HM-lite implementation was about 400 lines in
   `ty.rs` plus another 200 across `arena.rs`, `construct.rs`,
   `library.rs`, `builtin.rs`. The search-side type machinery
   (canonical hashing, root-tag pre-filter, type-aware obs-eq, target-
   type derivation, instantiation-with-fresh-vars) added another 150.
   Removing it deleted ~600 lines net.

## What was deleted

- `crates/lang/src/ty.rs` entirely: `Ty`, `TyCon`, `TyVar`,
  `TypeScheme`, `Subst`, `unify`, `instantiate`, `canonicalize`,
  `rename`, `TyVarGen`, `UnifyError`.
- `Node::ty` field; the structural hash now mixes only kind + children.
- `NodeKind::Lambda::param_ty`; lambdas are body-only.
- `Primitive::ty: TypeScheme` → `Primitive::arity: u8`.
- `BuiltinId::ty()` → `BuiltinId::arity()`.
- `construct::app`'s entire body collapses to
  `arena.intern(NodeKind::App { func, arg })` — no instantiation, no
  unification, no fresh-var threading, no ApplyMismatch error.
- `Error::Type`, `Error::ApplyMismatch`, `Error::PrimitiveTypeMismatch`.
- All search-side type machinery: `RootTag`, `root_tag`,
  `roots_compatible`, `compute_forbidden_tycons`, `type_is_forbidden`,
  `instantiate_free`, `applied_obs_key`'s type-shape gate, the
  `restrict_to_goal_tycons` and `forbidden_tycons` config knobs.
- `Task::target_type`, `ListExamplesTask::target_type`,
  `ListExamplesTask::arg_ret`, `programmatic_task`'s `arg_ty`/`ret_ty`
  parameters.

## What still works

- **Hash-cons.** Distinct kind+children tuples get distinct NodeIds;
  identical ones collapse. PrimRef nodes for the same primitive are
  now a single node regardless of how the original code might have
  "instantiated" them — which is the right behaviour.
- **α-equivalence.** De Bruijn indices give it for free, no separate
  pass.
- **Polymorphism.** Falls out of the runtime values: `nil` evaluates
  to `Value::List([])`, which works for any element type. `cons 1 nil`
  and `cons true nil` use the same `nil` and `cons` nodes.
- **Lazy `if`.** Still detected syntactically at the apex of three
  chained `App`s in `eval`, with the `BuiltinId::If` lookup unchanged.
- **Programs serialise identically** modulo the dropped type fields;
  `KindSerial` no longer carries `param_ty` or per-node `ty`.

## What changes for the search

- **Construction always succeeds.** `app(arena, f, a)` cannot fail —
  there is no type mismatch to detect. Mismatches show up at evaluation
  as `Value::Bottom`.
- **No type pre-filter.** The `root_tag` fast-path and the full
  unification path are both gone. Every `(f, a)` pair from the pool
  is admissible at construction; we discover whether it's useful by
  evaluating it.
- **Solution check is value-equality only.** `matches_goal` collapses
  to "values match expected and none are Bottom". `Value::PartialEq`
  is type-strict (`Int(0) ≠ List([])`) so this is sound.
- **`drop_all_bottom` carries more weight.** Without static types,
  many candidates Bottom-out, but they dedup to a single all-Bottom
  pool entry via obs-eq, so the bloat is bounded.
- **Probe-based obs-eq is unchanged in spirit but stricter.** The
  function-type gate is gone (no types to inspect), but we now
  refuse to dedup any probe whose result is `Bottom` or another
  closure. Two closures that Bottom-probe might still differ when
  used in a higher-order context (e.g. `App(add, 1)` Bottom-probes
  against a `List` input but is essential as a fold callback);
  collapsing them was a real bug we hit during the M2 rewrite.

## Performance impact

Same hardware, same trivial-list test set, release build:

| Task               | M2 with static types | M2 without static types |
|--------------------|----------------------|-------------------------|
| `identity`         | 12 µs                | 16 µs                   |
| `sum`              | 4 ms                 | 26 ms                   |
| `head`             | 4 ms                 | 33 ms                   |
| `length`           | 200 ms               | 16 s                    |
| `add-one-to-each`  | 2 s                  | does not solve in 60 s  |

The slowdown at deeper sizes is real and expected — the static type
filter was doing significant pruning. Without it, the search has to
evaluate more candidates. The M4 neural prior is what closes the gap;
the typed M2 numbers were artificial in that they leaned on the
filter's over-pruning.

## Knock-on effects

### `04-library.md` — anti-unification

The original design used type-driven anti-unification: hole types
constrained which sub-trees could collapse into the same hole. We
replace that with two empirically-derived signals:

- **Runtime-value variant** at hole positions across occurrences. If a
  hole sees `Int` everywhere it's an Int hole; if it sees both `Int`
  and `List`, it's almost certainly an over-generalisation.
- **Bottom-rate** when the candidate primitive is substituted back
  into the corpus. Over-general holes Bottom out frequently; score
  them down.

### `02-neural.md` — node embeddings

The structural part of node embeddings used to factor in tycons. It
now factors in only kind + child embeddings + (for primitives) the
PrimId. The value part is unchanged. The network has slightly less
prior information to work with, but more data to learn from
(runtime values are richer than tycon labels).

### `09-questions.md` #3 and #11

Both struck through with a pointer here.

## Open questions / things to flag for review

1. **Bottom-rate as a search prune.** Currently we only drop
   non-closure all-Bottom candidates (`drop_all_bottom`). A possible
   optimisation: track "fraction of children Bottom" and prune
   candidates above a threshold. Risky for higher-order contexts;
   defer to M4-with-data.
2. **Probe-based obs-eq soundness.** Documented in
   `m2-search-tasks.md` §4. The Bottom-skip we added during the
   un-typing rewrite makes the prune narrower (more candidates kept,
   slower search) but unblocks correctness. Worth revisiting if a
   future task family demands faster size-13+ enumeration.
3. **Library-version compatibility.** Before-vs-after-types
   serialisation is incompatible: old programs have `ty` fields, new
   ones don't. Anything we'd persisted is now a read-with-care
   exercise. (Unlikely to matter — we hadn't shipped anything.)
4. **Whether to bring back a *very* small typed sub-system later.**
   ARC grids may benefit from a single `Grid` runtime variant rather
   than encoding via `List<List<Int>>`. That's a Value-level
   addition, not a static-type addition.
