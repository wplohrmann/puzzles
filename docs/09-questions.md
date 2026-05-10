# Critical review and resolved decisions

This document records the points where I made design decisions you may want
to revisit, the places your original proposal had latent ambiguities, and
the answers we converged on through discussion.

After this document you should be able to start on `08-roadmap.md` Milestone 0
without further blocking decisions.

## Things I changed in your proposal (and why)

These are the spots I deviated most consciously. If any of them are wrong,
the architecture will need rework, so flagging them deliberately.

### 1. Bottom-up construction with value-aware embeddings (resolved)

Initial proposal: top-down expansion of typed holes. After discussion: the
search builds a *pool* of fully-formed, evaluable nodes from the leaves up,
adding one node per step (literal, primitive ref, or `Apply` of two pool
nodes), with the network seeing each node's actual values on the task's
examples. This is the BUSTLE / DeepCoder framing in modern PBE.

Trade-offs we accept:
- Larger per-step action space than top-down (any pair of pool nodes can
  be applied, modulo typing). Mitigated by aggressive type pruning and
  *observational equivalence* dedup (drop new nodes whose values match an
  existing pool node of the same type).
- Embedding caches split: structural part is cacheable across tasks, value
  part is cacheable only within a task. Per-edit cost is still O(1) in
  graph size, O(n) in example count.

### 2. Combinator-style first, lambdas optional

You described the language as "lambda calculus in a DAG." DAG sharing of
arbitrary lambda terms is subtle (free variables move scope when shared).
The pragmatic fix is to default to **combinator style**: no free variables,
higher-order arguments are passed by reference to primitives or library
entries. This is what DreamCoder does and what the IR I designed
prefers — but explicit `Lambda` nodes are still in the IR, with de Bruijn
levels, in case we need them.

Confirm this is acceptable, or insist on first-class anonymous lambdas and
I'll redesign.

### 3. ~~Types are mandatory~~ → no static types (revised at M2)

**Original position**: I built the architecture around Hindley-Milner
polymorphism, on the argument that without types the search would
explore ~100× more dead programs and library extraction would have
nothing to constrain anti-unification.

**Revised position (start of M2)**: stripped. Types were too restrictive
to support tasks that route through `Bool`/`Pair` intermediates, and
the auto-derived "goal-tycon filter" we'd need to make typed-search
tractable was both ad hoc and unsound. The neural recogniser sees
runtime values directly via node embeddings (`02-neural.md`); we let
it learn type-equivalents implicitly. Library extraction uses
runtime-value variants as a coarse type proxy (`04-library.md`).

The cost is a slower un-guided search at deeper sizes — `add-one-to-each`
(13 nodes) doesn't fit in 60 s without a prior. M4 (neural guidance)
is where this gets paid back.

See `docs/decisions/m2-strip-static-types.md` for the full discussion.

### 4. Two-component embeddings (revised after discussion)

Each node has a *structural* embedding (depends on kind, type, children's
structural embeddings — task-independent, cached across tasks) plus a
*value* embedding (depends on the node's concrete outputs on the task's
examples — task-specific, cached within a task). They're combined into the
final node embedding. The policy and value heads see both.

This preserves the cache benefit (per-edit cost stays O(1) in graph size)
while giving the network the "are we close to the target?" signal it
needs. See [02-neural.md](./02-neural.md).

### 5. AlphaZero-style MCTS as a *secondary* option

You raised AlphaZero. I think best-first beam search guided by the policy
prior is a better default for program synthesis (smaller per-step
expense, no rollouts, simpler). MCTS is implemented but not default.
Happy to flip the default if you'd rather lean into MCTS.

### 6. Total/strict semantics; no IO; finite fuel

I closed off non-termination by mandating strict evaluation with a fuel
counter and total primitives (`fold` etc. for recursion). This forecloses
on some classical functional programs (general recursion via Y) but makes
the search well-behaved.

### 7. `Bottom` instead of `Maybe` everywhere

When primitives fail (`head []`), the program returns `bottom` and the
task scores it as not-solved. This avoids forcing every list-touching
program to thread `Maybe` types around, which would balloon node counts.
Costs us a tiny bit of expressiveness. Confirm.

## Risks I think you should know about

- **Cache invalidation across library updates.** Every `LibraryRef`
  embedding becomes stale when a primitive's body changes. We track
  dependencies, but get this wrong and training silently degrades.
- **MDL counting subtlety.** Counting library bodies into the score lets
  the system trade program-savings against library-cost — but only if
  *novel* primitives count once globally, not once per use. We compute
  total-corpus size including library, not per-program.
- **Dreams over a young library.** Sampled programs are biased by the
  current library, which biases the network, which biases the next wake's
  search, which biases the next replay buffer. There's a feedback loop
  here that DreamCoder also has — it sometimes makes the library go down
  uninteresting paths. Audit-logging and library-GC are the levers.
- **ARC-AGI ambition vs. v0 reality.** ARC tasks are unusually hard for
  this approach. Expecting any meaningful pass rate in v0 will burn
  motivation. Treating it as the v1 stretch goal — with list/string as
  the v0 success criterion — is the right framing.

## Resolved decisions

These are the answers we've converged on.

1. **Neural framework: `tch` on Apple Silicon (MPS).** Trait-bounded
   `Tensor` so we can swap to `burn`/`candle` later without churn.
2. **Search: best-first beam first; MCTS later** as a benchmark
   comparison. Both share the same bottom-up state and action space.
3. **Lambdas: combinator-style by default; explicit `Lambda` nodes
   available in the IR** for cases (e.g. anonymous grid mappers in ARC)
   where they're cleaner. Search doesn't propose bare lambdas in v0.
4. **Floats and gradient-based literal optim: deferred** until after the
   list/string milestones.
5. **ARC is the real goal.** List/string are the warm-up that lets us
   prove the wake/sleep machinery is working before tackling ARC.
6. **Hardware: M5 Pro, 48 GB, single-machine.** No distributed training
   in v0; design throughput around this envelope.
7. **Online vs offline: tunable.** The training loop runs as long as
   you tell it to; checkpoints + a "freeze-and-export" path mean either
   mode is supported by a flag, not a redesign.
8. **Determinism: nondeterministic if it's significantly faster.**
   Keep a `--strict-determinism` flag for tests; not the default.
9. **Replay buffer: unbounded for now**, with a `Filter` hook so we
   can drop tasks (by age, family, program-size, etc.) later without
   restructuring.
10. **`graph-seek/` is throwaway.** New workspace; `crates/lang` from
    scratch.
11. **Type system: ~~HM-lite~~ → none (revised at M2)**. Nodes carry
    no static type. Mismatches surface as `Value::Bottom` at runtime.
    See item #3 above and `docs/decisions/m2-strip-static-types.md`.
12. **Runtime value variants in v0: `Int, Bool, Float, Char, Pair<A,B>,
    List<T>`, `Closure`, `Bottom`.** Trees, dicts, sets are *encoded*
    (`Tree<T> ≡ Pair<T, List<Tree<T>>>` etc.); sum types and ADTs
    deferred. Recursion is via `fold` and `unfold` primitives, not
    user-defined fixed points.

## Things explicitly out of scope for v0

So we're aligned on what we're *not* building first:

- Distributed/multi-machine training.
- Theorem proving / dependent types.
- An interactive UI.
- Probabilistic programs / stochastic primitives.
- Tasks that involve interacting with an environment.
- Multi-language program output (we generate one IR, period).
