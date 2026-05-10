# Library extraction (abstraction sleep)

## Goal

Given a corpus of programs (the replay buffer of recently-solved tasks),
identify recurring patterns and promote them to new library primitives, so
that next-iteration programs are smaller and the search-prior is sharper.

This is the most algorithmically delicate piece of the system, and the place
where being wrong about the objective will silently hurt for thousands of
training iterations. The objective is **Minimum Description Length (MDL)**:

  argmax_L  [ Σ_t (max_ρ log P(ρ | L) given ρ solves t) − α · |L| ]

i.e. choose a library `L` such that, with each task's best program rewritten
under `L`, the *total* node count (programs + library bodies) is minimised.

## Algorithm

We don't reach for full version-space algebras in v0; the simpler
"pattern-mine, anti-unify, score, beam" pipeline below already captures most
of the lift and is easy to debug.

### Step 1 — Mine candidate fragments

A *fragment* is a connected subgraph of one of the corpus programs, with
zero or more of its sub-trees marked as **abstraction holes** (positions
that will become parameters of the new primitive).

Candidates are generated in two passes:

1. **Anchor enumeration.** Walk every program and for every node `n`, take
   the subgraph rooted at `n` of size up to `K`. For each subgraph, mark
   each maximal sub-tree as either *included* or *abstracted*, stopping
   abstraction at depth `D`. This yields a finite multiset of fragments.
2. **Anti-unification across occurrences.** Group fragments that share an
   anchor *shape* (sequence of node kinds modulo abstracted leaves) and
   compute their anti-unifier: the most-specific common generalisation.
   The anti-unifier becomes a single candidate primitive.

Concrete bounds for v0: `K = 8` (max fragment size), `D = 2` (max abstraction
depth). These are aggressive limits but still produce thousands of
candidates per iteration, which is what beam search wants.

#### Anti-unification without static types

The original design used type-driven anti-unification: holes were
constrained by the polytypes of the sub-trees they replaced, so two
positions with incompatible types couldn't unify into the same hole.
Without a static type system (M2 onwards), we drop that.

Two compensating signals are available at no extra cost:

- **Runtime-value shape.** Every program in the replay buffer was
  evaluated against its task's example inputs during search. The
  `Value` variant observed at each hole position across occurrences
  (`Int` here, `List` there, mixed both) is a coarse runtime-derived
  type proxy. Reject candidates whose hole positions span
  contradictory variants beyond what a primitive could plausibly
  accept.
- **Bottom-rate.** A candidate primitive that, when substituted back
  into the corpus, produces `Bottom` more often than it doesn't is
  almost certainly an over-generalised hole. Score it down.

Both checks are cheap and replace the structural soundness that types
used to provide.

### Step 2 — Score each candidate

For candidate fragment `f` (representing a hypothetical new primitive with
body `body_f` and arity `arity_f`):

```
saving(f) = Σ_t [ size(ρ_t) − size(rewrite(ρ_t, f)) ]
            − size(body_f) − const_per_lib_entry
```

`rewrite(ρ_t, f)` finds every occurrence of `f`'s pattern in `ρ_t` and
replaces each with a single `App` chain referencing the new primitive plus
the arguments at the abstraction holes. `size(...)` is just the node count
in the canonical (hash-consed) representation.

Negative savings means adding `f` would make the corpus *bigger*; we drop
those.

### Step 3 — Beam search over libraries

Compression interactions matter: adding `f1` may make `f2` redundant or
make `f3` more profitable. So we don't just pick the top-K candidates —
we beam-search over libraries:

```
beam ← { current_library }
for step in 1..max_new_entries:
    successors ← ∅
    for L in beam:
        for f in mine_candidates(corpus_rewritten_under(L)):
            L' ← L ∪ {f}
            successors ← successors ∪ { (L', total_score(L')) }
    beam ← top-W successors by total_score
return best L over the trajectory
```

`W = 8`, `max_new_entries = 5` per abstraction sleep is plenty; this is more
than enough budget for the algorithm to find profitable abstractions and
escape local minima where two interacting fragments need to be added
simultaneously.

### Step 4 — Refit the corpus

Once we've picked a new `L*`, every program in the replay buffer is
rewritten under `L*` in canonical form. This rewritten corpus is what dream
training and the next wake phase consume.

## Concrete data flow

```rust
pub struct AbstractionInput<'a> {
    pub library:    &'a Library,
    pub corpus:     &'a [(TaskId, NodeId, Arena)],   // arenas are shared-able
    pub max_new_entries: usize,
}

pub struct AbstractionOutput {
    pub new_library:    Library,
    pub rewritten_corpus: Vec<(TaskId, NodeId)>,
    pub new_entry_log:  Vec<NewEntryAudit>,         // for telemetry
}

pub struct NewEntryAudit {
    pub name:     String,
    pub body:     NodeId,
    pub arity:    u8,
    pub savings:  i32,
    pub occurrences_replaced: usize,
}
```

`new_entry_log` is part of the contract: every abstraction sleep emits a
human-readable audit so we can sanity-check what the system is doing. ("It
just discovered `flip f a b = f b a`, occurrences=24, savings=47" is a much
better debug experience than a black-box library that grew by one entry.)

## Picking names

New primitives get **automatic names** like `f37` initially. A small post-
processing step can guess better names by inspecting the body — e.g.
"swap-args" for the obvious S-combinator-like body — but that's polish.

There are no types to infer; the body's runtime behaviour and the
arity (= number of holes) are all that's stored.

## Avoiding pathological libraries

A few guardrails worth designing in early:

- **Don't add nullary primitives** (no abstraction holes) unless they're
  used many times: a constant program fragment's "abstraction" is just a
  shorter name and barely worth a library slot.
- **Cap arity** at, say, 4. Beyond that the abstractions are usually too
  specific to be reused.
- **Reject equivalent entries**: if a candidate's body produces the
  same runtime values as an existing primitive (or a curried partial
  application of one) on a fixed probe-input set, drop it. (With
  static types removed, η/β-equivalence has to be tested
  observationally rather than structurally.)
- **Periodic library garbage collection**: once per N abstraction sleeps,
  drop primitives whose usage in the replay buffer fell below a threshold.
  Otherwise the library accumulates dead weight from earlier domains.

## Why not version-space algebras (yet)

DreamCoder uses VSAs to represent *all* refactorings of a program
efficiently, then mines patterns across these compact representations. It's
elegant and important at large scale. For v0 the bounded-fragment approach
above is much easier to implement, easier to debug, and good enough for
the small-domain benchmarks we'll start with. Once we have a baseline, we
revisit VSAs as a performance optimisation, not an architectural change.

## Connection to the prior

After abstraction sleep, the per-primitive prior `P(p | L)` is updated to
reflect the new corpus's primitive frequencies (smoothed):

```
P(p | L) ∝ (count(p) + 1) / (Σ count(q) + |L|)
```

This is the prior the search uses *before* the network has been retrained
on the new library. It's a cheap, useful initialisation; the network
overrides it after a few epochs of dreams.
