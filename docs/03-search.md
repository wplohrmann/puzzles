# Search

## Goal and framing

Given a `Task` (an oracle scoring programs) and a starting state (an empty
pool of nodes plus the task's input parameter nodes), find a node in the
pool whose values on the training examples match the task's targets, with
as small a program as possible, within a fixed compute budget.

We construct programs **bottom-up**: each search step grows the pool by one
typed node, formed by combining nodes already in the pool or by introducing
a literal / primitive reference. Every node, the moment it joins the pool,
has a concrete value per training example, which the neural policy uses to
score candidates. This is the dominant framing in modern PBE literature
(BUSTLE, DeepCoder, etc.) and the framing the network architecture in
[02-neural.md](./02-neural.md) is designed around.

We do not maintain a partial program with holes. The pool is a multiset of
fully-formed, evaluable nodes, and the program is "wherever in the pool a
solving node appears."

## State and actions

```rust
pub struct SearchState {
    pub arena: Arena,                 // shared, hash-consed, append-only
    pub pool:  HashSet<NodeId>,       // every node currently available
    pub size:  u32,                   // |pool|
    pub log_prior: f32,               // sum of log π over actions taken so far
    pub task_id: TaskId,
    pub examples: Vec<(Value, Value)>,// (input, target)
}

pub enum Action {
    /// Introduce a literal of given type/value. (Includes "literal copied
    /// from the task's examples", which is a separate proposal stream.)
    Literal(LitValue),

    /// Introduce a reference to a primitive (or library entry).
    PrimRef(PrimId),

    /// Apply one pool node to another. Both must already be in the pool;
    /// `func` must have a function type whose argument unifies with `arg`'s
    /// type.
    Apply { func: NodeId, arg: NodeId },
}
```

The starting state contains the task's parameter nodes (one `Param` of the
goal-type's argument) and nothing else. The first few actions seed the pool
with primitive refs and literals; later actions assemble them via `Apply`.

## Action enumeration

```rust
pub trait ActionEnumerator {
    fn admissible(&self, state: &SearchState, lib: &Library) -> Vec<Action>;
}
```

For a state with `m` pool nodes:

- **Literals**: a small fixed seed set (`0, 1, true, false, [], …`)
  plus a *value-copy* proposal stream — every distinct value that
  appears in the task's example inputs/outputs.
- **Primitive refs**: every primitive in the library.
- **Apply**: every `(f, a)` pair from the pool. The language has no
  static type system (see `01-language.md`), so construction always
  succeeds; mismatched pairs surface as `Value::Bottom` at evaluation
  time and are dropped via `drop_all_bottom` or collapsed via
  observational-equivalence dedup.

Hash-cons-canonical pool: any candidate whose `NodeId` already exists
is filtered out at zero cost.

## Scoring candidates

For each admissible action, we materialise the candidate node into the
arena (cheap — hash-cons may even return an existing id), compute its
values on the task's examples (one primitive call per example), then ask
the network to score it. See [02-neural.md](./02-neural.md) for the
network details.

The network scores are batched across all candidates in one forward pass.
Action priors `log π(a | state)` flow back into the search priority.

## Algorithm: best-first beam (default)

```
queue ← { initial_state }                           # priority = 0
while budget remains and queue not empty:
    s ← pop max priority
    actions ← enumerate(s)
    eval each candidate's values; check for solution
        if any candidate solves the task: return it (smallest one if many)
    log_πs ← Policy.score(s, candidates)            # one batched NN call
    for (a, lp) in zip(actions, log_πs):
        s' ← apply(s, a)
        if s'.size > size_limit: continue
        priority' ← s.log_prior + lp - α · s'.size  # MDL term
        queue.push(s', priority')
return None
```

Notes:

- "Solution check" is a free side-effect of evaluating each candidate.
  The moment a candidate's runtime values match the task's expected
  outputs on every example, we're done. (`Value::PartialEq` is
  type-strict, so value-equality alone is sufficient — there's no
  need for a separate type check.) This is the bottom-up design's
  biggest single ergonomic win over top-down.
- We do not call the value head every step; reserved for MCTS or for "give
  up" pruning.
- The size penalty `α · |pool|` enforces the MDL prior at search time.

## Algorithm: PUCT MCTS (option B)

Replace the single best-first queue with the standard AlphaZero PUCT tree.
Rollouts walk down to a leaf (continuing actions until a budget cap), the
value head returns an estimate (or rollouts use the policy as a default
policy and we evaluate at the end), and we back up.

In bottom-up search this is straightforward: nodes in the search tree are
`SearchState`s, edges are `Action`s, and the tree depth equals program
size. We expect this to be most useful for harder tasks where the value
head can prune large branches.

## Sharing across the pool

Because the arena is hash-consed, two semantically identical candidate
nodes collapse to the same `NodeId`. So:

- **Pool is a set, not a list**: adding a structurally-equal duplicate is a
  no-op; that branch of the tree is pruned automatically.
- **Sub-results are reused**: a node like `add 1 2` constructed early is
  available as an argument forever after, with no recomputation.
- **Library entries plug in identically**: a primitive ref is just a node
  in the pool; the search uses it via `Apply` like any other node.

## Policy / value training data

Every search produces a *trajectory*: the ordered sequence of actions taken
on each (winning and losing) branch.

```rust
pub struct Trajectory {
    pub task_id:  TaskId,
    pub steps:    Vec<TrajectoryStep>,
    pub solved:   bool,
}

pub struct TrajectoryStep {
    pub state_hash:        u64,                 // for debugging/dedup
    pub action_taken:      Action,
    pub action_log_prior:  f32,
    pub on_winning_path:   bool,                // value target
    pub candidate_count:   u16,                 // for diagnostics
}
```

`training` ingests these to train the network — see
[06-training.md](./06-training.md).

## Pruning beyond Apply

The M2 search already ships:

- **Observational equivalence** over runtime values. Two candidates
  whose stored value tuples match (across all task examples) collapse
  to one pool entry — exactly BUSTLE's "obs-eq" prune. This is the
  largest practical speedup available without a neural prior.
- **Probe-based obs-eq** for closure-typed candidates: apply each
  closure to its corresponding example input and dedup on the result.
  Soundness caveat documented in `decisions/m2-search-tasks.md` §4
  and `decisions/m2-strip-static-types.md`.
- **`drop_all_bottom`**: any candidate whose runtime values are all
  `Bottom` is skipped (it can't be a solution and can't usefully
  compose).
- **Closure-as-`f` prefilter**: any pool entry whose values contain
  no closure is skipped as an `f`-side App argument (applying a
  non-function produces `Bottom`).

A future addition once the system has actually run on more tasks:

- **Reachability**: drop any candidate whose runtime *value shape*
  can't reach the expected output shape within the remaining budget.
  This is the value-level analogue of the type-reachability prune we
  considered in the typed regime; needs empirical data on which
  shape transformations the primitive set can perform.

## Parallelism

- Within a search: action enumeration and candidate-evaluation are
  embarrassingly parallel; policy scoring is one big batched NN call. CPU
  parallelism for the former, GPU for the latter.
- Across searches: different tasks within a wake phase are independent.
  Run tasks on a thread pool; share an NN inference micro-batcher across
  tasks so the GPU stays busy.

## Outputs

```rust
pub struct SearchResult {
    pub program:    Option<NodeId>,
    pub solved:     bool,
    pub size:       u32,
    pub time:       Duration,
    pub trajectory: Trajectory,
}
```

The trajectory is what `training` consumes. The program (if any) is the
canonical id in the search arena and is portable into the library's arena
via a fold-copy.
