# Neural network

## What the network does

Two jobs:

1. **Embed each program node** into a vector that combines what the node *is*
   (structure, type, primitive choice) and what the node *produces* on this
   task's inputs (its concrete values).
2. **Score actions and states** for `search`: a policy `π(a | state, task)`
   over candidate next nodes, and a value `V(state, task) ∈ [0, 1]`
   estimating the probability that the current pool of nodes will lead to a
   solution within the remaining budget.

This is a **bottom-up, value-aware** design (after BUSTLE, DeepCoder, and
similar). Every node in the program has a concrete value per task example
the moment it is constructed, so the network never has to guess whether a
fragment is "close" — it sees the actual outputs and compares them to the
target. Structural information still matters (it tells the network *why*
the values are what they are, and supports generalisation across tasks),
but it's combined with the values, not used in isolation.

## Per-node embedding

For node `v` with children `c₁, ..., cₖ` and values `vals_v = [v(x₁), ..., v(xₙ)]`
on the task's `n` training examples:

```
h_v^struct = StructMLP( kind(v), type(v), [h_{cᵢ}^struct] )       # task-independent
h_v^value  = ValueEnc( vals_v, targets )                          # task-specific
h_v        = Combine( h_v^struct, h_v^value )
```

- `h_v^struct` depends only on the DAG; it is invariant across tasks. This
  is what we cache *globally*, keyed by the node's structural hash.
- `h_v^value` depends on the task's concrete examples. We cache it
  *per task*, keyed by `(task_id, node_hash)`.
- `Combine` is a small MLP / gating layer that mixes the two.

For nodes that don't depend on any `Param`, `vals_v` is constant — for
literals, primitive refs, and other parameterless subterms the value cache
is trivially invariant across input examples. Hash-cons sharing pays off
here too.

### Encoding values

`ValueEnc` is the tricky bit. A `Value` is one of: `Int`, `Bool`, `Float`,
`Char`, `List<T>`, `Pair<A, B>`, `Closure`, or `Bottom`. We encode each
*per example* and then aggregate across the `n` examples:

```
per_ex_v = [ValueTokenEnc(v(xᵢ), τᵢ_target) for i in 1..n]
h_v^value = AggregateMLP( per_ex_v )    # e.g. mean + max + concat-target-comparison
```

Per-type tokenisers:

- `Int` / `Float`: Fourier features over a log scale, plus a "matches target?"
  binary feature.
- `Bool` / `Char`: small embedding tables.
- `List<T>`: a tiny sequence encoder over the per-element tokens (deepset
  for unordered tasks, RNN/short-Transformer for ordered ones; choose per
  domain). We also include scalar features (length, contains-target?).
- `Pair`: concat of subencodings.
- `Closure`: fall back to the structural embedding of the closed-over body —
  closures aren't directly evaluable on examples without an argument.
- `Bottom`: a learned "failure" token. Important to encode rather than mask
  away, because programs that fail on *some* examples are still informative.
- `Grid<T>` (ARC domain): a small ConvNet.

`AggregateMLP` includes a comparison to the target outputs:

```
features per example = [encode(value), encode(target), encode(diff(value, target))]
```

The diff features are domain-specific (numeric difference, list-edit
distance, grid xor-mask, …) and are the network's strongest "are we close?"
signal. Implementing them well per domain is one of the highest-leverage
parts of this whole project.

## Caching, the central performance argument

Bottom-up plus hash-consing means every step of the search adds **exactly
one new node** to the pool. Everything else is unchanged.

When we add node `v` with already-cached children:

- `vals_v`: one primitive call per example (`n` invocations).
- `h_v^struct`: one `StructMLP` call.
- `h_v^value`: one `ValueEnc` call.
- All other nodes' embeddings are still valid — no recomputation.

So the per-edit NN cost is O(1) in graph size and O(n) in example count.
This is the property that makes a bottom-up GNN tractable for tree search
over programs.

Two caches:

```rust
struct StructCache  { table: DashMap<u64, Tensor> }       // node_hash → tensor
struct ValueCache   { table: DashMap<(TaskId, u64), Tensor> }
```

Invalidation:
- `StructCache` invalidates when network weights change (post-dream-sleep)
  or a primitive's body changes (library rewrite).
- `ValueCache` is per-task; it lives for the duration of a search and is
  dropped when the search ends.

## Heads

### Task encoder

Independent from the program embedder. Encodes the (input, output) examples
into a fixed-size context `c_t`. Permutation-invariant over examples (Set
Transformer / DeepSets). For grid tasks, a ConvNet per grid then concat. For
list/string tasks, recursively encode each list using the same per-token
encoder used inside `ValueEnc` (so the "what does this list look like"
signal is shared between task encoding and node-value encoding — this
sharing is desirable and why we design them together).

### Policy: scoring candidate new nodes

At step `t`, the search proposes a set of candidate new nodes — every
type-valid combination of existing nodes plus literal/primref candidates.
For each candidate `c`:

1. Compute `vals_c` (one primitive call per example — uses cached child
   values, very cheap).
2. Compute `h_c^struct` and `h_c^value` (one MLP call each).
3. Score:

```
π(c | state, task) ∝ exp(  PolicyMLP([h_c, h_state, c_t])  )
```

`h_state` is a state summary: aggregation (mean + max, or attention) over
all current pool nodes' embeddings, plus a few scalar features (pool size,
node count, has-any-pool-node-typed-as-goal?).

This means we **do** evaluate every candidate to get its values. That's the
whole point of the bottom-up design — the policy gets to see "this
candidate, on the task's examples, produces these values" before scoring.
The cost is one primitive evaluation per example per candidate, which is
much cheaper than the network forward pass that follows.

A useful early-exit: if `vals_c == targets` exactly, we have a solution
right there; no scoring required.

### Value head

```
V(state, task) = sigmoid( ValueMLP([h_state, c_t, frontier_features]) )
```

Trained from search trajectories: 1 if this state is on a path that solved
the task in this iteration, 0 otherwise.

### Bigram / parent context (optional refinement)

The DreamCoder bigram trick — score a candidate conditional on the parent
production type — is still useful here. We can add as features to
`PolicyMLP` the structural embedding of any node whose type matches `c`'s
type (so the policy "knows what this candidate would plug into"). Worth
trying once the basic version works.

## Summary of caching guarantees

| Tensor | Cache key | Invalidated by |
|---|---|---|
| `h_v^struct` | `(model_version, node_hash)` | weights change; primitive body change |
| `vals_v` | `(task_id, node_hash)` | task ends |
| `h_v^value` | `(task_id, node_hash)` | task ends |
| `c_t` (task encoding) | `(task_id, model_version)` | weights change; task ends |

In aggregate: per search step, work scales with `n` (examples) plus a small
constant per candidate, *not* with program size. This is what makes the
whole architecture practical.

## Training the network

Two data sources, mixed in mini-batches:

1. **Replays.** From solved searches: every `(state, action_taken,
   solved-from-here?)` tuple is a training example. The action that led to
   the winning program is the policy target on the winning path; value
   target = 1 on the winning path, 0 elsewhere.
2. **Dreams.** Sample a program ρ from the prior under the current library;
   sample inputs; compute outputs; treat as a synthetic task. Synthesise
   the dream's "training trajectory" by choosing a deterministic
   bottom-up construction order for ρ (topological order of its DAG nodes)
   — every prefix of the order is a state, the next node is the action.
   Crucially, dreams give the network value-encoding signal even when the
   wake phase produced too few replays.

Loss = policy cross-entropy + value MSE + small regulariser + entropy bonus.

Network update protocol after dream training:
- Bump `model_version`.
- `StructCache` is dropped wholesale; recomputed lazily.
- `ValueCache` is per-search anyway; nothing to invalidate.
- The next wake phase warms the new caches as it goes.

## Choice of framework

Recommend `tch` (libtorch bindings) for the prototype: best maturity,
broadest op coverage, fastest path to a working network. Hide it behind a
`Tensor` trait + a small `Module` trait so we can swap to `burn` later
without touching `search` or `lang`.
