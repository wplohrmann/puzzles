# Neural network

## What the network does

For each candidate App pair `(f, a)` from the pool, the network predicts a
single scalar

```
q(f, a | task) = "is App(f, a) the next pool addition for this task?"
```

The search ranks candidate pairs by `q(f, a)` and pops the highest-scoring
first. See [`03-search.md`](./03-search.md) for the search algorithm.

To compute `q`, we feed the per-example value embeddings of `f` and
`a`, the target's embedding, and the structural embeddings of `f` and
`a` into a small network:

```
q(f, a | task) = MLP(h_value(f, ·), h_value(a, ·), h_target(·),
                     h_struct(f), h_struct(a))
```

The candidate's *result* `App(f, a)` is not evaluated until the pair is
popped from the frontier. The network ranks pairs by predicting "would
applying `f` to `a` advance this task?" from the inputs' runtime state,
not from the computed output.

The whole architecture is one head, one loss, one signal. The same
`app_net` and leaf tables embed structure, intermediate runtime values,
and targets into a shared space.

## Embedding the DAG

Every node has a **structural embedding** `h_struct(N) ∈ R^N`, computed
bottom-up via a shared composition network `app_net`. Every leaf has its
own learned embedding.

### Leaf embeddings

Tables of trainable vectors:

- `prim_emb[p]` per primitive ID (one row per built-in / library entry)
- `param_emb` for `Param(0)`
- `lit_int_emb`, `lit_bool_emb`, `lit_float_emb`, `lit_char_emb` — one
  shared base vector per literal type
- `nil_emb`, `pair_emb` — shared with the corresponding primitive rows
  in `prim_emb`
- `bottom_emb` for runtime failure values

One dimension of every leaf vector (`literal_dim`, the last slot) is
reserved for the literal's numeric content:

| value | base vector | `literal_dim` |
|---|---|---|
| `Int(v)` | `lit_int_emb` | `signed_log1p(v)` |
| `Bool(b)` | `lit_bool_emb` | 1.0 if `b`, else 0.0 |
| `Float(f)` | `lit_float_emb` | `signed_log1p(f)` |
| `Char(c)` | `lit_char_emb` | `ascii(c) / 128` |
| `PrimRef(p)` | `prim_emb[p]` | 0 |
| `Param(0)` | `param_emb` | 0 |
| Bottom | `bottom_emb` | 0 |

The `literal_dim` is allocated and learned alongside the rest of the
embedding; the network can choose to ignore it for non-literal leaves.

### Composition (`app_net`)

A small MLP `app_net: R^{2N} → R^N` (e.g. `2N → 128 → N`, ReLU between
layers, layer-norm on the output).

```
h_struct(App(f, a)) = app_net([h_struct(f), h_struct(a)])
```

`h_struct(N)` is **task-independent** — it depends only on the DAG and
the model weights. Cached globally, keyed by `(model_version, NodeId)`.

### Value embeddings (per example)

For each task example `i` with input `x_i`, every node `N` has a value
embedding `h_value(N, i) ∈ R^N`. The rule:

1. We evaluate `N` on every `x_i` for the search's solution check, so
   `eval(N, x_i)` is available for free.
2. **If `eval(N, x_i)` is a concrete value** (`Int`, `Bool`, `Float`,
   `Char`, `List`, `Pair`): `h_value(N, i) = embed_value(eval(N, x_i))`.
3. **If it is a Closure** (partial application that cannot reduce
   further): walk `N`'s DAG via `app_net`, substituting `Param(0)` by
   `embed_value(x_i)` and any already-applied closure arg by its
   `embed_value(...)`. The same compositional rule, parameterised on the
   input.
4. **If it is Bottom**: `h_value(N, i) = bottom_emb`.

Case 2 is the strongest signal — when we know the runtime result, the
embedding reflects it directly. Case 3 captures the structure of partial
applications that cannot reduce.

`embed_value(v)` for a runtime `Value`:

- `Int(n)`, `Bool(b)`, `Float(f)`, `Char(c)` — the leaf embedding with
  `literal_dim` filled in.
- `List([h, t...])` — `app_net([app_net([prim_emb[cons], embed_value(h)]),
  embed_value(rest)])`. Empty list = `prim_emb[nil]`.
- `Pair(a, b)` — `app_net([app_net([prim_emb[pair], embed_value(a)]),
  embed_value(b)])`.

The **same `app_net` and leaf tables** are used for `h_struct`,
`h_value`, and `h_target`. This places targets, intermediate runtime
values, and program structure all in the same embedding space, so the
question "does this candidate get us closer to the target?" is a
learnable function over vectors of the same shape.

Value embeddings are cached per-task, keyed by
`(model_version, task_id, NodeId, example_idx)`. Dropped when the task
search ends.

### Target embedding

Each example's expected output `y_i` is a concrete `Value`. Its
embedding is `h_target(i) = embed_value(y_i)`. Same compositional rule;
shares all the embedding parameters.

## Computing `q(f, a)`

For a candidate pair `App(f, a)`:

1. **Per-example projection.**

```
per_ex_i = phi([h_value(f, i), h_value(a, i), h_target(i)])  # MLP, R^{3N} → R^N
```

2. **Aggregate across examples via cross-attention**, with the joint
   structural embedding of `(f, a)` as the query:

```
struct_pair = [h_struct(f), h_struct(a)]
query       = W_q · struct_pair                     # R^N
key_i       = W_k · per_ex_i
val_i       = W_v · per_ex_i
alpha_i     = softmax_i( query · key_i / sqrt(N) )
ctx         = Σ_i alpha_i · val_i                   # R^N
```

The candidate-conditioned query lets each pair attend to the examples
that are most diagnostic for it.

3. **Score head.**

```
q(f, a) = q_head([ctx, h_struct(f), h_struct(a)])   # MLP → scalar
```

`q(f, a)` is cached per `(model_version, task_id, f.NodeId, a.NodeId)`.
Once a candidate is scored, its score is stable for the rest of the
task's search.

### Filters

- **Hash-cons identity (at enqueue).** If `intern(App(f, a))` already
  corresponds to a pool entry, drop without scoring.
- **Bottom propagation (at pop).** When a popped pair is evaluated,
  `apply(f.values[i], a.values[i]) = Bottom` for any example `i`
  means the candidate cannot be on a solution path. Drop without pool
  admission. The evaluation itself is the same one that would happen
  on admission, so this costs nothing extra.

## Training

For each dream `D` with bottom-up trajectory `[S_1, …, S_T]` (the
canonical topo sort of `D`'s DAG), at each step `t`:

- `pool_t` = seeds ∪ `{S_1, …, S_{t-1}}`
- **Positive pair**: `(f_t, a_t)` such that `S_t = App(f_t, a_t)`.
- **Candidate set `C_t`**: the positive plus a sample of negative pairs
  drawn from `pool_t × pool_t`.

### Loss

Softmax cross-entropy over `C_t`'s `q` logits with the positive pair as
the target:

```
loss_t = -log( exp(q(f_t, a_t)) / Σ_{(f, a) ∈ C_t} exp(q(f, a)) )
```

Per dream the loss is `Σ_t loss_t`; per batch it is the mean over
dreams.

### Negative sampling

The full candidate set is `O(|pool_t|²)`, which is too large to score
exhaustively at every gradient step late in a long dream. Each step
samples `K` negatives from `pool_t × pool_t \ {(f_t, a_t)}`. Two
policies, optionally mixed:

- **Uniform** — cheap, smooth gradient.
- **Hard** — top-K by current `q`, ignoring the positive. Sharpens the
  model where it is most wrong.

`K = 64` with a 70/30 uniform/hard mix is a reasonable default.

### Backpropagation

Gradients flow through every leaf embedding, `app_net`, the per-example
projection `phi`, the attention parameters, and the `q_head` MLP.
Optimizer: Adam.

After a training round, bump `model_version` and drop the global
structural cache; per-task caches expire when the task search ends.

## Caches

| Tensor | Cache key | Invalidated by |
|---|---|---|
| `h_struct(N)` | `(model_version, NodeId)` | model weights bump |
| `h_value(N, i)` | `(model_version, task_id, NodeId, example_idx)` | task ends; weights bump |
| `h_target(i)` | `(model_version, task_id, example_idx)` | task ends; weights bump |
| `q(f, a)` | `(model_version, task_id, f.NodeId, a.NodeId)` | task ends; weights bump |

A bump to `model_version` (post-training-step) wipes everything global;
per-task caches drop when the task search finishes.

## Compute summary

Per pool-add at search time:

- 1 `app_net` forward for `h_struct(new)` (children's embeddings cached).
- `n` `app_net` walks for `h_value(new, i)` — one per example, children
  cached.
- For each existing pool entry `e`, two pair candidates `(new, e)` and
  `(e, new)`:
  - 2 × cross-attention pool + `q_head` MLP forward.

Per pop (only for the candidate that actually advances the search):

- `n` `apply` calls — the same evaluation the unguided baseline does on
  admission.

Total: **O(pool)** `q`-forwards per pool-add, **O(1) in graph size**
modulo cached children. The `q`-scoring work is parallelisable across
pairs and across examples.

## Choice of framework

Pure-Rust hand-rolled MLP + Adam. No external tensor-library dependency.
A thin trait wall keeps the search-facing API (`q(f, a) → f32`)
framework-agnostic so a retarget to a GPU library later is a localised
change.

Module layout (`crates/neural/`):

```
embed.rs    leaf tables, app_net, embed_value, structural / value walks
attn.rs     cross-attention pooler over examples
heads.rs    phi, q_head
mlp.rs      MLP primitive + Adam
network.rs  top-level Network: forward, train_step, cache management
```
