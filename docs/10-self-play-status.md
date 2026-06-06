# Self-play training status — 2026-06-06

Handoff doc capturing where we are after the first real training runs.

## What's built

The self-play training stack from `docs/09-self-play-plan.md` is
fully wired together and mechanically working:

- **Four heads on a shared trunk** (`leaves` + `app_net` + `phi` +
  `attn`):
  - `q_head` — scores `(f, a)` candidates for the q-search (3N input,
    scalar out).
  - `poser_head` — same shape as q_head but trained against the
    poser's tent reward. Includes a `poser_stop_bias` (default +1.0)
    that adds a constant logit when `f_node` is the `stop`
    primitive.
  - `value_head` — per-node scalar baseline for the q-head's
    actor-critic loss (input: `h_struct(node)`, output: scalar).
  - `forward_head` — predicts `h_value(App(f,a), i)` from
    `h_struct(f), h_struct(a), h_value(f,i), h_value(a,i)`. Trained
    against the trunk's own (detached) `h_value`. Self-supervised on
    `app_net`'s dynamics.

- **`stop` primitive** (`crates/lang/src/builtin.rs`) is a
  search-time sentinel. Eval semantics return `Bottom` — should
  never be reached at eval; reaching it means the search has built a
  program with a nested stop, which is a bug. The search filter
  rejects `(X, stop)` (stop never in arg position) and rejects
  `(stop, *)` for the q-search (q ignores stop). Picking `(stop, X)`
  in Construct mode terminates the search with `n = X` as the
  program. Validity at search time: rejects `(stop, X)` if X's
  evaluated values would contain a `Closure` or `Bottom`.

- **`solve_guided_training`** (`crates/search/src/training.rs`)
  generalises `solve_guided` with: (i) top-K softmax sampling at
  each frontier expansion, (ii) trajectory recording (per-step
  candidates + chosen index + created `NodeId`), (iii) two scoring
  heads (Q / Poser), (iv) two termination modes (Solve / Construct).

- **A2C-MC actor-critic loss with per-action credit**
  (`crates/training/src/actor_critic.rs`). `C_t = R · 1[n_t ∈
  S_nodes]` — only actions that created a node in the solution DAG
  receive credit. Three quirks:
  - Actor loss is gated on success — failed searches don't update
    the policy (we can't tell which steps were wrong vs.
    right-but-budget-exhausted).
  - `Baseline::ValueHead` for q (full value-head loss),
    `Baseline::Constant(ema)` for poser (no value loss).
  - Stop-grad-at-trunk flag for the poser keeps the adversarial
    gradient out of the trunk.

- **SIGReg** (`crates/neural/src/sigreg.rs`) — Sketched-Isotropic
  Gaussian Regularization from LeJEPA (Balestriero/LeCun 2025).
  Random 1-D projections + Epps-Pulley characteristic-function test.
  Pure aux loss, default `λ = 0.05` (we use 0.001 — see below).

- **Gold-standard task generator** (`crates/training/src/gold.rs`).
  Three categories, sampled uniformly:
  - `arith_unary`: `((op v0) a)`, op ∈ {add, sub, mul}, a ∈ [-3, 3].
  - `arith_compose`: `((op2 ((op1 v0) a1)) a2)`.
  - `bool_truth_table`: one of 16 boolean functions of `(Bool, Bool)`,
    always 4 examples (the full truth table).

- **`--use-gold-only` mode** bypasses the poser entirely. `gold::
  sample_gold` provides the dream's I/O directly; q-head, value-head,
  forward-head, SIGReg still train. Forward head trains on the
  q-trajectory's App nodes (decoupled from any specific
  "ground-truth program") so it gets signal regardless of mode.

- **Diagnostic tool** `graph-seek-diag-q`
  (`crates/cli/src/diag_q.rs`). Runs the q-search on a balanced
  per-category gold-task batch. Reports solve rate per category,
  sample SOLVED programs, sample FAILED tasks, and trunk embedding
  variance. Supports `--load PATH` for trained checkpoints.

- **Other CLIs**:
  - `graph-seek-self-play` — main training loop.
  - `graph-seek-show-poser` — sample one fresh-init poser-search and
    print outputs (debug).
  - `graph-seek-show-dreams` — sample PCFG dreams (debug).

## Test status

- 14 unit tests across the workspace, all passing.
- Build clean.
- All three CLIs run end-to-end on a smoke test.

## What works

- **Forward head learns**. MSE drops monotonically (2.05 → 0.5 in 50
  iters, → 0.008 in 200 iters). Pure self-supervised signal on
  trunk's `app_net` dynamics.
- **Mechanical pipeline**. Sample dreams → run search → compute
  rewards → assemble losses → backward → step. No crashes, no NaNs.
- **Gold tasks 100% valid**. By construction, never produce
  closure/bottom outputs.
- **Trunk does NOT collapse**. Embedding per-dim variance stays
  ~0.075 (vs 0.076 fresh init) across 50 iters of training. SIGReg
  isn't even necessary at the current scale — variance is healthy on
  its own.

## What doesn't work

**The q-head policy does not learn anything useful.**

Diagnostic comparing fresh-init vs 50-iter trained model on the same
balanced gold-task batch:

| Category | Fresh init | Trained 50 |
|---|---|---|
| arith_unary | 8% (1/12) | **8% (1/12)** — same |
| arith_compose | 17% (2/12) | **17% (2/12)** — same |
| bool_truth_table | 25% (3/12) | **25% (3/12)** — same |
| Embedding variance | 0.076 | 0.074 |

Same solve count per category. Same exact programs found (all
`program: 0`, `program: false`, `program: v0` — *every* "solved"
case is a pool-shortcut on a literal/identity-equivalent task).
**Zero non-trivial solves** in either condition.

The "solve rate" we were tracking is essentially noise from how
often the random gold-task sampler happens to draw a constant or
identity-equivalent task. Training is having no effect on the
search's actual capability.

## Diagnosis: cold-start dead-end

Confirmed by raising the search budget 10× (200 → 2000):

| Budget | arith_unary | arith_compose | bool_truth_table |
|---|---|---|---|
| 200 | 1/12 (trivial) | 2/12 (trivial) | 3/12 (trivial) |
| 2000 | 4/12 (all trivial) | 3/12 (all trivial) | 0/12 |

Even with 10× budget, **fresh-init random q-search cannot find a
single non-trivial 2-App program**. Reason: the frontier grows
quadratically with pool size. Pool-pairwise candidates dilute
per-step probability of hitting any specific 2-App target faster
than the budget increase compensates.

Conclusion: random-init q-search cannot generate the positive
trajectories the actor needs to learn from. Without positive
trajectories, the actor has no useful gradient signal. Forward head
+ SIGReg train the trunk regardless, but trunk learning alone can't
make the q-head good at policy without supervised structure.

## Where we landed on what to try next

Two concrete options, in order of cheapness:

1. **Restrict the search-time library by input kind.** For arith
   tasks, restrict to `{add, sub, mul, param0, literals}` (drop
   ~20 primitives like fold, unfold, k, b, if). Frontier per step
   shrinks ~4×; random search becomes tractable. This is a quick
   experiment to verify the "frontier explosion" diagnosis. If
   q-head solves 2-App arith tasks in 200 steps with a restricted
   library, we know the search algorithm itself is fine.

2. **BC warmup on canonical solving programs.** For each gold task,
   construct a canonical solving program (small `App` tree), then
   train the q-head supervised to imitate the construction order
   for that program. After enough demos, the q-head has a useful
   prior; adversarial / RL search can take over.

(1) is the easy diagnostic experiment. (2) is the actual fix.

Worth also revisiting: **disable pool-shortcut in training mode** so
trivial wins still produce a 1-step trajectory the actor can learn
from. Doesn't change cold-start situation but removes the
zero-information-trajectory problem.

## Repo state

Branch: `main`, 11 commits ahead of `origin/main`. Recent:

```
cba3ddb Gold-standard task generator + use_gold_only training mode
5f08a90 First training run: fix normalization + add cold-start scaffolding
dc445d3 Delete obsolete complexity-curriculum stack
63a9f42 graph-seek-self-play CLI binary
8361fc5 Top-level self-play training loop
be19301 Training-mode search + A2C-MC actor-critic loss
af8bd26 Wire forward/value/poser heads into Network
6e65876 Add stop primitive (identity) to seed library
e2ed402 Self-play foundation: SIGReg + forward/value/poser heads
ecd2c86 Step 1 stopgap: recursive closure filter; plan self-play training
```

Uncommitted changes (to be committed with this doc): `GoldCategory`
enum + `sample_gold_in_category` + the `graph-seek-diag-q` binary.

## How to resume tomorrow

Start session by reading this doc plus `docs/09-self-play-plan.md`.
Then:

1. Try restricting the search library. The simplest place: add a
   `library_filter: Option<Fn(BuiltinId) -> bool>` to `SearchConfig`
   that `solve_guided_training` consults when seeding the pool.
2. Re-run `graph-seek-diag-q` at fresh-init to see if 2-App arith
   tasks now solve in 200-step budget.
3. If yes: design BC warmup. If no: deeper rethink of search
   algorithm.

Useful diagnostic invocations:

```sh
# Fresh-init baseline (what random q can do)
./target/release/graph-seek-diag-q --per-category 12 --max-budget 200 --seed 7

# Larger budget — verifies search-frontier-explosion hypothesis
./target/release/graph-seek-diag-q --per-category 12 --max-budget 2000 --seed 7

# Trained model — compare against baseline
./target/release/graph-seek-self-play --use-gold-only --iterations 50 \
    --lambda-sigreg 0.001 --dreams-per-iter 8 --max-budget 200 \
    --time-budget 5 --save-model /tmp/gold_model.safetensors

./target/release/graph-seek-diag-q --per-category 12 --max-budget 200 --seed 7 \
    --load /tmp/gold_model.safetensors
```

The saved model from today's run is at `/tmp/gold_model.safetensors`
(may be gone after a reboot; rerun training to regenerate).
