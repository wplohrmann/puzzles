# Self-play training plan

## Motivation

The current complexity-curriculum training plateaus at L2. The plateau
has two root causes, one shallow and one deep:

- **Shallow**: `dream.rs:307` filters top-level closures from example
  outputs but not closures *nested inside* lists/pairs. Roughly half of
  the L2 dream distribution has closure-bearing outputs that compare
  not-equal to any other closure value, making those dreams provably
  unsolvable by exact-value-equality. The plateau on `solve_rate` is
  inevitable.
- **Deep**: the network is trained to imitate the *specific
  construction trajectory* that generated each dream. But the
  search-eval objective only cares about I/O behavior — many programs
  with the same I/O exist, and many paths lead to each. The training
  objective is much narrower than what `solve_rate` actually measures.
  This is the sparse-reward problem in disguise: per-step trajectory
  imitation can climb top-1 indefinitely while `solve_rate` stalls.

This plan replaces the supervised dream-imitation regime with a
self-play setup: a **poser** proposes tasks at the edge of the
**searcher**'s capability, a **forward-prediction** auxiliary head
densifies trunk gradient on every step regardless of search outcome,
and a **SIGReg** distributional regularizer prevents trunk collapse.

## End-state architecture

One shared trunk (`leaves`, `app_net`), three heads, plus SIGReg on the
trunk's `h_value` outputs.

```
shared trunk:   leaves  →  app_net  →  {h_struct, h_value}
                                            │       │
                ┌───────────────────────────┼───────┴────────────────┐
                │                           │                        │
         [forward head]                  [q-head]              [poser head]
          MSE: predict             A2C-MC actor on              REINFORCE on
          h_value(parent)          search trajectories,          shaped budget
          stop-grad on target      value-head baseline           tent reward
                │                           │                        │
                ▼                           ▼                        ▼
          full grad into              full grad into          stop-grad at trunk
          trunk                       trunk                   (head-only)
                                            │
                                            │
                          SIGReg auxiliary loss on h_value (λ ≈ 0.05)
```

Gradient-flow rationale:
- Forward + q are **cooperative** — both want a trunk that produces
  semantically-meaningful, predictive embeddings. Full gradient flow.
- Poser is **adversarial** — its objective is to find tasks where the
  searcher fails. If its gradient flowed into the trunk it could
  minimize its loss by corrupting the searcher's representation
  instead of by finding genuinely hard programs. Stop-grad at the
  trunk forces it to do the right thing.
- Forward-head target uses stop-grad to prevent trivial trunk collapse
  ("predict the constant 0"). SIGReg provides a second, independent
  anti-collapse pressure.

A **value head** (predicts per-action credit from a node embedding)
is added as a fourth cooperative head. Specifically: V takes
`h_struct(node)` for a single node and outputs a scalar. At each
search step where action `a_t` creates `App(f, a)`, we use
`V(App(f, a))` as the baseline for that step. The advantage becomes
`C_t − V(App(f, a)).detach()`. V is trained by regressing against
`C_t` per visited node. This is an *action-conditional* baseline —
strictly speaking it introduces a small policy-gradient bias, but in
practice it's a standard variance-reduction technique. No pool
aggregation needed; just the node embedding.

The poser uses a simpler EMA baseline (no value head) because its
trajectories are short and low-variance.

## Poser as a search

The poser uses the **same search machinery as the q-head** — it runs
a best-first guided search over candidate `App(f, a)` expansions,
scoring them with the poser-head instead of the q-head. The two
"searches" differ only in (a) which scoring head ranks the frontier
and (b) what termination condition triggers the trajectory's end.

To terminate poser-construction, the library gains a special
primitive **`stop`**, defined with identity eval semantics: `stop(x)
= x`. When the poser's search produces an expansion `App(stop, n)`,
construction halts and `n` is taken as the final program for the
dream. The poser-head is initialized with a logit bias toward
`(stop, n)` candidates at small `N_poser`, providing the
`<stop>`-bias cold-start.

The q-head's library is the same (no need to maintain two libraries);
the q-head will rarely pick `App(stop, n)` because doing so wastes a
node on a no-op. If empirically the q-head misuses `stop`, we can
strip it from the q-head's frontier candidates without affecting the
poser.

This unification means we maintain one search implementation, one
trajectory format, and one set of training-time recording mechanics.
The poser doesn't need a separate construction loop — it's just "run
the search, but with the poser-head scoring".

## Reward design

Two quantities per search trial:
- `S_pool` — frontier expansions consumed before solving (or hitting
  `max_budget`). The *difficulty* signal.
- `S_sol` — node count of the solving program returned. The *quality*
  signal.

### Searcher reward (terminal)

```
search_cost = S_pool + α · S_sol
r_searcher  = max(0, 1 − search_cost / max_budget)     if solved
            = 0                                         otherwise
```

`α` is the only new knob: how many pool expansions one solution-node
"costs". Default `α = 10`.

`r_searcher` is then *decomposed* into per-action credit `C_t` for the
q-head's loss — see "Per-action credit" below.

### Per-action credit (exploits problem structure)

The standard A2C-MC interpretation would assign every step on the
trajectory the same return `R = r_searcher`. That's wasteful — the
search makes many expansions that turn out to be dead ends, and
reinforcing the policy on those is counter-productive. The structure
of our problem makes this fixable.

After the search returns a solving program, the solution is a *DAG of
App nodes* — call that set `S_nodes`. Of all the actions the search
took, only some created nodes that ended up in `S_nodes`; the rest
created dead-end nodes that never appeared in the final solution.
(The search-loop's frontier-dedup guarantees every action creates a
fresh node, so there's no hash-consing ambiguity in this membership
check.) Define:

```
C_t = r_searcher    if the App node created by action a_t is in S_nodes
    = 0             otherwise
```

This is the per-action credit. Three important properties:

- **Order-invariant**: `C_t` depends only on whether the *node*
  produced by action `a_t` is in the solution, not on when in the
  trajectory it was created. Two search orderings that ultimately
  produce the same solution distribute credit identically.
- **Failed searches don't update the actor at all.** If no solution
  found, we *skip the actor loss for that trajectory entirely*. Why:
  a failed search is most often a search that ran out of budget; many
  of the actions taken on the way may have been correct, and we have
  no way to tell from the failure alone which were right and which
  were wrong. Penalizing them based on `−V(s_t)` would introduce
  systematic noise. The value head, forward head, and SIGReg all
  still train on failed trajectories — only the policy gradient is
  gated on success. As the trunk and value head improve, the policy's
  initial random behavior eventually intersects the poser's simplest
  programs, the first searches succeed, and the actor starts learning.
- **Dead-end actions inside a successful trajectory still get
  penalized.** If the search solved but action `a_t` produced a node
  not in `S_nodes`, advantage = `0 − V(s_t)` and the policy is pushed
  away from `a_t` proportional to how promising `V` thought `s_t`
  was. This is meaningful signal — the search succeeded, so we know
  the dead-end was avoidable.

The q-head's actor-critic loss uses `C_t` in place of the trajectory
return; the value head regresses against `C_t` per visited state on
every trajectory (solved or not). Mathematically this is a valid
baseline-subtracted advantage estimator on a (different,
problem-specific) reward signal — see the training-loop section for
the concrete loss.

### Poser reward (terminal)

Piecewise-linear tent on `S_pool` (only — not `S_sol`):

```
peak = β · N_poser
if invalid program (closure or ⊥ in any output) → 0
elif not solved within max_budget               → 0
elif S_pool ≤ peak:
    r_poser = small_floor + (1 − small_floor) · (S_pool / peak)
else:
    r_poser = max(0, (max_budget − S_pool) / (max_budget − peak))
```

`β` controls how forgiving the peak is of branching overhead. Default
`β = 8`: a 5-node poser program peaks at 40 pool expansions. Re-tune
empirically.

`small_floor = 0.05` — any valid program gets at least this, so the
poser never falls to zero on a successful trajectory and REINFORCE has
gradient even for solved-too-fast cases.

### Cross-effect (no explicit coupling)

The poser indirectly cares about `S_sol` via the searcher's behavior:
if poser-program has a hidden short equivalent, the searcher (which
optimizes `S_pool + α · S_sol`) will tend to find it quickly → low
`S_pool` → poser's tent reward drops. So the poser is implicitly
pushed toward programs that are near-optimal at their own length.
This is the desired auto-curriculum dynamic — no explicit
solution-quality term needed in the poser reward.

## SIGReg auxiliary

Per training step, after computing `h_value(node, example_i)` for all
nodes in the batch:

1. Stack into matrix `Z ∈ R^{B × N}` where B is the total
   (node, example) count and N is the embedding width.
2. Draw `M = 1024` random unit vectors `a_m ∈ S^{N−1}` (resampled
   every step — important).
3. Project: `P = Z A`, `P[:, m] = Z @ a_m`.
4. For each column, compute Epps–Pulley test value against `N(0, 1)`
   via trapezoidal quadrature with 17 points on `t ∈ [−5, 5]`,
   weight `σ = 1`.
5. `L_SIGReg = mean over M of EP(P[:, m])`.
6. Add to total loss with weight `λ ≈ 0.05`.

**Important**: do not LayerNorm or L2-normalize `h_value` before
SIGReg. SIGReg targets `N(0, I)` geometry, not a sphere. If the q-head
or forward-head want normalized features, branch — feed raw embedding
to SIGReg, a normalized copy to the heads.

Reference implementation at `github.com/rbalestr-lab/lejepa` — vendor
the ~50 lines directly into a new `crates/neural/src/sigreg.rs`.

## Training loop

Each iteration:

1. **Sample a batch of `D` dreams** (default `D = 16`). For each:
   - Run the poser-search: best-first guided search scored by the
     poser-head, sampling actions with top-K softmax (training mode).
     The search terminates when the frontier produces `App(stop, n)`.
     Cap at `max_poser_nodes` expansions; abandon if hit.
   - The resulting program is `n` (the argument to `stop`).
   - Generate `K` example inputs (default `K = 3`), run `n` → I/O
     examples. Check validity (no nested closures, no ⊥). Invalid →
     bail with `r_poser = 0`, no q-training data.
2. **Run guided search** with the q-head's softmax over frontier
   candidates as the action-sampling policy (during training; argmax
   at eval). Budget = `max_budget`. Record:
   - `S_pool` (frontier expansions),
   - `S_sol` (node count of returned program, if any),
   - the full search trajectory: for every step, the action chosen,
     its log-probability under π, and the set of alternative frontier
     candidates that were considered.
   - whether the search-found program solves the I/O examples
     (it may not be the poser's program — equivalent programs count).
3. **Compute rewards and per-action credit.** Compute `r_searcher`
   and `r_poser` per the formulas above. If the search solved, walk
   the solution DAG to enumerate `S_nodes` (the App nodes that are
   part of the solving program). For each action `a_t` taken during
   the search, look up the App node it produced, set:
   - `C_t = r_searcher` if that node is in `S_nodes`,
   - `C_t = 0` otherwise.
   For failed searches, `C_t = 0` for every t.
4. **Loss assembly**:
   - **Forward head** (auxiliary, dense): from the poser's program
     DAG (always available, regardless of search outcome), pick all
     internal `App(f, a)` nodes. For each, predict `h_value(App(f, a),
     i)` from `(h_struct(f), h_struct(a), h_value(f, i), h_value(a,
     i))`, loss = MSE against detached actual `h_value(App(f, a), i)`.
   - **Value head**: for each visited step `t` (action `a_t` creating
     node `n_t = App(f, a)`), `loss_V(n_t) = (V(n_t) − C_t)²`. Trains
     on *every* episode, solved or not — V learns "for this node, what
     was the credit when it was created?". Action-conditional baseline.
   - **Q-head (A2C-MC actor with structural credit)**: **only
     computed on successful trajectories.** For each visited step `t`
     with chosen action `a_t` (creating `n_t`),
     `loss_π(s_t, a_t) = −log π(a_t|s_t) · (C_t − V(n_t).detach()) − c_H · H[π(·|s_t)]`
     where `H` is policy entropy and `c_H` is an entropy bonus
     (default `c_H = 0.01`). `π(·|s_t)` is the softmax over frontier
     candidates at `s_t`. Solution-tree actions get advantage
     `r_searcher − V(s_t)` (positive when state was less promising
     than the outcome); dead-end actions in the same trajectory get
     `−V(s_t)` (penalty proportional to how promising V thought the
     state was). Failed trajectories contribute zero actor loss.
   - **Poser head** (REINFORCE with EMA baseline): for each
     construction step `t`,
     `loss_poser(t) = −log π_poser(a_t | s_t) · (r_poser − V̄_poser)`
     where `V̄_poser` is an exponential moving average of `r_poser`
     across recent dreams (EMA decay 0.99). Stop-grad at the trunk
     (only poser-head params get gradient from this loss).
   - **SIGReg**: as described, weight `λ ≈ 0.05`.
5. **One optimizer step** on the summed loss.

Cold-start: the q-head starts random → first few hundred searches
solve nothing → the actor loss is silent (we don't update the policy
on failed trajectories — see "Per-action credit" above). The system
still makes progress every iteration, just through different heads:

- **Forward head + SIGReg** train the trunk on every iteration
  regardless of search outcome, so the embeddings develop useful
  structure from step one.
- **Value head** trains on every visited state with `C_t = 0` as
  target → learns "expected credit is zero from here" everywhere, an
  uninformative but stable baseline that won't push the policy in bad
  directions.
- **Poser** receives `r_poser = 0` (no solve → tent reward zero) on
  every failed search. Its REINFORCE gradient pushes it away from
  its current shapes, toward simpler programs that the random-q
  searcher *can* solve by brute force. The `<stop>` init bias
  (`+3.0`) gives it a head-start in the right direction.

Once the poser produces a trivially-solvable program and the random
q happens to find it, the first successful trajectory arrives, the
actor loss fires for the first time, and the policy starts learning.
From there the system bootstraps.

## Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `max_poser_nodes` | 6 | Hard cap on `N_poser` |
| `max_budget` | 500 | Frontier-expansion ceiling for searcher |
| `α` | 10 | Solution-length weight in `search_cost` |
| `β` | 8 | Poser peak multiplier (peak at `β · N_poser`) |
| `small_floor` | 0.05 | Poser reward floor for valid programs |
| `λ_sigreg` | 0.05 | SIGReg loss weight |
| `M_slices` | 1024 | SIGReg projection count |
| `EP_quad_points` | 17 | Epps–Pulley quadrature |
| `c_H` (q-head entropy bonus) | 0.01 | Encourages exploration in the actor loss |
| `K_examples` | 3 | I/O pairs per dream |
| `D_dreams_per_iter` | 16 | Batch size |
| `poser_stop_init_bias` | +3.0 | Logit bias on `<stop>` at init for early-iter simplicity |
| `ema_decay` (poser baseline) | 0.99 | EMA decay for `V̄_poser` |

`max_budget` is the iteration-speed knob; everything else has a
principled default to start from.

## Implementation

Two steps: a tiny stopgap fix, then a single rip-and-replace of the
training stack to the end-state design above.

### Step 1 — stopgap: recursive closure filter

Fix `dream.rs:307` to reject examples whose outputs contain a closure
*anywhere* in the list/pair tree (not just at the top level). Cheap;
takes a few minutes. This is done first so the codebase is in a sane
state before the larger rewrite — but the existing complexity-curriculum
metrics are not used as a benchmark for what follows. The old
performance numbers were poisoned by the same bug, so they aren't a
meaningful baseline.

### Step 2 — rip and replace

Implement the end-state design directly. No intermediate
configurations, no comparison against the old training stack.

**Add (new code):**

- `crates/neural/src/sigreg.rs` — vendor the Epps–Pulley test + slicing
  from `github.com/rbalestr-lab/lejepa` (~50 lines core).
- `crates/neural/src/forward_head.rs` — MLP head predicting
  `h_value(App(f, a), i)` from
  `(h_struct(f), h_struct(a), h_value(f, i), h_value(a, i))`.
- `crates/neural/src/value_head.rs` — MLP head predicting per-action
  credit. Input: `h_struct(node)` (single node, `(1, N)`). Output:
  scalar `V(node)`.
- `crates/neural/src/poser_head.rs` — MLP head scoring `(f, a)`
  candidates with the same input shape as the q-head
  (`[ctx, h_struct(f), h_struct(a)]`), trained against the poser tent
  reward. Used as the priority function in the poser-search.
- `crates/training/src/actor_critic.rs` — A2C-MC actor + value losses
  for the q-head (policy term) and value head (regression target),
  computed from search trajectories. Also handles poser REINFORCE
  with EMA baseline.
- `crates/training/src/self_play.rs` — top-level training loop that
  composes all losses and orchestrates one iteration.

**Modify:**

- `crates/lang/src/builtin.rs` — add a new `Stop` builtin with
  identity eval semantics (`stop(x) = x`). Added to the seed library
  via `seed_builtin_library`. Used by the poser to terminate
  construction.
- `crates/neural/src/network.rs` — register the three new heads;
  expose forward/value/poser forward methods alongside `q_score`.
- `crates/neural/src/train.rs` — replace `train_step` with the
  multi-loss version (forward + value + q + poser + SIGReg).
- `crates/search/src/` — additions to the search loop:
  1. **Frontier dedup against pool** is already present in
     `guided.rs` (the `pool.contains` checks at lines 120 and 210).
     Confirm coverage when adding trajectory recording.
  2. **New training-mode entry point.** Add
     `solve_guided_training(arena, lib, task, cfg, scoring_head,
     temp, max_steps)` that runs the same search but: (a) at each
     frontier expansion, peek the top-K candidates, softmax over their
     priorities at temperature `temp`, sample one, record
     `log π(a|s)` and the K candidates; (b) emit a `Trajectory`
     containing the per-step record + final outcome + `S_pool` +
     solution DAG (if solved). The `scoring_head` argument is either
     `q-head` (for searcher-search) or `poser-head` (for
     poser-search). The existing `solve_guided` stays unchanged for
     eval/argmax use.
  3. **Stop primitive support** in the search loop: when a poser-
     search expansion produces `App(stop, n)`, terminate construction
     and return `n` as the program (instead of evaluating
     `App(stop, n)` and continuing).
- `crates/cli/src/` — replace `complexity_main.rs` with a new
  `self_play_main.rs` (`graph-seek-self-play` binary). Flags are
  the hyperparameters from the table above plus `--seed`,
  `--iterations`, `--out`.

**Delete:**

- `crates/training/src/complexity.rs` — explicit complexity loop
  and top-1-target-shrink machinery.
- `crates/training/src/curriculum.rs` — size-stage schedule.
- `crates/training/src/trajectory.rs::dream_to_samples` and its
  supervised-trajectory infrastructure.
- `[[bin]] graph-seek-complexity` in `crates/cli/Cargo.toml` and
  `crates/cli/src/complexity_main.rs`.

**Keep:**

- `crates/training/src/dream.rs` — `sample_dream` and the PCFG.
  Demoted to a tooling utility, used by `graph-seek-show-dreams` for
  inspection. Not in the training path.
- Tests for the dream sampler stay.

Cold-start handling stays as specified in the training-loop section:
poser logits initialized with `<stop>` bias so the first hundreds of
iterations produce trivial programs that even a random q-head can
solve, generating the first positive-advantage signal for the actor.
No separate BC warmup phase — the forward head + SIGReg train the
trunk from step one regardless of search outcome, which gives q a
useful prior to build on.

## Open risks

- **REINFORCE variance at long episodes.** With `S_pool` up to
  `max_budget = 500`, the q-head's policy-gradient estimator has
  long-horizon variance. The value-head baseline is our first line of
  defense; if it's not enough we can add (a) reward discounting
  `γ < 1`, (b) generalized advantage estimation (GAE), or (c) move to
  PPO's clipped objective. None of these are necessary up front.
- **Cold-start: zero successful trajectories.** The actor is gated
  on search success, so until the poser produces a program that
  random q can stumble onto, the policy doesn't learn at all. If the
  `<stop>`-biased poser doesn't trigger the first successful search
  within ~1000 iterations, either bump `poser_stop_init_bias` or
  manually inject a few trivial supervised trajectories to bootstrap.
- **Poser may exploit reward shaping in unexpected ways.** Standard
  RL pathology. Concrete examples we should watch for: programs that
  trigger fuel-exhaustion `⊥` *just barely past* validation (gaming
  the indicator), or programs with input distributions that
  consistently land on edge cases. Monitor poser-output diversity and
  manual-inspect samples periodically.
- **SIGReg's distributional target may fight the q-head's
  preferred geometry.** Embedding norm becomes constrained by
  `N(0, I)` rather than by what the q-head wants. The paper claims
  λ tolerates two orders of magnitude variation, but we should still
  ablate.
- **`β = 8` is a guess.** Probably needs empirical tuning. Worst case
  this is a tunable parameter we adjust per training run; not a
  fundamental risk.

## How we know it's working

Per-iteration logging:

- `r_searcher`, `r_poser` (mean over batch).
- Mean `S_pool`, `S_sol` on successful searches; fraction of searches
  that solve.
- Mean `N_poser` (current poser-program length distribution — should
  grow monotonically as training proceeds).
- Forward-head MSE.
- SIGReg loss value.
- Per-dim variance of `h_value` (sanity check that SIGReg keeps the
  trunk from collapsing).
- Q-head actor loss, value-head MSE, and policy entropy (entropy
  should be high early, decrease as q sharpens; if it collapses to
  near-zero too fast, increase `c_H`).
- Mean advantage `C_t − V(s_t)` magnitude (proxy for how much signal
  the actor is actually getting — should be non-trivial once searches
  start solving).
- Per-trajectory ratio of solution-tree-actions to total actions
  (= `|S_nodes| / S_pool`). Approaches 1 as the searcher gets
  efficient. A useful "branching efficiency" diagnostic.

Success indicators, in rough order of appearance:

1. Forward-head MSE descends below "predict the mean" baseline within
   the first ~100 iterations.
2. Per-dim `h_value` variance stays away from zero.
3. Search starts solving non-trivial poser programs (`N_poser ≥ 3`).
4. Mean `N_poser` grows over time — the implicit curriculum is
   visible.
5. The system reaches `N_poser = max_poser_nodes` consistently with
   nonzero solve-rate.

If (1) and (2) fail: SIGReg / forward-head wiring is broken — debug
before going further. If (3) doesn't happen within ~1000 iterations:
either q is not getting positive-advantage signal (check that *any*
search is solving) or the cold-start handling is insufficient (bump
`poser_stop_init_bias`).

Manual-inspect tooling: `graph-seek-show-dreams` already exists and
reads PCFG-sampled dreams; add a sibling `graph-seek-show-poser` that
samples from the current poser-head and prints `(N_poser, program,
examples, r_poser)` per sample. Run it periodically to sanity-check
that the poser hasn't degenerated into reward-hacking shapes.
