//! Top-level self-play training loop.
//!
//! One iteration:
//! 1. For each of `dreams_per_iter` dreams:
//!    a. Sample an `InputKind` and `K` probe inputs.
//!    b. Run a **poser-search** with those inputs → trajectory + a
//!       constructed program `n` (the argument of `App(stop, n)`).
//!    c. Evaluate `n` on the inputs → I/O examples. Validate: no ⊥,
//!       no nested closures.
//!    d. If valid: run a **q-search** on `(inputs, outputs)` →
//!       trajectory and possibly a solution.
//!    e. Compute `r_searcher`, `r_poser`, `N_poser`.
//! 2. Assemble losses:
//!    - **Forward head**: per `App` node in `n`, per example, MSE
//!      between predicted and (detached) actual `h_value`.
//!    - **Q-head + value-head** (A2C-MC): on the q-trajectory.
//!    - **Poser-head** (REINFORCE with EMA baseline, stop-grad trunk):
//!      on the poser-trajectory.
//!    - **SIGReg**: on all `h_value` tensors collected across dreams.
//! 3. One optimizer step.
//!
//! Cold-start: the q-head starts random, so the first hundred or so
//! q-searches find nothing → `actor_loss` for the q-head is silent
//! (skipped on failure). The trunk still trains every iteration via
//! forward-head + SIGReg + value-head. As the poser learns to produce
//! simpler programs (via REINFORCE on its tent reward), the q-search
//! eventually solves one and the policy gradient kicks in.

use candle_core::{DType, Result, Tensor};
use candle_nn::{AdamW, Optimizer};

use lang::arena::{Arena, NodeId};
use lang::eval::{eval, Value};
use lang::ir::NodeKind;
use lang::library::Library;

use neural::{
    sigreg_loss, EmbedCache, Network, Rng, SigRegCfg,
};

use search::{
    solve_guided_teacher_forced, solve_guided_training, GuidedConfig, ScoringHead,
    SearchConfig, SearchMode, Trajectory, TrainingCfg,
};

use crate::actor_critic::{
    actor_critic_loss, poser_reward, searcher_reward, AcLossCfg, Baseline,
};
use crate::dream::{sample_input_kind, sample_input_of_kind};
use crate::gold::{sample_gold, GoldCategory};

/// Hyperparameters for one self-play iteration.
#[derive(Clone, Debug)]
pub struct SelfPlayCfg {
    /// Hard cap on poser program node count.
    pub max_poser_nodes: u32,
    /// Hard cap on q-search frontier expansions.
    pub max_budget: u32,
    /// `r_searcher = max(0, 1 − (S_pool + α · S_sol) / max_budget)`.
    pub alpha: f32,
    /// Poser tent peak = `β · N_poser`.
    pub beta: f32,
    /// Floor on poser reward for valid programs.
    pub small_floor: f32,
    /// Weight on the SIGReg auxiliary loss.
    pub lambda_sigreg: f32,
    /// SIGReg internal config.
    pub sigreg: SigRegCfg,
    /// Top-K + temperature for the action-sampling search.
    pub training_search: TrainingCfg,
    /// Temperature + entropy bonus for the actor-critic loss.
    pub ac: AcLossCfg,
    /// Frontier / pool caps used by both poser- and q-searches.
    pub guided: GuidedConfig,
    /// Number of dreams per iteration (=batch size).
    pub dreams_per_iter: usize,
    /// Number of I/O examples per dream.
    pub examples_per_dream: usize,
    /// Eval fuel per primitive call.
    pub fuel: u32,
    /// Time budget per search.
    pub time_budget_secs: u64,
    /// EMA decay for the poser baseline.
    pub poser_ema_decay: f32,
    /// If true, skip the poser entirely and draw tasks from
    /// `gold::sample_gold`. The q-head, value-head, forward-head,
    /// and SIGReg still train; the poser-head is bypassed.
    pub use_gold_only: bool,
    /// If true, when the sampled q-search fails to find a solution we
    /// replay the search teacher-forced down the *canonical* program
    /// (the poser's own program in self-play mode, or the gold task's
    /// recipe in gold-only mode) and add a behaviour-cloning loss to
    /// the q-head. This gives the node-picker policy gradient even on
    /// dreams it can't yet solve — and self-disables on the dreams it
    /// can. See `actor_critic_loss` called with `reward = 1.0` and a
    /// `Constant(0.0)` baseline (advantage ≡ 1 ⇒ cross-entropy).
    pub teacher_forcing: bool,
}

impl Default for SelfPlayCfg {
    fn default() -> Self {
        Self {
            max_poser_nodes: 6,
            max_budget: 500,
            alpha: 10.0,
            beta: 8.0,
            small_floor: 0.05,
            lambda_sigreg: 0.05,
            sigreg: SigRegCfg::default(),
            training_search: TrainingCfg {
                top_k: 16,
                temperature: 1.0,
                max_steps: 500,
            },
            ac: AcLossCfg::default(),
            guided: GuidedConfig::default(),
            dreams_per_iter: 16,
            examples_per_dream: 3,
            fuel: 50_000,
            time_budget_secs: 30,
            poser_ema_decay: 0.99,
            use_gold_only: false,
            teacher_forcing: true,
        }
    }
}

/// EMA baseline for the poser's REINFORCE update.
#[derive(Clone, Debug)]
pub struct EmaBaseline {
    pub current: f32,
    pub decay: f32,
}

impl EmaBaseline {
    pub fn new(initial: f32, decay: f32) -> Self {
        Self { current: initial, decay }
    }
    pub fn update(&mut self, value: f32) {
        self.current = self.decay * self.current + (1.0 - self.decay) * value;
    }
    pub fn value(&self) -> f32 { self.current }
}

/// Per-iteration diagnostics.
#[derive(Clone, Debug, Default)]
pub struct SelfPlayStats {
    pub iter: usize,
    /// Mean `r_searcher` across the batch (zero-imputed for failed
    /// trajectories).
    pub mean_r_searcher: f32,
    pub mean_r_poser: f32,
    /// Mean `S_pool` across batch (q-search frontier expansions).
    pub mean_s_pool: f32,
    /// Mean `S_sol` across solved trajectories.
    pub mean_s_sol: f32,
    /// Mean `N_poser` across the batch (size of poser's program).
    pub mean_n_poser: f32,
    /// Fraction of dreams where the q-search solved.
    pub fraction_solved: f32,
    /// Fraction of dreams where the poser produced a valid program.
    pub fraction_valid: f32,
    /// Forward-head MSE (mean).
    pub forward_mse: f32,
    /// SIGReg loss value.
    pub sigreg_value: f32,
    /// Mean policy entropy across q-trajectory steps.
    pub policy_entropy: f32,
    /// Mean advantage magnitude across q-trajectory steps.
    pub mean_advantage: f32,
    /// Total loss after summing all components.
    pub total_loss: f32,
    /// Number of dreams that produced *any* training signal (had at
    /// least a non-trivial poser trajectory).
    pub usable_dreams: usize,
    /// Number of dreams where the q-search failed but teacher forcing
    /// successfully reconstructed the canonical program, contributing a
    /// behaviour-cloning loss to the q-head.
    pub bc_dreams: usize,
    /// Mean behaviour-cloning loss (cross-entropy) across those dreams.
    pub mean_bc_loss: f32,
    /// Per-gold-category breakdown (gold-only mode). Empty in self-play
    /// mode, where dreams have no category. Ordered by
    /// `GoldCategory::ALL`.
    pub per_category: Vec<CategoryStat>,
}

/// Per-category diagnostics for one iteration (gold-only mode).
#[derive(Clone, Copy, Debug)]
pub struct CategoryStat {
    pub category: GoldCategory,
    /// Dreams sampled from this category this iteration.
    pub dreams: usize,
    /// Of those, how many the sampled q-search solved on its own.
    pub solved: usize,
    /// Of those, how many fell back to teacher-forced behaviour cloning.
    pub bc_fired: usize,
}

impl CategoryStat {
    /// Fraction of this category's dreams the q-search solved.
    pub fn solve_rate(&self) -> f32 {
        if self.dreams == 0 { 0.0 } else { self.solved as f32 / self.dreams as f32 }
    }
}

/// Run one training iteration. Returns per-iteration diagnostics.
pub fn train_self_play_iter(
    iter: usize,
    net: &Network,
    opt: &mut AdamW,
    lib: &Library,
    rng: &mut Rng,
    poser_baseline: &mut EmaBaseline,
    cfg: &SelfPlayCfg,
) -> Result<SelfPlayStats> {
    let mut stats = SelfPlayStats {
        iter,
        ..Default::default()
    };

    // One arena per iteration. Shared across all dreams.
    let mut arena = Arena::new();

    // Run all the searches first (collects trajectories + outcomes).
    let mut dreams: Vec<DreamRecord> = Vec::with_capacity(cfg.dreams_per_iter);
    for _ in 0..cfg.dreams_per_iter {
        let dream = run_one_dream(&mut arena, lib, rng, net, cfg);
        dreams.push(dream);
    }

    // Now compute losses across all dreams. Each dream contributes:
    // - forward-head terms (per App node in poser program × examples)
    // - q actor + value loss (if it has a q-trajectory)
    // - poser actor loss (if it has a poser-trajectory with solution)
    // - SIGReg accumulator: h_value tensors collected via cache.
    //
    // Each loss component is collected separately and reduced to a
    // MEAN inside its own bucket before being summed into the final
    // loss. Without this normalisation the per-iter gradient scales
    // with batch size and trajectory length, which spikes hard on
    // the first iter and trashes the weights.
    let mut forward_terms: Vec<Tensor> = Vec::new();
    let mut q_actor_terms: Vec<Tensor> = Vec::new();
    let mut q_value_terms: Vec<Tensor> = Vec::new();
    let mut poser_actor_terms: Vec<Tensor> = Vec::new();
    let mut bc_terms: Vec<Tensor> = Vec::new();
    let mut h_value_batch: Vec<Tensor> = Vec::new();

    let mut sum_r_searcher = 0.0f32;
    let mut sum_r_poser = 0.0f32;
    let mut sum_s_pool = 0.0f32;
    let mut sum_s_sol = 0.0f32;
    let mut count_s_sol = 0usize;
    let mut sum_n_poser = 0.0f32;
    let mut solved_count = 0usize;
    let mut valid_count = 0usize;
    let mut usable_count = 0usize;
    let mut sum_entropy = 0.0f32;
    let mut entropy_div = 0usize;
    let mut sum_advantage = 0.0f32;
    let mut advantage_div = 0usize;
    let mut sum_forward_mse = 0.0f32;
    let mut forward_mse_count = 0usize;
    let mut bc_dreams = 0usize;
    let mut sum_bc_loss = 0.0f32;
    // Per-category (dreams, solved, bc_fired), indexed by position in
    // `GoldCategory::ALL`. Untouched in self-play mode (category None).
    let mut cat_acc: [(usize, usize, usize); GoldCategory::ALL.len()] =
        [(0, 0, 0); GoldCategory::ALL.len()];

    for dream in &dreams {
        let DreamRecord {
            inputs,
            outputs,
            valid,
            poser_traj,
            q_traj,
            bc_traj,
            n_poser,
            r_searcher,
            r_poser,
            category,
        } = dream;

        if let Some(cat) = category {
            if let Some(i) = GoldCategory::ALL.iter().position(|c| c == cat) {
                cat_acc[i].0 += 1;
                if q_traj.as_ref().is_some_and(|t| t.solution.is_some()) {
                    cat_acc[i].1 += 1;
                }
                if bc_traj.is_some() {
                    cat_acc[i].2 += 1;
                }
            }
        }

        sum_r_searcher += r_searcher;
        sum_r_poser += r_poser;
        if *valid { valid_count += 1; }
        if let Some(q_traj) = q_traj {
            sum_s_pool += q_traj.s_pool as f32;
            if let Some(sol) = &q_traj.solution {
                solved_count += 1;
                sum_s_sol += sol.size as f32;
                count_s_sol += 1;
            }
        }
        if let Some(poser_traj) = poser_traj {
            if poser_traj.solution.is_some() {
                sum_n_poser += *n_poser as f32;
                usable_count += 1;
            }
        }

        // Per-dream cache: re-used across the dream's losses so trunk
        // computations are deduplicated.
        let mut cache = EmbedCache::default();

        // --- Forward-head loss: self-supervised on the trunk's
        // dynamics. For every `App` node the q-search produced, for
        // every example, predict `h_value(App, i)` from the children's
        // embeddings and MSE against the trunk's own (detached) value.
        // Decoupled from any specific "ground-truth program" — works
        // in both gold-only and self-play modes. The q-search's pool
        // gives many more training samples than the poser's program
        // alone, and a failed q-search still contributes here.
        if let Some(q_traj) = q_traj {
            let mut seen: rustc_hash::FxHashSet<lang::arena::NodeId> = Default::default();
            for step in &q_traj.steps {
                let node = step.created_node;
                if !seen.insert(node) { continue; }
                let (func, arg) = match arena.kind(node) {
                    NodeKind::App { func, arg } => (*func, *arg),
                    _ => continue,
                };
                for (i, input) in inputs.iter().enumerate() {
                    let pred = net.forward_predict(
                        func, arg, i, input, &arena, lib, cfg.fuel, &mut cache,
                    )?;
                    let actual = neural::h_value(
                        node, i, input, &arena, lib,
                        &net.leaves, &net.app_net, net.lp, cfg.fuel, &mut cache,
                    )?;
                    let target = actual.detach();
                    let diff = (pred - target)?;
                    let term = diff.mul(&diff)?.mean_all()?; // 0-d
                    sum_forward_mse += scalar(&term)?;
                    forward_mse_count += 1;
                    forward_terms.push(term);
                }
            }
        }

        // --- Q-head + value-head A2C-MC loss (only if we have a
        // q-trajectory at all).
        if let Some(q_traj) = q_traj {
            let out = actor_critic_loss(
                net, &arena, lib, q_traj, *r_searcher,
                ScoringHead::Q, Baseline::ValueHead, false,
                inputs, Some(outputs),
                &cfg.ac, cfg.fuel, &mut cache,
            )?;
            if let Some(actor) = out.actor_loss { q_actor_terms.push(actor); }
            if let Some(value) = out.value_loss { q_value_terms.push(value); }
            sum_entropy += out.mean_entropy * out.num_steps as f32;
            entropy_div += out.num_steps;
            sum_advantage += out.mean_advantage.abs() * out.num_steps as f32;
            advantage_div += out.num_steps;
        }

        // --- Q-head behaviour-cloning loss on the teacher-forced
        // canonical trajectory (only on dreams the sampled q-search
        // failed). With `reward = 1.0` and a `Constant(0.0)` baseline,
        // every forced step has advantage 1, so the actor term reduces
        // to `mean(−log π(canonical_action))` — plain cross-entropy.
        // No value loss (constant baseline); trunk gradient flows
        // (cooperative head, so `stop_grad_trunk = false`).
        if let Some(bc_traj) = bc_traj {
            let out = actor_critic_loss(
                net, &arena, lib, bc_traj, 1.0,
                ScoringHead::Q, Baseline::Constant(0.0), false,
                inputs, Some(outputs),
                &cfg.ac, cfg.fuel, &mut cache,
            )?;
            if let Some(actor) = out.actor_loss {
                sum_bc_loss += scalar(&actor)?;
                bc_terms.push(actor);
                bc_dreams += 1;
            }
        }

        // --- Poser-head REINFORCE loss with EMA baseline + stop-grad.
        if let Some(poser_traj) = poser_traj {
            if poser_traj.solution.is_some() {
                let out = actor_critic_loss(
                    net, &arena, lib, poser_traj, *r_poser,
                    ScoringHead::Poser, Baseline::Constant(poser_baseline.value()), true,
                    inputs, None,
                    &cfg.ac, cfg.fuel, &mut cache,
                )?;
                if let Some(actor) = out.actor_loss { poser_actor_terms.push(actor); }
                // No value_loss for Constant baseline.
            }
        }

        // Update the poser EMA after using the prior value for
        // baselining. Skip update for dreams that didn't even produce
        // a valid poser program — we don't want zeros to drag the
        // baseline arbitrarily low.
        if *valid {
            poser_baseline.update(*r_poser);
        }

        // --- Collect h_value tensors for SIGReg.
        for t in cache.h_value.values() {
            h_value_batch.push(t.clone());
        }
    }

    // Reduce each per-dream-summed bucket to a *mean across dreams*.
    // (Each `actor_critic_loss` call already produced a per-step
    // mean; this second mean averages those per-trajectory means.)
    let mut total_terms: Vec<Tensor> = Vec::new();
    if !forward_terms.is_empty() {
        let m = mean_tensors(&forward_terms)?;
        total_terms.push(m);
    }
    if !q_actor_terms.is_empty() {
        let m = mean_tensors(&q_actor_terms)?;
        total_terms.push(m);
    }
    if !q_value_terms.is_empty() {
        let m = mean_tensors(&q_value_terms)?;
        total_terms.push(m);
    }
    if !poser_actor_terms.is_empty() {
        let m = mean_tensors(&poser_actor_terms)?;
        total_terms.push(m);
    }
    if !bc_terms.is_empty() {
        let m = mean_tensors(&bc_terms)?;
        total_terms.push(m);
    }

    // SIGReg: stack collected h_values into (B, N), apply loss.
    if !h_value_batch.is_empty() {
        let stacked = Tensor::cat(
            &h_value_batch.iter().collect::<Vec<_>>(),
            0,
        )?; // (B, N)
        let sigreg = sigreg_loss(&stacked, &cfg.sigreg)?;
        stats.sigreg_value = scalar(&sigreg)?;
        let weighted = sigreg.affine(cfg.lambda_sigreg as f64, 0.0)?;
        total_terms.push(weighted);
    }

    // --- Sum, backward, step.
    if !total_terms.is_empty() {
        let mut acc = total_terms[0].clone();
        for t in &total_terms[1..] {
            acc = (acc + t)?;
        }
        stats.total_loss = scalar(&acc)?;
        let grads = acc.backward()?;
        opt.step(&grads)?;
    }

    // Aggregate stats.
    let n = cfg.dreams_per_iter as f32;
    stats.mean_r_searcher = sum_r_searcher / n;
    stats.mean_r_poser = sum_r_poser / n;
    stats.mean_s_pool = sum_s_pool / n;
    stats.mean_s_sol = if count_s_sol > 0 { sum_s_sol / count_s_sol as f32 } else { 0.0 };
    stats.mean_n_poser = if usable_count > 0 { sum_n_poser / usable_count as f32 } else { 0.0 };
    stats.fraction_solved = solved_count as f32 / n;
    stats.fraction_valid = valid_count as f32 / n;
    stats.policy_entropy = if entropy_div > 0 { sum_entropy / entropy_div as f32 } else { 0.0 };
    stats.mean_advantage = if advantage_div > 0 { sum_advantage / advantage_div as f32 } else { 0.0 };
    stats.forward_mse = if forward_mse_count > 0 { sum_forward_mse / forward_mse_count as f32 } else { 0.0 };
    stats.usable_dreams = usable_count;
    stats.bc_dreams = bc_dreams;
    stats.mean_bc_loss = if bc_dreams > 0 { sum_bc_loss / bc_dreams as f32 } else { 0.0 };
    stats.per_category = GoldCategory::ALL
        .iter()
        .enumerate()
        .filter(|(i, _)| cat_acc[*i].0 > 0)
        .map(|(i, &cat)| {
            let (dreams, solved, bc_fired) = cat_acc[i];
            CategoryStat { category: cat, dreams, solved, bc_fired }
        })
        .collect();

    Ok(stats)
}

/// Per-dream artifacts gathered before loss assembly.
struct DreamRecord {
    inputs: Vec<Value>,
    outputs: Vec<Value>, // valid outputs of n on inputs; empty if invalid
    valid: bool,
    poser_traj: Option<Trajectory>,
    q_traj: Option<Trajectory>,
    /// Teacher-forced trajectory down the canonical program. `Some`
    /// only when the sampled q-search failed *and* forcing
    /// reconstructed the program. Feeds the q-head behaviour-cloning
    /// loss. The recorded `solution` is the canonical program itself.
    bc_traj: Option<Trajectory>,
    n_poser: u32,
    r_searcher: f32,
    r_poser: f32,
    /// Gold category, for per-category diagnostics. `None` in self-play
    /// mode (poser-generated dreams have no category).
    category: Option<GoldCategory>,
}

/// Replay a failed q-search teacher-forced down a canonical program to
/// produce a behaviour-cloning trajectory for the q-head.
///
/// Returns `None` when forcing is disabled, the sampled search already
/// solved, there's no canonical program, or forcing failed to
/// reconstruct it (e.g. a canonical node never entered the frontier).
/// `canonical` is the ground-truth program's root node in `arena`.
#[allow(clippy::too_many_arguments)]
fn teacher_force_bc(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    net: &Network,
    cfg: &SelfPlayCfg,
    scfg: &SearchConfig,
    inputs: &[Value],
    outputs: &[Value],
    q_solved: bool,
    canonical: Option<NodeId>,
) -> Option<Trajectory> {
    if !cfg.teacher_forcing || q_solved {
        return None;
    }
    let canonical = canonical?;
    // Build order: the canonical program's App nodes, children-first.
    let targets: Vec<NodeId> = arena
        .reachable_topo(canonical)
        .into_iter()
        .filter(|&id| matches!(arena.kind(id), NodeKind::App { .. }))
        .collect();
    if targets.is_empty() {
        // Canonical is a bare seed (e.g. a constant/projection); the
        // q-search's seed-match shortcut already covers these, so there
        // is nothing to clone.
        return None;
    }
    let mut tcfg = cfg.training_search.clone();
    // One forced expansion per target; the root terminates via
    // value-match on its final step.
    tcfg.max_steps = targets.len() as u32 + 1;
    let traj = solve_guided_teacher_forced(
        arena, lib, scfg, net, &cfg.guided, inputs, Some(outputs), &tcfg, &targets, rng,
    );
    // Only useful if forcing actually rebuilt the program.
    traj.solution.as_ref()?;
    Some(traj)
}

fn run_one_dream(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    net: &Network,
    cfg: &SelfPlayCfg,
) -> DreamRecord {
    if cfg.use_gold_only {
        return run_one_gold_dream(arena, lib, rng, net, cfg);
    }
    let input_kind = sample_input_kind(rng);
    let inputs: Vec<Value> = (0..cfg.examples_per_dream)
        .map(|_| sample_input_of_kind(rng, input_kind))
        .collect();

    let scfg = SearchConfig {
        time_budget: std::time::Duration::from_secs(cfg.time_budget_secs),
        max_program_size: cfg.training_search.max_steps,
        eval_fuel: cfg.fuel,
        literal_seeds: literal_seed_set(),
        ..SearchConfig::default()
    };

    // Poser-search has a separate cap. We can't make this too tight:
    // even with the cold-start `poser_stop_bias`, the poser still
    // has to land its sample on a stop-candidate at *some* step. At
    // a pool of ~30 leaves the stop-fraction is small, so allow a
    // few hundred steps.
    let mut poser_train_cfg = cfg.training_search.clone();
    poser_train_cfg.max_steps = (cfg.max_poser_nodes * 30).max(120);
    let poser_traj = solve_guided_training(
        arena, lib, &scfg, net, &cfg.guided,
        ScoringHead::Poser, SearchMode::Construct,
        &inputs, None, &poser_train_cfg, rng,
    );

    let (n_node, n_size) = match &poser_traj.solution {
        Some(s) => (Some(s.root), s.size),
        None => (None, 0),
    };

    // Evaluate n on inputs, validate.
    let (outputs, valid) = match n_node {
        Some(n) if n_size <= cfg.max_poser_nodes => evaluate_program(arena, lib, n, &inputs, cfg.fuel),
        Some(_) => (Vec::new(), false), // exceeds poser node cap
        None => (Vec::new(), false),
    };

    // Run q-search only on valid dreams.
    let q_traj = if valid {
        let mut q_train_cfg = cfg.training_search.clone();
        q_train_cfg.max_steps = cfg.max_budget;
        Some(solve_guided_training(
            arena, lib, &scfg, net, &cfg.guided,
            ScoringHead::Q, SearchMode::Solve,
            &inputs, Some(&outputs), &q_train_cfg, rng,
        ))
    } else {
        None
    };

    // Compute rewards.
    let (s_pool, s_sol, solved) = match &q_traj {
        Some(t) => {
            let solved = t.solution.is_some();
            let s_sol = t.solution.as_ref().map(|s| s.size).unwrap_or(0);
            (t.s_pool, s_sol, solved)
        }
        None => (0, 0, false),
    };
    let r_searcher = searcher_reward(s_pool, s_sol, cfg.max_budget, cfg.alpha, solved);
    let r_poser = poser_reward(
        s_pool, n_size, cfg.max_budget, cfg.beta, cfg.small_floor, valid, solved,
    );

    // On a failed q-search, clone the poser's own program (the dream's
    // ground truth) into the q-head via teacher forcing.
    let canonical = if valid { n_node } else { None };
    let bc_traj = teacher_force_bc(
        arena, lib, rng, net, cfg, &scfg, &inputs, &outputs, solved, canonical,
    );

    DreamRecord {
        inputs,
        outputs,
        valid,
        poser_traj: Some(poser_traj),
        q_traj,
        bc_traj,
        n_poser: n_size,
        r_searcher,
        r_poser,
        category: None,
    }
}

/// Gold-only variant of `run_one_dream`: sample task I/O directly
/// from `gold::sample_gold`, skip the poser-search entirely, run the
/// q-search on the sampled `(inputs, outputs)`.
fn run_one_gold_dream(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    net: &Network,
    cfg: &SelfPlayCfg,
) -> DreamRecord {
    let task = sample_gold(rng, cfg.examples_per_dream);

    let scfg = SearchConfig {
        time_budget: std::time::Duration::from_secs(cfg.time_budget_secs),
        max_program_size: cfg.training_search.max_steps,
        eval_fuel: cfg.fuel,
        literal_seeds: literal_seed_set(),
        ..SearchConfig::default()
    };

    // Materialise the gold task's canonical program up front (interns
    // into the shared arena; its seed leaves hash-cons with the
    // q-search's seed pool).
    let canonical = task.build_canonical(arena, lib);

    let mut q_train_cfg = cfg.training_search.clone();
    q_train_cfg.max_steps = cfg.max_budget;

    let q_traj = solve_guided_training(
        arena, lib, &scfg, net, &cfg.guided,
        ScoringHead::Q, SearchMode::Solve,
        &task.inputs, Some(&task.outputs), &q_train_cfg, rng,
    );

    let solved = q_traj.solution.is_some();
    let s_pool = q_traj.s_pool;
    let s_sol = q_traj.solution.as_ref().map(|s| s.size).unwrap_or(0);
    let r_searcher = searcher_reward(s_pool, s_sol, cfg.max_budget, cfg.alpha, solved);

    let bc_traj = teacher_force_bc(
        arena, lib, rng, net, cfg, &scfg, &task.inputs, &task.outputs, solved, canonical,
    );

    DreamRecord {
        inputs: task.inputs,
        outputs: task.outputs,
        valid: true,         // gold tasks are constructed valid by design
        poser_traj: None,    // no poser involvement in gold-only mode
        q_traj: Some(q_traj),
        bc_traj,
        n_poser: 0,
        r_searcher,
        r_poser: 0.0,
        category: Some(task.category),
    }
}

/// Literal seeds the q-search starts with: small ints (matching the
/// gold arith constants' range) plus the two booleans (for bool
/// truth-table tasks).
fn literal_seed_set() -> Vec<lang::ir::LitValue> {
    use lang::ir::LitValue;
    vec![
        LitValue::Int(-3), LitValue::Int(-2), LitValue::Int(-1),
        LitValue::Int(0), LitValue::Int(1), LitValue::Int(2), LitValue::Int(3),
        LitValue::Bool(true), LitValue::Bool(false),
    ]
}

/// Evaluate program `n` on each input. Returns `(outputs, valid)`.
/// Valid means: every output is finite (not Bottom) and contains no
/// `Closure` anywhere in its `List/Pair` tree.
fn evaluate_program(
    arena: &mut Arena,
    lib: &Library,
    n: lang::arena::NodeId,
    inputs: &[Value],
    fuel: u32,
) -> (Vec<Value>, bool) {
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let env = [input.clone()];
        let mut f = fuel;
        let v = match eval(arena, lib, n, &env, &mut f) {
            Ok(v) => v,
            Err(_) => return (Vec::new(), false),
        };
        if v.is_bottom() || contains_closure(&v) {
            return (Vec::new(), false);
        }
        outputs.push(v);
    }
    // We *allow* constant and identity programs through self-play
    // validation. The q-search will pool-shortcut them (S_pool=0) so
    // r_poser collapses to `small_floor` — tiny but nonzero. That
    // beats rejecting them outright (r_poser=0): the poser still
    // gets a faint "you produced something" signal, which matters at
    // cold-start when valid non-trivial programs are vanishingly
    // rare under a random network.
    (outputs, true)
}

fn contains_closure(v: &Value) -> bool {
    match v {
        Value::Closure(_) => true,
        Value::List(xs) => xs.iter().any(contains_closure),
        Value::Pair(p) => contains_closure(&p.0) || contains_closure(&p.1),
        _ => false,
    }
}

fn scalar(t: &Tensor) -> Result<f32> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(v[0])
}

/// Mean of a non-empty list of 0-d tensors.
fn mean_tensors(ts: &[Tensor]) -> Result<Tensor> {
    debug_assert!(!ts.is_empty());
    let mut acc = ts[0].clone();
    for t in &ts[1..] {
        acc = (acc + t)?;
    }
    acc.affine(1.0 / ts.len() as f64, 0.0)
}

#[allow(dead_code)]
fn ensure_dtype(t: &Tensor) -> bool {
    t.dtype() == DType::F32
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use lang::builtin::seed_builtin_library;
    use neural::{make_optimizer, Network, NetworkCfg};

    #[test]
    fn iter_runs_end_to_end() {
        let lib = seed_builtin_library();
        let net_cfg = NetworkCfg { n: 8, ..NetworkCfg::default() };
        let net = Network::new(&net_cfg, &lib, Device::Cpu).unwrap();
        let mut opt = make_optimizer(&net, net_cfg.lr, net_cfg.weight_decay).unwrap();
        let mut rng = Rng::new(0x1234);
        let mut baseline = EmaBaseline::new(0.0, 0.99);

        let cfg = SelfPlayCfg {
            dreams_per_iter: 2,
            examples_per_dream: 2,
            max_poser_nodes: 4,
            max_budget: 40,
            time_budget_secs: 5,
            training_search: TrainingCfg { top_k: 4, temperature: 2.0, max_steps: 40 },
            ..SelfPlayCfg::default()
        };

        let stats = train_self_play_iter(0, &net, &mut opt, &lib, &mut rng, &mut baseline, &cfg)
            .expect("iter should succeed");

        assert!(stats.total_loss.is_finite(), "total loss must be finite, got {}", stats.total_loss);
        assert!(stats.mean_r_searcher >= 0.0);
        assert!(stats.mean_r_poser >= 0.0);
    }

    /// Gold-only + teacher forcing: with a random network the q-search
    /// fails on essentially every dream, so teacher forcing should fire
    /// and contribute behaviour-cloning signal. Verify it runs, fires,
    /// and produces a finite loss / gradient step.
    #[test]
    fn teacher_forcing_gold_only_fires() {
        let lib = seed_builtin_library();
        let net_cfg = NetworkCfg { n: 8, ..NetworkCfg::default() };
        let net = Network::new(&net_cfg, &lib, Device::Cpu).unwrap();
        let mut opt = make_optimizer(&net, net_cfg.lr, net_cfg.weight_decay).unwrap();
        let mut rng = Rng::new(0xBC0FF);
        let mut baseline = EmaBaseline::new(0.0, 0.99);

        let cfg = SelfPlayCfg {
            dreams_per_iter: 8,
            examples_per_dream: 3,
            max_budget: 30, // tiny — guarantees the sampled q-search fails
            time_budget_secs: 5,
            use_gold_only: true,
            teacher_forcing: true,
            training_search: TrainingCfg { top_k: 8, temperature: 1.0, max_steps: 30 },
            ..SelfPlayCfg::default()
        };

        let stats = train_self_play_iter(0, &net, &mut opt, &lib, &mut rng, &mut baseline, &cfg)
            .expect("iter should succeed");

        assert!(stats.total_loss.is_finite(), "total loss must be finite");
        assert!(stats.bc_dreams > 0, "teacher forcing should fire on failed gold dreams");
        assert!(stats.mean_bc_loss.is_finite() && stats.mean_bc_loss >= 0.0);
    }
}
