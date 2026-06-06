//! Train + evaluate the network on dreams of increasing **complexity**,
//! where complexity = number of `App` nodes in the dream's hash-consed
//! DAG.
//!
//! Three design choices worth noting:
//!
//! 1. **Cumulative uniform sampling.** At level `k`, each dream's
//!    complexity is drawn uniformly from `[1, k]` — so reaching a new
//!    level *expands* the training distribution rather than shifting
//!    it. This keeps earlier complexities live and avoids catastrophic
//!    forgetting.
//! 2. **Adaptive pacing tied to solve-rate.** The thing we ultimately
//!    care about is held-out solve-rate from guided search, not
//!    per-step ranking accuracy (top-1). Top-1 over training samples
//!    is a cheap proxy whose relationship to solve-rate depends on
//!    program size, search budget, and frontier dynamics — so we don't
//!    pick a top-1 threshold a priori. Instead, we train until top-1
//!    hits a *current* target, then run the search-eval to measure
//!    actual solve-rate.
//! 3. **Geometric top-1 shrink on miss.** When the eval misses the
//!    solve-rate target, we bump the top-1 target via
//!    `1 − shrink·(1 − current_top1)`. With `shrink = 0.8` this
//!    closes 20% of the remaining error each round, so the increments
//!    shrink as we approach 1.0 (exactly where each percent costs the
//!    most training). The rule keeps tightening the gate until the
//!    eval clears `target_solve_rate`. There is no cap — a level the
//!    network cannot reach blocks indefinitely; Ctrl-C to intervene.

use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;

use lang::arena::{Arena, NodeId};
use lang::ir::{LitValue, NodeKind};
use lang::library::Library;

use candle_nn::AdamW;

use neural::{train_step, Network, Rng, TrainSample};

use search::{solve_guided, GuidedConfig, SearchConfig};
use tasks::{ListExamplesTask, TaskId};

use crate::dream::{sample_dream, DreamCfg, DreamTask};
use crate::trajectory::dream_to_samples;

/// Count `App` nodes reachable from the dream's root.
pub fn app_count(arena: &Arena, root: NodeId) -> usize {
    arena
        .reachable_topo(root)
        .iter()
        .filter(|&&id| matches!(arena.kind(id), NodeKind::App { .. }))
        .count()
}

/// Sample dreams with exactly `complexity` App nodes (in the DAG).
///
/// Each retry uses a fresh arena. Returns the dreams along with the
/// arenas they live in.
pub fn sample_complexity_dreams(
    lib: &Library,
    rng: &mut Rng,
    complexity: usize,
    count: usize,
    cfg: &ComplexityCfg,
) -> Vec<(Arena, DreamTask)> {
    let mut out: Vec<(Arena, DreamTask)> = Vec::new();
    let mut attempts = 0usize;
    let cap = cfg.sample_attempt_cap.max(count * 200);
    while out.len() < count && attempts < cap {
        attempts += 1;
        let mut arena = Arena::new();
        let dream_cfg = DreamCfg {
            // Tree-size of a program with N apps is 2N+1; allocate a
            // size budget around there.
            max_size: ((2 * complexity + 1) as u32).max(2),
            examples_per_dream: cfg.examples_per_dream,
            fuel: cfg.fuel,
            // Disable the fold-template at small complexities since the
            // template alone needs ≥5 App nodes.
            p_template: if complexity >= 5 { 0.6 } else { 0.0 },
            p_app: cfg.p_app,
            max_resample: 100,
            max_bottom_frac: cfg.max_bottom_frac,
            ..DreamCfg::default()
        };
        let dream = match sample_dream(&mut arena, lib, rng, &dream_cfg) {
            Some(d) => d,
            None => continue,
        };
        if app_count(&arena, dream.program) != complexity { continue; }
        out.push((arena, dream));
    }
    out
}

#[derive(Clone, Debug)]
pub struct ComplexityCfg {
    /// Complexity levels to step through, e.g. `[1, 2, 3, 4, 5, 6]`.
    pub levels: Vec<usize>,
    /// Examples per dream (the user wants 3 per dream).
    pub examples_per_dream: usize,
    /// Dreams drawn per training iteration.
    pub dreams_per_iter: usize,
    /// Minimum training iterations per training phase before the
    /// reach test is allowed to fire. A floor on phase length so a
    /// lucky streak in the first few iters doesn't trigger an eval.
    pub min_iters_per_level: usize,
    /// Number of recent training iterations whose C=level samples are
    /// pooled into the Wilson LCB test. Tradeoff: larger windows mean
    /// more confident tests but slower reaction to the network's
    /// current performance.
    pub top1_lcb_iter_window: usize,
    /// Z-score for the Wilson one-sided lower confidence bound. 1.645
    /// = 95% confidence, 2.326 = 99% confidence. The curriculum
    /// advances when `LCB_z(p̂, n) ≥ top1_target` — so this is the
    /// probability of a false advance per check.
    pub top1_lcb_z: f32,
    /// Initial top1@max target when starting a new level. Each
    /// failed eval round tightens this target via geometric shrink.
    pub initial_top1_target: f32,
    /// Fraction of the remaining error preserved on each top-1 bump
    /// after a missed eval. `next = 1 − shrink·(1 − current)`. Default
    /// 0.8 closes 20% of the error per round.
    pub top1_shrink_factor: f32,
    /// The held-out search-eval solve-rate that, once reached at the
    /// current level, advances the curriculum. The actual quantity
    /// we care about — top1@max is a proxy used between evals.
    pub target_solve_rate: f32,
    /// Negatives per training step.
    pub max_negatives: usize,
    /// Per-example eval fuel.
    pub fuel: u32,
    /// Max sample attempts when looking for a dream of a given
    /// complexity. Defaults to `count * 200`.
    pub sample_attempt_cap: usize,
    /// Number of held-out test dreams per complexity level.
    pub eval_dreams_per_level: usize,
    /// Time budget per guided search at evaluation.
    pub eval_search_budget: Duration,
    /// Hash-cons-canonical literal seeds passed to dream/trajectory.
    pub literal_seeds: Vec<LitValue>,
    /// PCFG `p_app` (probability of forming an App).
    pub p_app: f32,
    /// PCFG bottom-fraction tolerance.
    pub max_bottom_frac: f32,
    /// Guided-search max pool size.
    pub guided_pool_cap: usize,
    /// Guided-search max frontier.
    pub guided_max_frontier: usize,
    /// Max program size the search will enumerate during eval.
    pub eval_max_program_size: u32,
}

impl Default for ComplexityCfg {
    fn default() -> Self {
        Self {
            levels: vec![1, 2, 3, 4, 5, 6],
            examples_per_dream: 3,
            dreams_per_iter: 8,
            min_iters_per_level: 10,
            top1_lcb_iter_window: 5,
            top1_lcb_z: 1.645,
            initial_top1_target: 0.95,
            target_solve_rate: 0.50,
            top1_shrink_factor: 0.8,
            max_negatives: 16,
            fuel: 50_000,
            sample_attempt_cap: 4_000,
            eval_dreams_per_level: 20,
            eval_search_budget: Duration::from_secs(8),
            literal_seeds: vec![
                LitValue::Int(-3), LitValue::Int(-2), LitValue::Int(-1),
                LitValue::Int(0), LitValue::Int(1), LitValue::Int(2), LitValue::Int(3),
            ],
            p_app: 0.7,
            max_bottom_frac: 0.5,
            guided_pool_cap: 4_000,
            guided_max_frontier: 30_000,
            eval_max_program_size: 24,
        }
    }
}

/// One row of the report.
#[derive(Clone, Debug)]
pub struct ComplexityRow {
    pub level: usize,
    /// Total training iterations spent at this level (summed across
    /// all train-then-eval rounds).
    pub iters: usize,
    /// Number of search-eval rounds the level needed. `1` means the
    /// first eval at the initial top-1 target already cleared the
    /// solve-rate bar.
    pub eval_rounds: usize,
    /// Top-1@max measured at the eval that cleared the solve-rate bar.
    pub final_top1: f32,
    /// The top-1 *target* the level was training toward at the eval
    /// that cleared. This (not `final_top1`) is what gets carried to
    /// the next level — see the geometric-shrink rule for the reason.
    pub final_top1_target: f32,
    pub eval_total: usize,
    pub eval_solved: usize,
    pub eval_avg_size: f32,
    pub eval_avg_ms: f32,
}

/// Event stream from the curriculum trainer. The CLI prints these;
/// callers that drive the trainer programmatically can filter.
#[derive(Clone, Debug)]
pub enum CurriculumEvent {
    Iter {
        level: usize,
        iter: usize,
        loss: f32,
        top1_overall: f32,
        top1_at_max: Option<f32>,
        top1_target: f32,
    },
    /// Fired just before `eval_complexity` runs at a level. The eval
    /// can take many seconds, so this lets the UI swap to a spinner
    /// instead of leaving the train-phase status frozen.
    EvalStart {
        level: usize,
        round: usize,
    },
    Eval {
        level: usize,
        round: usize,
        measured_top1: f32,
        eval_total: usize,
        eval_solved: usize,
        solve_rate: f32,
        target_solve_rate: f32,
        /// If the eval missed the target, the next top1 target the
        /// curriculum will train toward via geometric shrink.
        /// `None` when the eval passed (we're advancing).
        next_top1_target: Option<f32>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct ComplexityReport {
    pub rows: Vec<ComplexityRow>,
}

impl ComplexityReport {
    pub fn render_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("| Level | Iters | Evals | Final top1 | Eval solved | Solve rate | Avg size | Avg ms |\n");
        s.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for r in &self.rows {
            let solve_pct = if r.eval_total > 0 {
                100.0 * r.eval_solved as f32 / r.eval_total as f32
            } else { 0.0 };
            s.push_str(&format!(
                "| {} | {} | {} | {:.2} | {}/{} | {:.0}% | {:.1} | {:.0} |\n",
                r.level,
                r.iters,
                r.eval_rounds,
                r.final_top1,
                r.eval_solved,
                r.eval_total,
                solve_pct,
                r.eval_avg_size,
                r.eval_avg_ms,
            ));
        }
        s
    }
}

/// Result of one training iteration.
#[derive(Clone, Debug, Default)]
struct IterStats {
    loss: f32,
    /// Top-1 over all samples in this iter (mixed complexities).
    top1_overall: f32,
    /// Top-1 over only the samples drawn from dreams at the current
    /// max complexity. `None` if the iter produced no such samples.
    top1_at_max: Option<f32>,
    /// Per-sample correctness flags for samples drawn from C=max
    /// dreams in this iter. The curriculum loop accumulates these
    /// into a sliding window sized by the current target's error
    /// rate. Empty if no C=max samples this iter.
    at_max_correct: Vec<bool>,
}

/// Run one training iteration of `dreams_per_iter` dreams whose
/// complexities are drawn uniformly from `[1, complexity]`. The
/// returned `top1_at_max` is computed only over samples from C=max
/// dreams — that is the metric the curriculum uses to advance, so it
/// is not diluted by easy samples from lower complexities.
fn train_one_level_iter(
    net: &Network,
    opt: &mut AdamW,
    lib: &Library,
    rng: &mut Rng,
    complexity: usize,
    cfg: &ComplexityCfg,
) -> IterStats {
    // Bucket the per-iter dream requests by uniformly drawn complexity
    // in [1, complexity], then sample each bucket in one batched call.
    let mut counts = vec![0usize; complexity];
    for _ in 0..cfg.dreams_per_iter {
        let c = rng.gen_range(complexity);
        counts[c] += 1;
    }
    let mut dreams: Vec<(Arena, DreamTask, usize)> = Vec::new();
    for (idx, &count) in counts.iter().enumerate() {
        if count == 0 { continue; }
        let c = idx + 1;
        for (arena, dream) in sample_complexity_dreams(lib, rng, c, count, cfg) {
            dreams.push((arena, dream, c));
        }
    }
    if dreams.is_empty() { return IterStats::default(); }

    let mut arenas: Vec<Arena> = Vec::new();
    let mut samples: Vec<TrainSample> = Vec::new();
    let mut arena_idx: Vec<usize> = Vec::new();
    let mut sample_complexity: Vec<usize> = Vec::new();

    for (mut arena, dream, c) in dreams {
        let s = dream_to_samples(
            &mut arena, lib, &dream, &cfg.literal_seeds, rng, cfg.max_negatives,
        );
        if s.samples.is_empty() { continue; }
        let aidx = arenas.len();
        arenas.push(arena);
        for sample in s.samples {
            arena_idx.push(aidx);
            sample_complexity.push(c);
            samples.push(sample);
        }
    }

    if samples.is_empty() { return IterStats::default(); }

    let batch: Vec<(&TrainSample, &Arena, &Library)> = samples.iter().enumerate()
        .map(|(i, s)| (s, &arenas[arena_idx[i]], lib))
        .collect();

    match train_step(net, opt, &batch, cfg.fuel) {
        Ok(stats) => {
            // Bucketed top-1: only samples from C=complexity dreams.
            let mut at_max_correct: Vec<bool> = Vec::new();
            for (i, &c) in sample_complexity.iter().enumerate() {
                if c != complexity { continue; }
                let correct = stats.per_sample_correct.get(i).copied().unwrap_or(false);
                at_max_correct.push(correct);
            }
            let top1_at_max = if at_max_correct.is_empty() {
                None
            } else {
                let c = at_max_correct.iter().filter(|&&b| b).count();
                Some(c as f32 / at_max_correct.len() as f32)
            };
            IterStats {
                loss: stats.loss,
                top1_overall: stats.positive_top1,
                top1_at_max,
                at_max_correct,
            }
        }
        Err(e) => {
            eprintln!("train_step error: {e}");
            IterStats::default()
        }
    }
}

/// Evaluate the network on held-out dreams of `complexity`. For each
/// dream, build a `ListExamplesTask` and run guided search; success =
/// search returns a program whose outputs match the dream's I/O
/// examples.
pub fn eval_complexity(
    net: &Network,
    lib: &Library,
    rng: &mut Rng,
    complexity: usize,
    cfg: &ComplexityCfg,
) -> (usize, usize, f32, f32) {
    let dreams = sample_complexity_dreams(lib, rng, complexity, cfg.eval_dreams_per_level, cfg);
    let total = dreams.len();
    let mut solved = 0;
    let mut size_sum = 0.0;
    let mut ms_sum = 0.0;

    let gcfg = GuidedConfig {
        size_penalty: 0.0,
        guided_pool_cap: cfg.guided_pool_cap,
        max_frontier: cfg.guided_max_frontier,
        priority_floor: f32::NEG_INFINITY,
    };
    // Scale the budget by complexity — the search space grows fast.
    let budget = Duration::from_millis(
        (cfg.eval_search_budget.as_millis() as u64) * (complexity as u64).max(1),
    );
    // Match the search's literal seeds to the dream's literal alphabet
    // so the search can reconstruct dreams that use literals like
    // `Int(2)` or `Int(-3)` that aren't in the default seeds.
    let scfg = SearchConfig {
        time_budget: budget,
        max_program_size: cfg.eval_max_program_size,
        eval_fuel: cfg.fuel,
        literal_seeds: cfg.literal_seeds.clone(),
        ..SearchConfig::default()
    };

    for (_arena, dream) in dreams {
        let task = ListExamplesTask {
            id: TaskId(0xeeee_0000 + complexity as u64),
            examples: dream.examples.clone(),
            fuel: cfg.fuel,
        };
        let mut search_arena = Arena::new();
        let started = Instant::now();
        let r = solve_guided(&mut search_arena, lib, &task, &scfg, net, &gcfg);
        let elapsed = started.elapsed();
        if r.solved {
            solved += 1;
            size_sum += r.size as f32;
        }
        ms_sum += elapsed.as_millis() as f32;
    }

    let avg_size = if solved > 0 { size_sum / solved as f32 } else { 0.0 };
    let avg_ms = if total > 0 { ms_sum / total as f32 } else { 0.0 };
    (total, solved, avg_size, avg_ms)
}

/// Geometric bump rule: shrink the gap to 1.0 by `(1 − shrink)`.
/// Example: `current = 0.89`, `shrink = 0.8` → `0.912`.
///
/// This ignores the value of the observed solve-rate; it just tightens
/// the gate monotonically. There is no cap — a level the network
/// cannot reach blocks indefinitely.
pub fn next_top1_target(current_top1: f32, shrink: f32) -> f32 {
    let next = 1.0 - shrink * (1.0 - current_top1);
    next.min(1.0).max(current_top1)
}

/// Wilson one-sided lower confidence bound on a binomial proportion.
/// Returns the value `p_lo` such that we are `Φ(z)` confident the true
/// rate is at least `p_lo`. `z = 1.645` for 95% one-sided confidence,
/// `2.326` for 99%.
///
/// Replaces the naive `mean ≥ target` test in the curriculum advance
/// criterion. With small `n`, the LCB sits well below `p̂`, which
/// prevents lucky-tail false positives; as `n` grows the bound tightens
/// toward `p̂`. So the test naturally requires more samples for tighter
/// targets without us hand-picking a window size.
pub fn wilson_lcb(p_hat: f32, n: usize, z: f32) -> f32 {
    if n == 0 { return 0.0; }
    let n = n as f32;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p_hat + z2 / (2.0 * n);
    let margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n).sqrt();
    ((center - margin) / denom).max(0.0).min(1.0)
}

/// Train a curriculum of increasing complexity. Returns one report row
/// per level.
///
/// Per-level loop: train until windowed `top1@max` reaches the
/// current target, run the search-eval, decide. On miss: record a
/// `(top1, solve_rate)` calibration point, linearly extrapolate to
/// the new top1 target, train again. The curriculum advances when an
/// eval clears `cfg.target_solve_rate`, and bails (stopping the
/// curriculum) when extrapolation pushes the top1 target past
/// `cfg.max_top1_target` — the network cannot reach that level under
/// the current search budget / network capacity.
///
/// `on_event` receives `CurriculumEvent::Iter` for every training
/// step and `CurriculumEvent::Eval` after each search-eval.
pub fn train_complexity_curriculum(
    net: &Network,
    opt: &mut AdamW,
    lib: &Library,
    rng: &mut Rng,
    cfg: &ComplexityCfg,
    mut on_event: impl FnMut(CurriculumEvent),
) -> ComplexityReport {
    let mut report = ComplexityReport::default();
    let levels = cfg.levels.clone();
    let min_iters = cfg.min_iters_per_level.max(1);
    for level in levels {
        // Carry the previous level's *target* across — higher complexity
        // never needs *less* per-step accuracy than the previous level
        // did, so this saves redundant calibration on level k+1. (We
        // carry the target rather than the measured top-1: a lucky
        // overshoot to 1.00 shouldn't lock the next level into a 1.00
        // target it can never reach.)
        let mut top1_target = report.rows.last()
            .map(|r| r.final_top1_target.max(cfg.initial_top1_target))
            .unwrap_or(cfg.initial_top1_target);
        let mut total_iters = 0usize;
        let mut round = 0usize;
        // Sliding buffer of recent iters' per-sample correctness for
        // C=level samples. Each entry = one iter's bools. We pool over
        // the last `top1_lcb_iter_window` iters for the LCB test.
        let mut iter_samples: VecDeque<Vec<bool>> = VecDeque::new();
        let iter_window = cfg.top1_lcb_iter_window.max(1);
        let row = loop {
            // Phase 1: train until Wilson LCB(p̂, n) ≥ top1_target over
            // the trailing `iter_window` iters of C=level samples.
            let mut phase_iters = 0usize;
            let measured_top1 = loop {
                let stats = train_one_level_iter(net, opt, lib, rng, level, cfg);
                total_iters += 1;
                phase_iters += 1;
                on_event(CurriculumEvent::Iter {
                    level,
                    iter: total_iters - 1,
                    loss: stats.loss,
                    top1_overall: stats.top1_overall,
                    top1_at_max: stats.top1_at_max,
                    top1_target,
                });
                if !stats.at_max_correct.is_empty() {
                    iter_samples.push_back(stats.at_max_correct);
                    while iter_samples.len() > iter_window {
                        iter_samples.pop_front();
                    }
                }
                if phase_iters >= min_iters && iter_samples.len() >= iter_window {
                    let mut n = 0usize;
                    let mut c = 0usize;
                    for buf in &iter_samples {
                        n += buf.len();
                        c += buf.iter().filter(|&&b| b).count();
                    }
                    if n > 0 {
                        let p_hat = c as f32 / n as f32;
                        let lcb = wilson_lcb(p_hat, n, cfg.top1_lcb_z);
                        if lcb >= top1_target {
                            break p_hat;
                        }
                    }
                }
            };

            // Phase 2: search-eval at exactly the level's complexity.
            round += 1;
            on_event(CurriculumEvent::EvalStart { level, round });
            let (eval_total, eval_solved, eval_avg_size, eval_avg_ms) =
                eval_complexity(net, lib, rng, level, cfg);
            let solve_rate = if eval_total > 0 {
                eval_solved as f32 / eval_total as f32
            } else { 0.0 };

            if solve_rate >= cfg.target_solve_rate {
                on_event(CurriculumEvent::Eval {
                    level, round, measured_top1,
                    eval_total, eval_solved, solve_rate,
                    target_solve_rate: cfg.target_solve_rate,
                    next_top1_target: None,
                });
                break ComplexityRow {
                    level,
                    iters: total_iters,
                    eval_rounds: round,
                    final_top1: measured_top1,
                    final_top1_target: top1_target,
                    eval_total, eval_solved, eval_avg_size, eval_avg_ms,
                };
            }

            // Miss: tighten the top-1 *target* (not the measured value)
            // geometrically. Using the target avoids the edge case where
            // an overshoot to 1.00 locks the next round into a 1.00 gate
            // the network can no longer pass.
            let next_target = next_top1_target(top1_target, cfg.top1_shrink_factor);
            on_event(CurriculumEvent::Eval {
                level, round, measured_top1,
                eval_total, eval_solved, solve_rate,
                target_solve_rate: cfg.target_solve_rate,
                next_top1_target: Some(next_target),
            });
            top1_target = next_target;
        };
        report.rows.push(row);
    }
    report
}
