//! Train + evaluate the network on dreams of increasing **complexity**,
//! where complexity = number of `App` nodes in the dream's hash-consed
//! DAG. For each level, train the network on dreams of that complexity,
//! then run guided search on held-out dreams of the same complexity and
//! report the success rate (programs whose outputs match the target
//! examples, not necessarily the same program).
//!
//! The user's prompt: "Starting with a single node, then up to two,
//! then up to three etc. We should see the network able to recreate
//! programs purely from the input/output pairs, three examples per
//! dream."

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
    /// Number of training iterations at each level.
    pub iters_per_level: usize,
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
    /// If true, at level k we sample dreams uniformly from
    /// complexities `1..=k` instead of just `k`. Prevents catastrophic
    /// forgetting of earlier complexity levels.
    pub cumulative: bool,
}

impl Default for ComplexityCfg {
    fn default() -> Self {
        Self {
            levels: vec![1, 2, 3, 4, 5, 6],
            examples_per_dream: 3,
            dreams_per_iter: 8,
            iters_per_level: 30,
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
            cumulative: false,
        }
    }
}

/// One row of the report.
#[derive(Clone, Debug)]
pub struct ComplexityRow {
    pub level: usize,
    pub iters: usize,
    pub train_loss_first: f32,
    pub train_loss_last: f32,
    pub train_top1_first: f32,
    pub train_top1_last: f32,
    pub eval_total: usize,
    pub eval_solved: usize,
    pub eval_avg_size: f32,
    pub eval_avg_ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ComplexityReport {
    pub rows: Vec<ComplexityRow>,
}

impl ComplexityReport {
    pub fn render_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("| Complexity | Iters | Loss start → end | Top-1 start → end | Eval solved | Avg size | Avg ms |\n");
        s.push_str("|---:|---:|---|---|---:|---:|---:|\n");
        for r in &self.rows {
            s.push_str(&format!(
                "| {} | {} | {:.3} → {:.3} | {:.2} → {:.2} | {}/{} ({:.0}%) | {:.1} | {:.0} |\n",
                r.level,
                r.iters,
                r.train_loss_first,
                r.train_loss_last,
                r.train_top1_first,
                r.train_top1_last,
                r.eval_solved,
                r.eval_total,
                if r.eval_total > 0 {
                    100.0 * r.eval_solved as f32 / r.eval_total as f32
                } else { 0.0 },
                r.eval_avg_size,
                r.eval_avg_ms,
            ));
        }
        s
    }
}

/// Run one training iteration of `dreams_per_iter` dreams at the given
/// complexity (or, if `cumulative`, drawing complexities uniformly from
/// `1..=complexity`). Returns (loss, top-1).
fn train_one_level_iter(
    net: &Network,
    opt: &mut AdamW,
    lib: &Library,
    rng: &mut Rng,
    complexity: usize,
    cfg: &ComplexityCfg,
) -> (f32, f32) {
    let dreams = if cfg.cumulative {
        // Draw `dreams_per_iter` dreams across complexities 1..=complexity.
        let mut all: Vec<(Arena, DreamTask)> = Vec::new();
        let per_level = (cfg.dreams_per_iter / complexity).max(1);
        let mut remaining = cfg.dreams_per_iter;
        for c in 1..=complexity {
            let take = if c == complexity { remaining } else { per_level.min(remaining) };
            if take == 0 { continue; }
            let chunk = sample_complexity_dreams(lib, rng, c, take, cfg);
            remaining = remaining.saturating_sub(chunk.len());
            all.extend(chunk);
        }
        all
    } else {
        sample_complexity_dreams(lib, rng, complexity, cfg.dreams_per_iter, cfg)
    };
    if dreams.is_empty() { return (0.0, 0.0); }

    let mut arenas: Vec<Arena> = Vec::new();
    let mut samples: Vec<TrainSample> = Vec::new();
    let mut arena_idx: Vec<usize> = Vec::new();

    for (mut arena, dream) in dreams {
        let s = dream_to_samples(
            &mut arena, lib, &dream, &cfg.literal_seeds, rng, cfg.max_negatives,
        );
        if s.samples.is_empty() { continue; }
        let aidx = arenas.len();
        arenas.push(arena);
        for sample in s.samples {
            arena_idx.push(aidx);
            samples.push(sample);
        }
    }

    if samples.is_empty() { return (0.0, 0.0); }

    let batch: Vec<(&TrainSample, &Arena, &Library)> = samples.iter().enumerate()
        .map(|(i, s)| (s, &arenas[arena_idx[i]], lib))
        .collect();

    match train_step(net, opt, &batch, cfg.fuel) {
        Ok(stats) => (stats.loss, stats.positive_top1),
        Err(e) => {
            eprintln!("train_step error: {e}");
            (0.0, 0.0)
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

/// Train a curriculum of increasing complexity. Returns one report row
/// per level.
pub fn train_complexity_curriculum(
    net: &Network,
    opt: &mut AdamW,
    lib: &Library,
    rng: &mut Rng,
    cfg: &ComplexityCfg,
    mut on_iter: impl FnMut(usize, usize, f32, f32),
) -> ComplexityReport {
    let mut report = ComplexityReport::default();
    let levels = cfg.levels.clone();
    for level in levels {
        let mut loss_first = f32::NAN;
        let mut loss_last = f32::NAN;
        let mut top1_first = f32::NAN;
        let mut top1_last = f32::NAN;
        for it in 0..cfg.iters_per_level {
            let (loss, top1) = train_one_level_iter(net, opt, lib, rng, level, cfg);
            on_iter(level, it, loss, top1);
            if it == 0 { loss_first = loss; top1_first = top1; }
            loss_last = loss; top1_last = top1;
        }
        let (eval_total, eval_solved, eval_avg_size, eval_avg_ms) =
            eval_complexity(net, lib, rng, level, cfg);
        report.rows.push(ComplexityRow {
            level,
            iters: cfg.iters_per_level,
            train_loss_first: loss_first,
            train_loss_last: loss_last,
            train_top1_first: top1_first,
            train_top1_last: top1_last,
            eval_total, eval_solved, eval_avg_size, eval_avg_ms,
        });
    }
    report
}
