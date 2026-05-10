use std::time::Duration;

use lang::arena::NodeId;
use lang::ir::LitValue;

/// Tuning knobs for the search.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Inclusive upper bound on the size of any program the search will
    /// enumerate.
    pub max_program_size: u32,
    /// Hard cap on the number of pool entries.
    pub max_pool_size: usize,
    /// Fuel each per-example evaluation is given.
    pub eval_fuel: u32,
    /// Hard wall-clock budget.
    pub time_budget: Duration,
    /// Literal seeds added to the pool at size 1.
    pub literal_seeds: Vec<LitValue>,
    /// If true (default), drop any candidate whose values on every
    /// example are `Bottom`. Bottom-only candidates can't be solutions
    /// and only propagate Bottom when used as App args.
    pub drop_all_bottom: bool,
    /// If true (default), apply BUSTLE-style probe-based obs-eq to
    /// closure-valued candidates: apply the closure to each example
    /// input and dedup on the resulting values. Skipped if the probe
    /// itself yields another closure (we can't dedup a value tuple of
    /// closures).
    ///
    /// Soundness caveat: prunes closures that behave identically when
    /// applied directly to each example input but might differ when
    /// applied to other inputs (e.g. inside `fold`). For M2 trivial
    /// tasks every solution applies its outer closure directly to
    /// `Param(0)`, so the prune is sound.
    pub extended_obs_eq: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_program_size: 16,
            max_pool_size: 20_000_000,
            eval_fuel: 200_000,
            time_budget: Duration::from_secs(10),
            literal_seeds: vec![
                LitValue::Int(0),
                LitValue::Int(1),
            ],
            drop_all_bottom: true,
            extended_obs_eq: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub seeds: u64,
    pub apps_attempted: u64,
    pub apps_dup_node: u64,
    pub apps_obs_eq_pruned: u64,
    pub apps_bottom_pruned: u64,
    pub apps_added: u64,
    pub max_size_explored: u32,
    pub eval_errors: u64,
    /// Pool size by program-size bucket (entries `[1..=max_size]`).
    pub pool_by_size: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub program: Option<NodeId>,
    pub solved: bool,
    pub size: u32,
    pub elapsed: Duration,
    pub final_pool_size: usize,
    pub stats: SearchStats,
}

impl SearchResult {
    pub fn not_solved(elapsed: Duration, final_pool_size: usize, stats: SearchStats) -> Self {
        Self {
            program: None,
            solved: false,
            size: 0,
            elapsed,
            final_pool_size,
            stats,
        }
    }
}
