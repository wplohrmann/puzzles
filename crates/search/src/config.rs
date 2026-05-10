use std::time::Duration;

use lang::arena::NodeId;
use lang::ir::LitValue;
use lang::ty::TyCon;

/// Tuning knobs for the search.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Inclusive upper bound on the size of any program the search will
    /// enumerate. Programs of larger size are not reached.
    pub max_program_size: u32,
    /// Hard cap on the number of pool entries. The search stops adding
    /// once the pool reaches this size.
    pub max_pool_size: usize,
    /// Fuel each per-example evaluation is given. Per-call, not cumulative.
    pub eval_fuel: u32,
    /// Hard wall-clock budget. The search returns `not solved` if this is
    /// exceeded.
    pub time_budget: Duration,
    /// Literal seeds added to the pool at size 1. Duplicates of literals
    /// the search would synthesise from primitives are not a problem
    /// (hash-cons makes them no-ops).
    pub literal_seeds: Vec<LitValue>,
    /// If true (default), restrict the pool to candidates whose type only
    /// uses type constructors that appear in the task's target type, plus
    /// `Fn` (always allowed). Polymorphic types (free vars only) are
    /// always allowed. This pre-empties whole search branches that go
    /// through unrelated types (e.g. `Bool`/`Pair` for an `Int`-returning
    /// numeric task) and is the single most effective non-NN heuristic
    /// for keeping the pool small at deeper sizes.
    ///
    /// Disable for tasks that legitimately need indirection through other
    /// types (e.g. a task that uses `if` even though its return type is
    /// `Int`).
    pub restrict_to_goal_tycons: bool,
    /// Additional forbidden tycons applied on top of (or instead of) the
    /// auto-derivation. Useful for unit tests / manual tuning.
    pub forbidden_tycons: Vec<TyCon>,
    /// If true (default), drop any non-function-typed candidate whose
    /// values on the examples are all `Bottom`. They can't be solutions,
    /// and using them as App arguments only propagates Bottom further.
    pub drop_all_bottom: bool,
    /// If true (default), apply BUSTLE-style extended obs-eq to function-
    /// typed candidates whose type matches `goal_arg_ty -> R` with `R`
    /// concrete: probe the closure with each example input and dedup on
    /// the resulting values.
    ///
    /// **Soundness caveat**: this prunes function-typed candidates whose
    /// behaviour at the goal level coincides on every example. In M2 every
    /// such closure is only ever applied directly to `Param(0)`, so the
    /// prune is sound. Tasks that pass these closures into other
    /// higher-order primitives could in principle lose valid solutions
    /// when two probe-equivalent closures diverge on intermediate values
    /// the search would otherwise have used. See
    /// `docs/decisions/m2-search-tasks.md`.
    pub extended_obs_eq: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_program_size: 16,
            max_pool_size: 2_000_000,
            eval_fuel: 200_000,
            time_budget: Duration::from_secs(10),
            literal_seeds: vec![
                LitValue::Int(0),
                LitValue::Int(1),
            ],
            restrict_to_goal_tycons: true,
            forbidden_tycons: Vec::new(),
            drop_all_bottom: true,
            extended_obs_eq: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub seeds: u64,
    pub apps_attempted: u64,
    pub apps_typecheck_failed: u64,
    pub apps_dup_node: u64,
    pub apps_obs_eq_pruned: u64,
    pub apps_tycon_pruned: u64,
    pub apps_bottom_pruned: u64,
    pub apps_added: u64,
    pub max_size_explored: u32,
    pub eval_errors: u64,
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
