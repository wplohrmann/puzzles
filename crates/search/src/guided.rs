//! Best-first guided search using a `neural::Network` as the policy
//! prior. See `decisions/04-best-first-frontier.md`.
//!
//! The pool data structure is shared with the size-iterative search; the
//! difference is the order in which `App(f, a)` candidates are
//! materialised. A priority queue keyed by the policy logit drives the
//! expansion. Hash-cons identity dedup and `Bottom`-prune are applied
//! identically.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

use lang::arena::{Arena, NodeId};
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{apply, eval, Value};
use lang::library::Library;

use neural::{
    cand_features, state_features, task_features, Network,
    CAND_FEAT_DIM, STATE_DIM, TASK_FEAT_DIM,
};

use tasks::ListExamplesTask;

use crate::config::{SearchConfig, SearchResult, SearchStats};
use crate::pool::{AddOutcome, Pool};

/// Hyperparameters specific to the guided variant. Kept separate from
/// `SearchConfig` so existing code is undisturbed.
#[derive(Clone, Debug)]
pub struct GuidedConfig {
    /// `priority(c) = policy_logit - size_penalty * size(c)`. A positive
    /// `size_penalty` (the default) prefers smaller programs even at
    /// equal logit — useful since the unguided baseline has a strong
    /// "smaller is better" prior.
    pub size_penalty: f32,
    /// Hard cap on the number of candidates kept on the priority queue.
    /// When exceeded, the worst entries are dropped.
    pub max_frontier: usize,
    /// How often (in pool-add events) to recompute state features.
    /// Recomputing every step is correct but quadratic; this lets us
    /// cheaply approximate the state by reusing a slightly-stale summary
    /// across small batches of additions.
    pub state_refresh_every: usize,
    /// Cap on pool size specifically for guided search. Independent from
    /// `SearchConfig::max_pool_size` because guided pools should stay
    /// orders-of-magnitude smaller than the un-guided exhaustive pool.
    /// Hitting this cap stops further admissions but keeps draining the
    /// frontier — the search may still solve from already-pooled entries.
    pub guided_pool_cap: usize,
    /// Drop candidates whose policy-priority is below this threshold
    /// before enqueueing. Set to `f32::NEG_INFINITY` to disable. The
    /// purpose is to keep the priority queue (and per-step work) bounded.
    pub priority_floor: f32,
}

impl Default for GuidedConfig {
    fn default() -> Self {
        Self {
            size_penalty: 0.05,
            max_frontier: 50_000,
            state_refresh_every: 8,
            guided_pool_cap: 8_000,
            priority_floor: f32::NEG_INFINITY,
        }
    }
}

/// Best-first solve.
pub fn solve_guided(
    arena: &mut Arena,
    lib: &Library,
    task: &ListExamplesTask,
    cfg: &SearchConfig,
    net: &Network,
    gcfg: &GuidedConfig,
) -> SearchResult {
    let started = Instant::now();
    let mut stats = SearchStats::default();

    let inputs: Vec<Value> = task.examples.iter().map(|(x, _)| x.clone()).collect();
    let expected: Vec<Value> = task.examples.iter().map(|(_, y)| y.clone()).collect();

    let mut pool = Pool::new();
    let mut node_feats: Vec<[f32; CAND_FEAT_DIM]> = Vec::new();

    let task_f = task_features(&expected);

    // Seed the pool: param, literals, every PrimRef. After each batch,
    // a quick check that the seed already solves the task.
    {
        let p_node = param(arena, 0);
        let p_vals = eval_per_example(arena, lib, p_node, &inputs, cfg.eval_fuel, &mut stats);
        let p_f = cand_features(&p_vals, &expected);
        if let AddOutcome::Added = pool.try_add(p_node, 1, p_vals) {
            node_feats.push(p_f);
        }
        if let Some(found) = pool_solution(&pool, &expected) {
            return finished_solve(found, started, &pool, stats);
        }
    }
    for v in &cfg.literal_seeds {
        let id = lit(arena, v.clone());
        let vals = eval_per_example(arena, lib, id, &inputs, cfg.eval_fuel, &mut stats);
        let f = cand_features(&vals, &expected);
        if let AddOutcome::Added = pool.try_add(id, 1, vals) {
            node_feats.push(f);
        }
    }
    if let Some(found) = pool_solution(&pool, &expected) {
        return finished_solve(found, started, &pool, stats);
    }
    for p in 0..(lib.len() as u32) {
        let id = prim_ref(arena, p);
        let vals = eval_per_example(arena, lib, id, &inputs, cfg.eval_fuel, &mut stats);
        let f = cand_features(&vals, &expected);
        if let AddOutcome::Added = pool.try_add(id, 1, vals) {
            node_feats.push(f);
        }
    }
    if let Some(found) = pool_solution(&pool, &expected) {
        return finished_solve(found, started, &pool, stats);
    }
    stats.seeds = pool.len() as u64;

    // Initial frontier: every (f, a) of seeds.
    let mut frontier: BinaryHeap<Cand> = BinaryHeap::new();
    let mut state = state_features(&node_feats);
    let mut adds_since_state_refresh: usize = 0;
    let initial_pool_size = pool.len();
    for i_f in 0..initial_pool_size {
        for i_a in 0..initial_pool_size {
            enqueue(
                arena, lib,
                &mut frontier, gcfg, &mut stats,
                &pool, i_f, i_a,
                &state, &task_f, &expected, net, cfg.eval_fuel,
            );
        }
    }

    // Best-first loop.
    while let Some(c) = frontier.pop() {
        if stats.apps_attempted & 4095 == 0
            && started.elapsed() > cfg.time_budget
        {
            stats.pool_by_size = pool.by_size.iter().map(|v| v.len()).collect();
            return SearchResult::not_solved(started.elapsed(), pool.len(), stats);
        }
        stats.apps_attempted += 1;

        if pool.contains(c.node_id) {
            stats.apps_dup_node += 1;
            continue;
        }

        // Solution check at dequeue time. Both the values and node id
        // were finalised at enqueue.
        if values_match(&c.values, &expected) {
            stats.pool_by_size = pool.by_size.iter().map(|v| v.len()).collect();
            return SearchResult {
                program: Some(c.node_id),
                solved: true,
                size: c.size,
                elapsed: started.elapsed(),
                final_pool_size: pool.len(),
                stats,
            };
        }
        if c.values.iter().any(|v| v.is_bottom()) {
            stats.apps_bottom_pruned += 1;
            continue;
        }
        if c.size > cfg.max_program_size {
            continue;
        }
        // Hit the guided pool cap: keep popping (the queue may still
        // hold a solution candidate) but stop admitting new entries.
        if pool.len() >= gcfg.guided_pool_cap {
            // We already evaluated this candidate above; the
            // values_match early-out fired if it solved. Otherwise,
            // skip admission and move on.
            continue;
        }

        let new_idx = pool.len();
        let cand_f = cand_features(&c.values, &expected);
        match pool.try_add(c.node_id, c.size, c.values) {
            AddOutcome::Added => {
                node_feats.push(cand_f);
                stats.apps_added += 1;
                if c.size > stats.max_size_explored {
                    stats.max_size_explored = c.size;
                }
            }
            AddOutcome::DuplicateNode => {
                stats.apps_dup_node += 1;
                continue;
            }
        }

        adds_since_state_refresh += 1;
        if adds_since_state_refresh >= gcfg.state_refresh_every {
            state = state_features(&node_feats);
            adds_since_state_refresh = 0;
        }

        // Enqueue every (new_idx, x) and (x, new_idx) candidate.
        let cur_pool_size = pool.len();
        for x in 0..cur_pool_size {
            enqueue(
                arena, lib, &mut frontier, gcfg, &mut stats,
                &pool, new_idx, x,
                &state, &task_f, &expected, net, cfg.eval_fuel,
            );
            if x != new_idx {
                enqueue(
                    arena, lib, &mut frontier, gcfg, &mut stats,
                    &pool, x, new_idx,
                    &state, &task_f, &expected, net, cfg.eval_fuel,
                );
            }
        }

        if frontier.len() > gcfg.max_frontier {
            let cap = gcfg.max_frontier;
            let mut sorted: Vec<Cand> = frontier.drain().collect();
            sorted.sort_unstable_by(|a, b| {
                b.priority.partial_cmp(&a.priority).unwrap_or(Ordering::Equal)
            });
            sorted.truncate(cap);
            frontier = BinaryHeap::from(sorted);
        }
    }

    stats.pool_by_size = pool.by_size.iter().map(|v| v.len()).collect();
    SearchResult::not_solved(started.elapsed(), pool.len(), stats)
}

#[allow(clippy::too_many_arguments)]
fn enqueue(
    arena: &mut Arena,
    lib: &Library,
    frontier: &mut BinaryHeap<Cand>,
    gcfg: &GuidedConfig,
    stats: &mut SearchStats,
    pool: &Pool,
    f_idx: usize,
    a_idx: usize,
    state: &[f32; STATE_DIM],
    task_f: &[f32; TASK_FEAT_DIM],
    expected: &[Value],
    net: &Network,
    fuel: u32,
) {
    let f_entry = pool.entry(f_idx);
    let a_entry = pool.entry(a_idx);
    let new_node = app(arena, f_entry.node, a_entry.node);
    if pool.contains(new_node) {
        stats.apps_dup_node += 1;
        return;
    }
    let size = f_entry.size + a_entry.size + 1;
    let values = apply_values(arena, lib, &f_entry.values, &a_entry.values, fuel, stats);

    // Cheap pre-filter: if every example Bottoms, the candidate can't
    // contribute to a solution and we don't need to score it. (The
    // dequeue loop also drops these, but skipping the network call here
    // is a meaningful speedup at search-time.)
    if values.iter().all(|v| v.is_bottom()) {
        stats.apps_bottom_pruned += 1;
        return;
    }

    let cand_f = cand_features(&values, expected);
    let logit = net.policy_logit(&cand_f, task_f, state);
    let prio = logit - gcfg.size_penalty * size as f32;

    if prio < gcfg.priority_floor {
        return;
    }

    frontier.push(Cand {
        priority: prio,
        node_id: new_node,
        size,
        values,
    });
}

#[derive(Debug)]
struct Cand {
    priority: f32,
    node_id: NodeId,
    size: u32,
    values: Vec<Value>,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority pops first. Tie-break preferring smaller size,
        // then deterministic node id.
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.size.cmp(&self.size))
            .then_with(|| other.node_id.raw().cmp(&self.node_id.raw()))
    }
}

fn pool_solution(pool: &Pool, expected: &[Value]) -> Option<SolutionFound> {
    for entry in pool.entries.iter() {
        if values_match(&entry.values, expected) {
            return Some(SolutionFound { node: entry.node, size: entry.size });
        }
    }
    None
}

fn values_match(values: &[Value], expected: &[Value]) -> bool {
    if values.len() != expected.len() { return false; }
    if values.iter().any(|v| v.is_bottom()) { return false; }
    values == expected
}

struct SolutionFound { node: NodeId, size: u32 }

fn finished_solve(
    found: SolutionFound,
    started: Instant,
    pool: &Pool,
    stats: SearchStats,
) -> SearchResult {
    SearchResult {
        program: Some(found.node),
        solved: true,
        size: found.size,
        elapsed: started.elapsed(),
        final_pool_size: pool.len(),
        stats,
    }
}

fn eval_per_example(
    arena: &Arena,
    lib: &Library,
    node: NodeId,
    inputs: &[Value],
    fuel: u32,
    stats: &mut SearchStats,
) -> Vec<Value> {
    inputs.iter().map(|input| {
        let env = [input.clone()];
        let mut f = fuel;
        match eval(arena, lib, node, &env, &mut f) {
            Ok(v) => v,
            Err(_) => { stats.eval_errors += 1; Value::bottom("seed eval error") }
        }
    }).collect()
}

fn apply_values(
    arena: &Arena,
    lib: &Library,
    f_vals: &[Value],
    a_vals: &[Value],
    fuel: u32,
    stats: &mut SearchStats,
) -> Vec<Value> {
    f_vals.iter().zip(a_vals.iter()).map(|(fv, av)| {
        let mut f = fuel;
        match apply(arena, lib, fv.clone(), av.clone(), &mut f) {
            Ok(v) => v,
            Err(_) => { stats.eval_errors += 1; Value::bottom("apply error") }
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang::builtin::seed_builtin_library;
    use neural::{Network, NetworkCfg, Rng};
    use std::time::Duration;
    use tasks::TaskId;

    #[test]
    fn guided_finds_identity_at_seeding() {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let task = ListExamplesTask {
            id: TaskId(1),
            examples: vec![
                (
                    Value::list_from(vec![Value::Int(1), Value::Int(2)]),
                    Value::list_from(vec![Value::Int(1), Value::Int(2)]),
                ),
                (Value::list_from(vec![]), Value::list_from(vec![])),
            ],
            fuel: 100_000,
        };
        let cfg = SearchConfig {
            time_budget: Duration::from_secs(5),
            ..SearchConfig::default()
        };
        let mut rng = Rng::new(123);
        let net = Network::new(&NetworkCfg::default(), &mut rng);
        let r = solve_guided(&mut arena, &lib, &task, &cfg, &net, &GuidedConfig::default());
        assert!(r.solved);
        assert_eq!(r.size, 1);
    }
}
