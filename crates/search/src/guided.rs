//! Best-first guided search using a `neural::Network` as the prior.
//!
//! See `docs/03-search.md`. Priority is `q(f, a | task)` from the network.
//! The pool data structure is shared with the unguided variant; the
//! difference is the order in which `App(f, a)` candidates are popped.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

use lang::arena::{Arena, NodeId};
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{apply, eval, Value};
use lang::library::Library;

use neural::{scalar_f32, EmbedCache, Network};

use tasks::ListExamplesTask;

use crate::config::{SearchConfig, SearchResult, SearchStats};
use crate::pool::{AddOutcome, Pool};

#[derive(Clone, Debug)]
pub struct GuidedConfig {
    /// Subtract `size_penalty * size(c)` from the score so smaller
    /// candidates have an edge at equal `q`.
    pub size_penalty: f32,
    /// Hard cap on frontier length.
    pub max_frontier: usize,
    /// Pool-size cap; admissions stop past this but the frontier keeps
    /// draining (a solution may still pop).
    pub guided_pool_cap: usize,
    /// Drop candidates below this priority before enqueueing.
    pub priority_floor: f32,
}

impl Default for GuidedConfig {
    fn default() -> Self {
        Self {
            size_penalty: 0.0,
            max_frontier: 50_000,
            guided_pool_cap: 8_000,
            priority_floor: f32::NEG_INFINITY,
        }
    }
}

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
    let mut cache = EmbedCache::default();
    let target_rows = match net.target_stack(&expected) {
        Ok(t) => t,
        Err(_) => {
            // Network failed to embed targets; bail out cleanly.
            return SearchResult::not_solved(started.elapsed(), 0, stats);
        }
    };

    // Seed: param(0), literals, every PrimRef.
    {
        let p_node = param(arena, 0);
        let vals = eval_per_example(arena, lib, p_node, &inputs, cfg.eval_fuel, &mut stats);
        let _ = pool.try_add(p_node, 1, vals);
        if let Some(found) = pool_solution(&pool, &expected) {
            return finished_solve(found, started, &pool, stats);
        }
    }
    for v in &cfg.literal_seeds {
        let id = lit(arena, v.clone());
        let vals = eval_per_example(arena, lib, id, &inputs, cfg.eval_fuel, &mut stats);
        let _ = pool.try_add(id, 1, vals);
    }
    if let Some(found) = pool_solution(&pool, &expected) {
        return finished_solve(found, started, &pool, stats);
    }
    for p in 0..(lib.len() as u32) {
        let id = prim_ref(arena, p);
        let vals = eval_per_example(arena, lib, id, &inputs, cfg.eval_fuel, &mut stats);
        let _ = pool.try_add(id, 1, vals);
    }
    if let Some(found) = pool_solution(&pool, &expected) {
        return finished_solve(found, started, &pool, stats);
    }
    stats.seeds = pool.len() as u64;

    let mut frontier: BinaryHeap<Cand> = BinaryHeap::new();
    let initial_pool_size = pool.len();
    for i_f in 0..initial_pool_size {
        for i_a in 0..initial_pool_size {
            if let Some(found) = enqueue(
                arena, lib, &mut frontier, gcfg, &mut stats,
                &pool, i_f, i_a, &inputs, &expected,
                net, &target_rows, &mut cache, cfg.eval_fuel,
            ) {
                return finished_solve(found, started, &pool, stats);
            }
        }
    }

    while let Some(c) = frontier.pop() {
        if stats.apps_attempted & 4095 == 0 && started.elapsed() > cfg.time_budget {
            stats.pool_by_size = pool.by_size.iter().map(|v| v.len()).collect();
            return SearchResult::not_solved(started.elapsed(), pool.len(), stats);
        }
        stats.apps_attempted += 1;

        if pool.contains(c.node_id) {
            stats.apps_dup_node += 1;
            continue;
        }

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
        if c.size > cfg.max_program_size { continue; }
        if pool.len() >= gcfg.guided_pool_cap { continue; }

        let new_idx = pool.len();
        match pool.try_add(c.node_id, c.size, c.values) {
            AddOutcome::Added => {
                stats.apps_added += 1;
                if c.size > stats.max_size_explored { stats.max_size_explored = c.size; }
            }
            AddOutcome::DuplicateNode => {
                stats.apps_dup_node += 1;
                continue;
            }
        }

        let cur_pool_size = pool.len();
        for x in 0..cur_pool_size {
            if let Some(found) = enqueue(
                arena, lib, &mut frontier, gcfg, &mut stats,
                &pool, new_idx, x, &inputs, &expected,
                net, &target_rows, &mut cache, cfg.eval_fuel,
            ) {
                return finished_solve(found, started, &pool, stats);
            }
            if x != new_idx {
                if let Some(found) = enqueue(
                    arena, lib, &mut frontier, gcfg, &mut stats,
                    &pool, x, new_idx, &inputs, &expected,
                    net, &target_rows, &mut cache, cfg.eval_fuel,
                ) {
                    return finished_solve(found, started, &pool, stats);
                }
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
    inputs: &[Value],
    expected: &[Value],
    net: &Network,
    target_rows: &candle_core::Tensor,
    cache: &mut EmbedCache,
    fuel: u32,
) -> Option<SolutionFound> {
    let f_entry = pool.entry(f_idx);
    let a_entry = pool.entry(a_idx);
    let new_node = app(arena, f_entry.node, a_entry.node);
    if pool.contains(new_node) {
        stats.apps_dup_node += 1;
        return None;
    }
    let size = f_entry.size + a_entry.size + 1;
    let values = apply_values(arena, lib, &f_entry.values, &a_entry.values, fuel, stats);

    // Early-out: candidate already solves the task. The values check is
    // O(K) so this is essentially free per pair.
    if values_match(&values, expected) {
        return Some(SolutionFound { node: new_node, size });
    }

    if values.iter().all(|v| v.is_bottom()) {
        stats.apps_bottom_pruned += 1;
        return None;
    }

    let q_t = match net.q_score(
        f_entry.node, a_entry.node, arena, lib, inputs, target_rows, fuel, cache,
    ) {
        Ok(t) => t,
        Err(_) => return None,
    };
    let logit = match scalar_f32(&q_t) {
        Ok(v) => v,
        Err(_) => return None,
    };
    let prio = logit - gcfg.size_penalty * size as f32;

    if prio < gcfg.priority_floor { return None; }

    frontier.push(Cand { priority: prio, node_id: new_node, size, values });
    None
}

#[derive(Debug)]
struct Cand {
    priority: f32,
    node_id: NodeId,
    size: u32,
    values: Vec<Value>,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool { self.priority == other.priority }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
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
