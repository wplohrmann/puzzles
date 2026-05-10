//! The search loop. See `lib.rs` for the high-level shape.

use std::time::Instant;

use lang::arena::{Arena, NodeId};
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{apply, eval, Value};
use lang::library::Library;

use tasks::ListExamplesTask;

use crate::config::{SearchConfig, SearchResult, SearchStats};
use crate::pool::{AddOutcome, Pool};

/// M2 solve. Bottom-up size-iterative enumeration, no behavioural
/// pruning beyond hash-cons-canonical dedup. Speed comes from the M4
/// neural prior, not from hand-tuned heuristics.
pub fn solve(
    arena: &mut Arena,
    lib: &Library,
    task: &ListExamplesTask,
    config: &SearchConfig,
) -> SearchResult {
    let started = Instant::now();
    let mut stats = SearchStats::default();

    let inputs: Vec<Value> = task.examples.iter().map(|(x, _)| x.clone()).collect();
    let expected: Vec<Value> = task.examples.iter().map(|(_, y)| y.clone()).collect();

    let mut pool = Pool::new();

    seed_param(&mut pool, arena, lib, &inputs, config, &mut stats);
    if let Some(found) = check_pool_for_solution(&pool, &expected) {
        return finished(found, started, &pool, stats);
    }
    seed_literals(&mut pool, arena, lib, &inputs, config, &mut stats);
    if let Some(found) = check_pool_for_solution(&pool, &expected) {
        return finished(found, started, &pool, stats);
    }
    seed_primitives(&mut pool, arena, lib, &inputs, config, &mut stats);
    if let Some(found) = check_pool_for_solution(&pool, &expected) {
        return finished(found, started, &pool, stats);
    }

    for size in 2..=config.max_program_size {
        stats.max_size_explored = size;

        for k_f in 1..size {
            let k_a = size - 1 - k_f;
            if k_a == 0 || k_a >= size {
                continue;
            }
            let f_indices: Vec<usize> = pool.entries_with_size(k_f).to_vec();
            let a_indices: Vec<usize> = pool.entries_with_size(k_a).to_vec();
            for &i_f in &f_indices {
                for &i_a in &a_indices {
                    if stats.apps_attempted & 4095 == 0
                        && (started.elapsed() > config.time_budget
                            || pool.len() >= config.max_pool_size)
                    {
                        stats.pool_by_size = pool.by_size.iter().map(|v| v.len()).collect();
                        return SearchResult::not_solved(started.elapsed(), pool.len(), stats);
                    }
                    stats.apps_attempted += 1;

                    let f_node = pool.entry(i_f).node;
                    let a_node = pool.entry(i_a).node;
                    let new_node = app(arena, f_node, a_node);
                    if pool.contains(new_node) {
                        stats.apps_dup_node += 1;
                        continue;
                    }

                    let f_vals = pool.entry(i_f).values.clone();
                    let a_vals = pool.entry(i_a).values.clone();
                    let values = apply_values(arena, lib, &f_vals, &a_vals, config.eval_fuel, &mut stats);

                    if values_match(&values, &expected) {
                        let entry_size = pool.entry(i_f).size + pool.entry(i_a).size + 1;
                        return SearchResult {
                            program: Some(new_node),
                            solved: true,
                            size: entry_size,
                            elapsed: started.elapsed(),
                            final_pool_size: pool.len(),
                            stats,
                        };
                    }

                    // A candidate whose values contain `Bottom` on any
                    // example is incapable of being a solution: `apply`
                    // propagates `Bottom` strictly, so every downstream
                    // composition stays `Bottom` on that position, and
                    // `values_match` rejects any `Bottom`. Skipping the
                    // pool insertion avoids enumerating dead chains.
                    if values.iter().any(|v| v.is_bottom()) {
                        stats.apps_bottom_pruned += 1;
                        continue;
                    }

                    let entry_size = pool.entry(i_f).size + pool.entry(i_a).size + 1;
                    match pool.try_add(new_node, entry_size, values) {
                        AddOutcome::Added => stats.apps_added += 1,
                        AddOutcome::DuplicateNode => stats.apps_dup_node += 1,
                    }
                }
            }
        }
    }

    stats.pool_by_size = pool.by_size.iter().map(|v| v.len()).collect();
    SearchResult::not_solved(started.elapsed(), pool.len(), stats)
}

// ------- seeding ---------------------------------------------------------

fn seed_param(
    pool: &mut Pool,
    arena: &mut Arena,
    lib: &Library,
    inputs: &[Value],
    config: &SearchConfig,
    stats: &mut SearchStats,
) {
    let p = param(arena, 0);
    let values = eval_per_example(arena, lib, p, inputs, config.eval_fuel, stats);
    let _ = pool.try_add(p, 1, values);
    stats.seeds += 1;
}

fn seed_literals(
    pool: &mut Pool,
    arena: &mut Arena,
    lib: &Library,
    inputs: &[Value],
    config: &SearchConfig,
    stats: &mut SearchStats,
) {
    for v in &config.literal_seeds {
        let id = lit(arena, v.clone());
        let values = eval_per_example(arena, lib, id, inputs, config.eval_fuel, stats);
        let _ = pool.try_add(id, 1, values);
        stats.seeds += 1;
    }
}

fn seed_primitives(
    pool: &mut Pool,
    arena: &mut Arena,
    lib: &Library,
    inputs: &[Value],
    config: &SearchConfig,
    stats: &mut SearchStats,
) {
    for p in 0..(lib.len() as u32) {
        let id = prim_ref(arena, p);
        let values = eval_per_example(arena, lib, id, inputs, config.eval_fuel, stats);
        let _ = pool.try_add(id, 1, values);
        stats.seeds += 1;
    }
}

// ------- value computation ----------------------------------------------

fn eval_per_example(
    arena: &Arena,
    lib: &Library,
    node: NodeId,
    inputs: &[Value],
    fuel: u32,
    stats: &mut SearchStats,
) -> Vec<Value> {
    inputs
        .iter()
        .map(|input| {
            let env = [input.clone()];
            let mut f = fuel;
            match eval(arena, lib, node, &env, &mut f) {
                Ok(v) => v,
                Err(_) => {
                    stats.eval_errors += 1;
                    Value::bottom("seed eval error")
                }
            }
        })
        .collect()
}

/// Incremental application of `f` to `a` per example. Mirrors the
/// `Value` produced by `eval` on the canonical `App(f, a)` node, except
/// that the lazy-`if` short-circuit in `eval` is **not** preserved: a
/// candidate whose top-level happens to be `if cond x y` will see
/// `Bottom` propagate through `apply` from either branch. The
/// solution-validator path uses `eval`, so an `if`-shaped solution that
/// only Bottom-applies is still detected at the moment its node is
/// constructed via `values_match` against `expected` — but it won't
/// then live in the pool as a usable sub-result.
fn apply_values(
    arena: &Arena,
    lib: &Library,
    f_vals: &[Value],
    a_vals: &[Value],
    fuel: u32,
    stats: &mut SearchStats,
) -> Vec<Value> {
    f_vals
        .iter()
        .zip(a_vals.iter())
        .map(|(fv, av)| {
            let mut f = fuel;
            match apply(arena, lib, fv.clone(), av.clone(), &mut f) {
                Ok(v) => v,
                Err(_) => {
                    stats.eval_errors += 1;
                    Value::bottom("apply error")
                }
            }
        })
        .collect()
}

// ------- solution check --------------------------------------------------

fn check_pool_for_solution(pool: &Pool, expected: &[Value]) -> Option<SolutionFound> {
    for entry in pool.entries.iter() {
        if values_match(&entry.values, expected) {
            return Some(SolutionFound {
                node: entry.node,
                size: entry.size,
            });
        }
    }
    None
}

fn values_match(values: &[Value], expected: &[Value]) -> bool {
    if values.len() != expected.len() {
        return false;
    }
    if values.iter().any(|v| v.is_bottom()) {
        return false;
    }
    values == expected
}

struct SolutionFound {
    node: NodeId,
    size: u32,
}

fn finished(
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
