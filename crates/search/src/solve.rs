//! The search loop. See `lib.rs` for the high-level shape.

use std::time::Instant;

use lang::arena::{Arena, NodeId};
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{apply, eval, Value};
use lang::library::Library;

use tasks::ListExamplesTask;

use crate::config::{SearchConfig, SearchResult, SearchStats};
use crate::pool::{probe_obs_key, value_obs_key, AddOutcome, Pool};

/// M2 solve. Bottom-up size-iterative enumeration with
/// observational-equivalence dedup. No static types, no neural
/// guidance.
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

    // Seeds: Param(0), each literal seed, each primitive ref.
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

    // Iterate by program size.
    for size in 2..=config.max_program_size {
        stats.max_size_explored = size;

        // Iterate splits with large `k_f` first — splits like
        // `(size-2, 1)` (which complete an `App(closure, Param)` chain)
        // tend to produce concrete-typed results that often match the
        // goal, while splits with small `k_f` extend compositions and
        // explode the pool faster.
        for k_f in (1..size).rev() {
            let k_a = size - 1 - k_f;
            if k_a == 0 || k_a >= size {
                continue;
            }
            let f_indices: Vec<usize> = pool.entries_with_size(k_f).to_vec();
            let a_indices: Vec<usize> = pool.entries_with_size(k_a).to_vec();
            for &i_f in &f_indices {
                // Runtime-based prefilter: skip f-entries whose values
                // contain no closure on any example. Applying a non-
                // function value as `f` produces `Bottom` 100% of the
                // time, and obs-eq would collapse them all into a single
                // Bottom entry — so iterating them just wastes work.
                let f_vals_ref = &pool.entry(i_f).values;
                if !f_vals_ref.iter().any(|v| matches!(v, Value::Closure(_))) {
                    continue;
                }
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
                    // Construction always succeeds — no static type
                    // check. If `f`'s value isn't a closure, applying
                    // produces Bottom and obs-eq collapses it.
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

                    let entry_size = pool.entry(i_f).size + pool.entry(i_a).size + 1;
                    if config.drop_all_bottom && values.iter().all(|v| v.is_bottom()) {
                        stats.apps_bottom_pruned += 1;
                        continue;
                    }

                    let obs_key = obs_key_for(arena, lib, &values, &inputs, config);
                    match pool.try_add(new_node, entry_size, values, obs_key) {
                        AddOutcome::Added => stats.apps_added += 1,
                        AddOutcome::DuplicateNode => stats.apps_dup_node += 1,
                        AddOutcome::ObsEqPruned => stats.apps_obs_eq_pruned += 1,
                    }
                }
            }
        }
    }

    stats.pool_by_size = pool.by_size.iter().map(|v| v.len()).collect();
    SearchResult::not_solved(started.elapsed(), pool.len(), stats)
}

// ------- seeding ---------------------------------------------------------

fn obs_key_for(
    arena: &Arena,
    lib: &Library,
    values: &[Value],
    inputs: &[Value],
    config: &SearchConfig,
) -> Option<u64> {
    value_obs_key(values).or_else(|| {
        if config.extended_obs_eq {
            probe_obs_key(arena, lib, values, inputs, config.eval_fuel)
        } else {
            None
        }
    })
}

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
    let obs_key = obs_key_for(arena, lib, &values, inputs, config);
    let _ = pool.try_add(p, 1, values, obs_key);
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
        let obs_key = obs_key_for(arena, lib, &values, inputs, config);
        let _ = pool.try_add(id, 1, values, obs_key);
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
        let id = prim_ref(arena, lib, p);
        let values = eval_per_example(arena, lib, id, inputs, config.eval_fuel, stats);
        let obs_key = obs_key_for(arena, lib, &values, inputs, config);
        let _ = pool.try_add(id, 1, values, obs_key);
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

/// BUSTLE-style incremental application. **Does not preserve the
/// lazy-`if` optimisation** in `eval`: if the new node happens to be a
/// fully-applied `if cond then else`, both `then` and `else` have
/// already been computed in the cached values, and a Bottom in the
/// unchosen branch will propagate through `apply`. For M2 trivial
/// tasks this doesn't matter.
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

#[cfg(test)]
mod tests {
    use super::*;
    use lang::builtin::seed_builtin_library;
    use tasks::TaskId;

    #[test]
    fn solves_identity_at_seeding() {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let task = ListExamplesTask {
            id: TaskId(1),
            examples: vec![
                (
                    Value::list_from(vec![Value::Int(1), Value::Int(2)]),
                    Value::list_from(vec![Value::Int(1), Value::Int(2)]),
                ),
                (
                    Value::list_from(vec![]),
                    Value::list_from(vec![]),
                ),
            ],
            fuel: 100_000,
        };
        let cfg = SearchConfig::default();
        let r = solve(&mut arena, &lib, &task, &cfg);
        assert!(r.solved, "identity not solved: {:?}", r);
        assert_eq!(r.size, 1);
    }
}
