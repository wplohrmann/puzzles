//! The search loop. See `lib.rs` for the high-level shape.

use std::time::Instant;

use rustc_hash::FxHashSet;

use lang::arena::{Arena, NodeId};
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{apply, eval, Value};
use lang::ir::LitValue;
use lang::library::Library;
use lang::ty::{unify, Subst, Ty, TyCon, TyVarGen};

use tasks::ListExamplesTask;

use crate::config::{SearchConfig, SearchResult, SearchStats};
use crate::pool::{applied_obs_key, value_obs_key, AddOutcome, Pool};

/// M2 solve. Bottom-up size-iterative enumeration with
/// observational-equivalence dedup. No neural guidance.
///
/// `task` is currently restricted to `ListExamplesTask` because the value
/// cache requires concrete `(input, output)` examples; the trait itself
/// stays generic so future task families can implement their own scoring.
pub fn solve(
    arena: &mut Arena,
    lib: &Library,
    task: &ListExamplesTask,
    config: &SearchConfig,
) -> SearchResult {
    let started = Instant::now();
    let mut stats = SearchStats::default();
    let mut gen = TyVarGen::new();

    let (arg_ty, ret_ty) = task.arg_ret();
    let inputs: Vec<Value> = task.examples.iter().map(|(x, _)| x.clone()).collect();
    let expected: Vec<Value> = task.examples.iter().map(|(_, y)| y.clone()).collect();

    // Precompute the "forbidden tycon" set: tycons that may NOT appear in
    // any candidate's type. Always allows `Fn` and free type variables.
    let forbidden = compute_forbidden_tycons(&arg_ty, &ret_ty, config);

    let mut pool = Pool::new();

    // Seeds: Param(0), each literal seed, each primitive ref.
    // The Param seed isn't tycon-filtered — by construction its type IS
    // the goal arg type, which is always allowed.
    seed_param(&mut pool, arena, lib, &arg_ty, &inputs, config.eval_fuel, &mut stats);
    if let Some(found) = check_pool_for_solution(arena, &pool, &ret_ty, &expected, &mut gen) {
        return finished(found, started, &pool, stats);
    }
    seed_literals(&mut pool, arena, lib, &config.literal_seeds, &inputs, config.eval_fuel, &forbidden, &mut stats);
    if let Some(found) = check_pool_for_solution(arena, &pool, &ret_ty, &expected, &mut gen) {
        return finished(found, started, &pool, stats);
    }
    seed_primitives(&mut pool, arena, lib, &inputs, config.eval_fuel, &forbidden, &mut stats);
    if let Some(found) = check_pool_for_solution(arena, &pool, &ret_ty, &expected, &mut gen) {
        return finished(found, started, &pool, stats);
    }

    // Iterate by program size.
    for size in 2..=config.max_program_size {
        stats.max_size_explored = size;

        // We snapshot the candidate splits before iterating because the
        // pool grows during iteration (higher-size entries are added but
        // they're irrelevant to the current size).
        //
        // Iteration order: large `k_f` first, so splits like
        // `(size-2, 1)` (which complete an `App(closure, Param)` chain)
        // are tried before the deeply-nested `(1, size-2)` ones. With a
        // pool cap, this matters: the latter explode the pool with new
        // closure-typed candidates while the former tend to produce
        // concrete-typed results that often match the goal.
        for k_f in (1..size).rev() {
            let k_a = size - 1 - k_f;
            if k_a == 0 || k_a >= size {
                continue;
            }
            let f_indices: Vec<usize> = pool.entries_with_size(k_f).to_vec();
            let a_indices: Vec<usize> = pool.entries_with_size(k_a).to_vec();
            for &i_f in &f_indices {
                let f_node = pool.entry(i_f).node;
                // Pre-filter: only consider F that's function-typed (or
                // polymorphic — could specialise to a function).
                let f_arg_root = match arena.ty(f_node) {
                    Ty::Con(TyCon::Fn, fa) => Some(root_tag(&fa[0])),
                    Ty::Var(_) => Some(RootTag::Wildcard),
                    _ => None,
                };
                let Some(f_arg_root) = f_arg_root else { continue };
                for &i_a in &a_indices {
                    if stats.apps_attempted & 4095 == 0
                        && (started.elapsed() > config.time_budget
                            || pool.len() >= config.max_pool_size)
                    {
                        return SearchResult::not_solved(started.elapsed(), pool.len(), stats);
                    }
                    stats.apps_attempted += 1;
                    let a_node = pool.entry(i_a).node;
                    let a_root = root_tag(arena.ty(a_node));
                    if !roots_compatible(f_arg_root, a_root) {
                        stats.apps_typecheck_failed += 1;
                        continue;
                    }
                    let new_node = match app(arena, &mut gen, f_node, a_node) {
                        Ok(n) => n,
                        Err(_) => {
                            stats.apps_typecheck_failed += 1;
                            continue;
                        }
                    };
                    if pool.contains(new_node) {
                        stats.apps_dup_node += 1;
                        continue;
                    }
                    // Forbidden-tycon prune. Cheaper than computing values.
                    if !forbidden.is_empty() && type_is_forbidden(arena.ty(new_node), &forbidden) {
                        stats.apps_tycon_pruned += 1;
                        continue;
                    }
                    let f_vals = pool.entry(i_f).values.clone();
                    let a_vals = pool.entry(i_a).values.clone();
                    let values = apply_values(arena, lib, &f_vals, &a_vals, config.eval_fuel, &mut stats);
                    // Solution check before obs-eq dedup, so that a candidate
                    // identical-by-value to an existing pool entry but
                    // matching the goal is still recognised.
                    if matches_goal(arena, new_node, &ret_ty, &values, &expected, &mut gen) {
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
                    if config.drop_all_bottom
                        && !is_function_type(arena.ty(new_node))
                        && values.iter().all(|v| v.is_bottom())
                    {
                        stats.apps_bottom_pruned += 1;
                        continue;
                    }
                    let ty = arena.ty(new_node).clone();
                    let obs_key = value_obs_key(&ty, &values).or_else(|| {
                        if config.extended_obs_eq {
                            applied_obs_key(
                                arena, lib, &ty, &values,
                                &inputs, &arg_ty, config.eval_fuel,
                            )
                        } else {
                            None
                        }
                    });
                    match pool.try_add(new_node, entry_size, values, obs_key) {
                        AddOutcome::Added => stats.apps_added += 1,
                        AddOutcome::DuplicateNode => stats.apps_dup_node += 1,
                        AddOutcome::ObsEqPruned => stats.apps_obs_eq_pruned += 1,
                    }
                }
            }
        }
    }

    SearchResult::not_solved(started.elapsed(), pool.len(), stats)
}

// ------- seeding ---------------------------------------------------------

fn seed_param(
    pool: &mut Pool,
    arena: &mut Arena,
    lib: &Library,
    arg_ty: &Ty,
    inputs: &[Value],
    fuel: u32,
    stats: &mut SearchStats,
) {
    let p = param(arena, 0, arg_ty.clone());
    let values = eval_per_example(arena, lib, p, inputs, fuel, stats);
    let ty = arena.ty(p).clone();
    let obs_key = value_obs_key(&ty, &values);
    let _ = pool.try_add(p, 1, values, obs_key);
    stats.seeds += 1;
}

fn seed_literals(
    pool: &mut Pool,
    arena: &mut Arena,
    lib: &Library,
    literal_seeds: &[LitValue],
    inputs: &[Value],
    fuel: u32,
    forbidden: &FxHashSet<TyCon>,
    stats: &mut SearchStats,
) {
    for v in literal_seeds {
        let id = lit(arena, v.clone());
        if !forbidden.is_empty() && type_is_forbidden(arena.ty(id), forbidden) {
            stats.apps_tycon_pruned += 1;
            continue;
        }
        let values = eval_per_example(arena, lib, id, inputs, fuel, stats);
        let ty = arena.ty(id).clone();
        let obs_key = value_obs_key(&ty, &values);
        let _ = pool.try_add(id, 1, values, obs_key);
        stats.seeds += 1;
    }
}

fn seed_primitives(
    pool: &mut Pool,
    arena: &mut Arena,
    lib: &Library,
    inputs: &[Value],
    fuel: u32,
    forbidden: &FxHashSet<TyCon>,
    stats: &mut SearchStats,
) {
    for p in 0..(lib.len() as u32) {
        let id = prim_ref(arena, lib, p);
        if !forbidden.is_empty() && type_is_forbidden(arena.ty(id), forbidden) {
            stats.apps_tycon_pruned += 1;
            continue;
        }
        let values = eval_per_example(arena, lib, id, inputs, fuel, stats);
        let ty = arena.ty(id).clone();
        let obs_key = value_obs_key(&ty, &values);
        let _ = pool.try_add(id, 1, values, obs_key);
        stats.seeds += 1;
    }
}

// ------- value computation ----------------------------------------------

/// Evaluate `node` once per example. Used for seed entries (Param, Lit,
/// PrimRef) where there's no f/arg value pair to apply incrementally.
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

/// Apply `f_vals[i]` to `a_vals[i]` per example. This is BUSTLE-style
/// incremental application: it does **not** preserve the lazy-`if`
/// optimisation in `eval`. For M2 trivial tasks this is fine — none use
/// `if` — but the limitation is documented in the M2 decisions log.
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

/// Walk the pool checking for any entry whose type matches the goal
/// return type and whose values equal the expected outputs. Used after
/// seeding (when no size-2+ enumeration has happened yet).
fn check_pool_for_solution(
    arena: &Arena,
    pool: &Pool,
    ret_ty: &Ty,
    expected: &[Value],
    gen: &mut TyVarGen,
) -> Option<SolutionFound> {
    for (idx, entry) in pool.entries.iter().enumerate() {
        if matches_goal(arena, entry.node, ret_ty, &entry.values, expected, gen) {
            return Some(SolutionFound {
                node: entry.node,
                size: entry.size,
                _idx: idx,
            });
        }
    }
    None
}

fn matches_goal(
    arena: &Arena,
    node: NodeId,
    ret_ty: &Ty,
    values: &[Value],
    expected: &[Value],
    gen: &mut TyVarGen,
) -> bool {
    if values.len() != expected.len() {
        return false;
    }
    if values.iter().any(|v| v.is_bottom()) {
        return false;
    }
    if values != expected {
        return false;
    }
    // Type unifies (after instantiating polymorphic free vars in the
    // candidate's stored type with fresh vars).
    let candidate_ty = instantiate_free(arena.ty(node), gen);
    let mut subst = Subst::default();
    unify(&candidate_ty, ret_ty, &mut subst).is_ok()
}

/// Tycons used in any non-Fn position of `t`. `Fn` itself is not collected.
fn collect_tycons(t: &Ty, out: &mut FxHashSet<TyCon>) {
    match t {
        Ty::Var(_) => {}
        Ty::Con(TyCon::Fn, args) => {
            for a in args { collect_tycons(a, out); }
        }
        Ty::Con(c, args) => {
            out.insert(*c);
            for a in args { collect_tycons(a, out); }
        }
    }
}

fn compute_forbidden_tycons(arg_ty: &Ty, ret_ty: &Ty, config: &SearchConfig) -> FxHashSet<TyCon> {
    let mut allowed: FxHashSet<TyCon> = FxHashSet::default();
    if config.restrict_to_goal_tycons {
        collect_tycons(arg_ty, &mut allowed);
        collect_tycons(ret_ty, &mut allowed);
    }
    let all: [TyCon; 6] = [TyCon::Int, TyCon::Bool, TyCon::Float, TyCon::Char, TyCon::List, TyCon::Pair];
    let mut forbidden: FxHashSet<TyCon> = FxHashSet::default();
    if config.restrict_to_goal_tycons {
        for c in all {
            if !allowed.contains(&c) {
                forbidden.insert(c);
            }
        }
    }
    for c in &config.forbidden_tycons {
        forbidden.insert(*c);
    }
    forbidden
}

fn type_is_forbidden(t: &Ty, forbidden: &FxHashSet<TyCon>) -> bool {
    match t {
        Ty::Var(_) => false,
        Ty::Con(TyCon::Fn, args) => args.iter().any(|a| type_is_forbidden(a, forbidden)),
        Ty::Con(c, args) => {
            forbidden.contains(c) || args.iter().any(|a| type_is_forbidden(a, forbidden))
        }
    }
}

fn is_function_type(t: &Ty) -> bool {
    matches!(t, Ty::Con(TyCon::Fn, _))
}

/// Cheap "root constructor" tag for type compatibility filtering.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RootTag {
    Wildcard,
    Con(TyCon),
}

fn root_tag(t: &Ty) -> RootTag {
    match t {
        Ty::Var(_) => RootTag::Wildcard,
        Ty::Con(c, _) => RootTag::Con(*c),
    }
}

fn roots_compatible(a: RootTag, b: RootTag) -> bool {
    match (a, b) {
        (RootTag::Wildcard, _) | (_, RootTag::Wildcard) => true,
        (RootTag::Con(x), RootTag::Con(y)) => x == y,
    }
}

fn instantiate_free(t: &Ty, gen: &mut TyVarGen) -> Ty {
    let frees = t.free_vars();
    if frees.is_empty() {
        return t.clone();
    }
    let mut subst = Subst::default();
    for v in frees {
        subst.insert(v, Ty::Var(gen.fresh()));
    }
    subst.apply(t)
}

struct SolutionFound {
    node: NodeId,
    size: u32,
    _idx: usize,
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
    use lang::ty::TypeScheme;
    use tasks::TaskId;

    /// Trivial: identity on `List<Int>` is the seeded `Param(0)` itself.
    #[test]
    fn solves_identity_at_seeding() {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let task = ListExamplesTask {
            id: TaskId(1),
            target_type: TypeScheme::mono(Ty::func(
                Ty::list(Ty::int()),
                Ty::list(Ty::int()),
            )),
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
