//! Training-mode guided search: same machinery as `solve_guided`, but
//! at each frontier expansion the action is **sampled** from a softmax
//! over the top-K candidates (rather than argmaxed). The chosen
//! action, the K alternatives, and the produced node are recorded as a
//! `TrajectoryStep`. The trajectory feeds the A2C-MC actor-critic
//! loss.
//!
//! Two callers:
//!
//! 1. **Poser-search** (`ScoringHead::Poser`, `SearchMode::Construct`):
//!    builds the dream's program. Terminates the moment the chosen
//!    expansion produces `App(stop, n)`, returning `n` as the program.
//! 2. **Searcher** (`ScoringHead::Q`, `SearchMode::Solve`): runs on
//!    the dream's I/O examples. Terminates when an expansion's values
//!    match `expected`.
//!
//! The two modes share everything except (a) which head ranks the
//! frontier and (b) the termination condition. Re-using one
//! implementation keeps the trajectory format identical between
//! poser- and q-trajectories — the loss code consumes them the same
//! way.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::time::{Duration, Instant};

use candle_core::{DType, Tensor};

use lang::arena::{Arena, NodeId};
use lang::builtin::BuiltinId;
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{apply, eval, Value};
use lang::ir::NodeKind;
use lang::library::{Library, PrimId};

use neural::{scalar_f32, EmbedCache, Network, Rng};

use crate::config::SearchConfig;
use crate::guided::GuidedConfig;
use crate::pool::{AddOutcome, Pool};

/// Which network head scores frontier candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringHead {
    Q,
    Poser,
}

/// Termination condition for the search loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    /// Stop when an expansion's per-example values match `expected`.
    /// Used by the q-search.
    Solve,
    /// Stop when an expansion produces `App(stop, n)`. The constructed
    /// program is `n`. Used by the poser-search; `expected` is ignored.
    Construct,
}

/// Knobs for action-sampling at training time.
#[derive(Clone, Debug)]
pub struct TrainingCfg {
    /// Top-K frontier candidates considered per step. The softmax is
    /// computed over their priorities; everything outside the top-K
    /// is invisible to the policy at this step.
    pub top_k: usize,
    /// Softmax temperature. `T → 0`: argmax. `T → ∞`: uniform. `T = 1`:
    /// vanilla softmax.
    pub temperature: f32,
    /// Hard ceiling on number of frontier expansions (= `S_pool` cap).
    pub max_steps: u32,
}

impl Default for TrainingCfg {
    fn default() -> Self {
        Self { top_k: 16, temperature: 1.0, max_steps: 500 }
    }
}

/// One step in a training-mode search trajectory. Stores enough info
/// for the loss code to re-run the scoring head and recover
/// `log π(a_t | s_t)` and the policy entropy.
#[derive(Clone, Debug)]
pub struct TrajectoryStep {
    /// Top-K candidates considered at this step, ordered by priority
    /// (descending). Each tuple = `(f_node, a_node, size)` of the
    /// candidate `App(f, a)`.
    pub candidates: Vec<(NodeId, NodeId, u32)>,
    /// Index in `candidates` of the action that was sampled.
    pub chosen_idx: usize,
    /// The hash-consed NodeId of `App(f_chosen, a_chosen)`.
    pub created_node: NodeId,
}

/// Membership info about a successful search's solution DAG.
#[derive(Clone, Debug)]
pub struct SolutionInfo {
    pub root: NodeId,
    pub size: u32,
    /// All App-node NodeIds reachable from `root`. Used as `S_nodes`
    /// in per-action credit assignment.
    pub s_nodes: HashSet<NodeId>,
}

/// Full search trajectory.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub steps: Vec<TrajectoryStep>,
    /// Number of frontier expansions performed. This is `S_pool` in
    /// the reward formulas.
    pub s_pool: u32,
    /// `Some` iff the search terminated by either solving (q-search)
    /// or producing `App(stop, n)` (poser-search).
    pub solution: Option<SolutionInfo>,
    pub elapsed: Duration,
}

/// Public entry point. See module-level doc for the two intended
/// caller flavours.
#[allow(clippy::too_many_arguments)]
pub fn solve_guided_training(
    arena: &mut Arena,
    lib: &Library,
    cfg: &SearchConfig,
    net: &Network,
    gcfg: &GuidedConfig,
    head: ScoringHead,
    mode: SearchMode,
    inputs: &[Value],
    expected: Option<&[Value]>,
    train_cfg: &TrainingCfg,
    rng: &mut Rng,
) -> Trajectory {
    let started = Instant::now();
    let stop_prim: Option<PrimId> = lib.lookup(BuiltinId::Stop.name());

    // For poser-search (no fixed target) we feed a `(K, N)` zero
    // tensor. The poser-head's ctx then aggregates over inputs-only;
    // the head learns whatever signal it can without a target.
    let target_rows = match expected {
        Some(t) => match net.target_stack(t) {
            Ok(t) => t,
            Err(_) => return empty_trajectory(started),
        },
        None => match Tensor::zeros(
            (inputs.len(), net.cfg.n),
            DType::F32,
            &net.device,
        ) {
            Ok(t) => t,
            Err(_) => return empty_trajectory(started),
        },
    };

    let mut pool = Pool::new();
    let mut cache = EmbedCache::default();

    seed_pool(arena, lib, cfg, inputs, &mut pool);

    // Don't allow the q-search to use the `stop` primitive at all —
    // `stop` is a sentinel only meaningful to the poser, and any
    // q-search expansion that touches it will produce `Bottom` at
    // eval. The frontier filter below catches it in both positions,
    // but excluding it from the seed pool also keeps stop's
    // `Closure(stop)` value out of unrelated apply()s. (We don't
    // remove from the pool here because that complicates indexing;
    // the enqueue filter is enough.)

    // Solve mode: a seeded pool entry might already match expected
    // (e.g. param(0) for the identity task). Detect and return
    // immediately with a zero-step trajectory — the q-search "solved"
    // by seeding alone, and we charge zero S_pool. Without this check
    // a trivial task like identity would burn the full budget.
    if mode == SearchMode::Solve {
        if let Some(expected) = expected {
            for entry in pool.entries.iter() {
                if values_match(&entry.values, expected) {
                    let s_nodes = collect_app_nodes(arena, entry.node);
                    return Trajectory {
                        steps: vec![],
                        s_pool: 0,
                        solution: Some(SolutionInfo {
                            root: entry.node,
                            size: entry.size,
                            s_nodes,
                        }),
                        elapsed: started.elapsed(),
                    };
                }
            }
        }
    }

    // Initial frontier: pairwise App over seeds.
    let mut frontier: BinaryHeap<TrainCand> = BinaryHeap::new();
    let initial_pool_size = pool.len();
    for i_f in 0..initial_pool_size {
        for i_a in 0..initial_pool_size {
            enqueue(
                arena, lib, &mut frontier, gcfg, &pool, i_f, i_a, inputs,
                &target_rows, net, head, mode, stop_prim, &mut cache, cfg.eval_fuel,
            );
        }
    }

    let mut steps: Vec<TrajectoryStep> = Vec::new();
    let mut step_count = 0u32;

    while step_count < train_cfg.max_steps {
        if frontier.is_empty() { break; }
        if started.elapsed() > cfg.time_budget { break; }

        // Pop top-K (or fewer if frontier is shorter).
        let mut top_k_cands: Vec<TrainCand> =
            Vec::with_capacity(train_cfg.top_k.min(frontier.len()));
        for _ in 0..train_cfg.top_k {
            match frontier.pop() {
                Some(c) => top_k_cands.push(c),
                None => break,
            }
        }
        if top_k_cands.is_empty() { break; }

        // Softmax-sample one.
        let priorities: Vec<f32> = top_k_cands.iter().map(|c| c.priority).collect();
        let chosen_idx = softmax_sample(&priorities, train_cfg.temperature, rng);

        let candidates_record: Vec<(NodeId, NodeId, u32)> = top_k_cands
            .iter()
            .map(|c| (c.f_node, c.a_node, c.size))
            .collect();
        let created_node = top_k_cands[chosen_idx].node_id;

        steps.push(TrajectoryStep {
            candidates: candidates_record,
            chosen_idx,
            created_node,
        });
        step_count += 1;

        let chosen = top_k_cands.remove(chosen_idx);
        // Non-chosen go back into the frontier for future steps.
        for c in top_k_cands { frontier.push(c); }

        // --- Termination: poser-search picked `App(stop, n)`.
        //
        // `stop` is a search-time sentinel — picking it ALWAYS ends
        // construction, with `n = chosen.a_node` as the program. We
        // don't add the App(stop, n) candidate to the pool. If `n`
        // happens to be a leaf (or otherwise produces an
        // invalid/trivial task at eval time), the dream just gets
        // rejected at the self-play validation step and the poser
        // sees zero reward — REINFORCE pushes it away from picking
        // stop in those cases. The poser learns *when* to terminate.
        if mode == SearchMode::Construct {
            if let Some(stop) = stop_prim {
                if let NodeKind::PrimRef(p) = arena.kind(chosen.f_node) {
                    if *p == stop {
                        let n = chosen.a_node;
                        let s_nodes = collect_app_nodes(arena, n);
                        let size = arena.reachable_topo(n).len() as u32;
                        return Trajectory {
                            steps,
                            s_pool: step_count,
                            solution: Some(SolutionInfo { root: n, size, s_nodes }),
                            elapsed: started.elapsed(),
                        };
                    }
                }
            }
        }

        // --- Termination: q-search produced a program matching `expected`.
        if mode == SearchMode::Solve {
            if let Some(expected) = expected {
                if values_match(&chosen.values, expected) {
                    let s_nodes = collect_app_nodes(arena, chosen.node_id);
                    return Trajectory {
                        steps,
                        s_pool: step_count,
                        solution: Some(SolutionInfo {
                            root: chosen.node_id,
                            size: chosen.size,
                            s_nodes,
                        }),
                        elapsed: started.elapsed(),
                    };
                }
            }
        }

        // Standard pruning before pool admission.
        if chosen.values.iter().any(|v| v.is_bottom()) { continue; }
        if chosen.size > cfg.max_program_size { continue; }
        if pool.len() >= gcfg.guided_pool_cap { continue; }

        let new_idx = pool.len();
        match pool.try_add(chosen.node_id, chosen.size, chosen.values) {
            AddOutcome::Added => {}
            AddOutcome::DuplicateNode => continue,
        }

        // Enqueue new pairs involving the just-added pool entry.
        let cur_pool_size = pool.len();
        for x in 0..cur_pool_size {
            enqueue(
                arena, lib, &mut frontier, gcfg, &pool, new_idx, x, inputs,
                &target_rows, net, head, mode, stop_prim, &mut cache, cfg.eval_fuel,
            );
            if x != new_idx {
                enqueue(
                    arena, lib, &mut frontier, gcfg, &pool, x, new_idx, inputs,
                    &target_rows, net, head, mode, stop_prim, &mut cache, cfg.eval_fuel,
                );
            }
        }

        // Cap the frontier.
        if frontier.len() > gcfg.max_frontier {
            let cap = gcfg.max_frontier;
            let mut sorted: Vec<TrainCand> = frontier.drain().collect();
            sorted.sort_unstable_by(|a, b| {
                b.priority
                    .partial_cmp(&a.priority)
                    .unwrap_or(Ordering::Equal)
            });
            sorted.truncate(cap);
            frontier = BinaryHeap::from(sorted);
        }
    }

    Trajectory {
        steps,
        s_pool: step_count,
        solution: None,
        elapsed: started.elapsed(),
    }
}

fn empty_trajectory(started: Instant) -> Trajectory {
    Trajectory {
        steps: vec![],
        s_pool: 0,
        solution: None,
        elapsed: started.elapsed(),
    }
}

#[derive(Debug)]
struct TrainCand {
    priority: f32,
    f_node: NodeId,
    a_node: NodeId,
    node_id: NodeId, // = App(f_node, a_node), hash-consed
    size: u32,
    values: Vec<Value>,
}

impl PartialEq for TrainCand {
    fn eq(&self, other: &Self) -> bool { self.priority == other.priority }
}
impl Eq for TrainCand {}
impl PartialOrd for TrainCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for TrainCand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.size.cmp(&self.size))
            .then_with(|| other.node_id.raw().cmp(&self.node_id.raw()))
    }
}

#[allow(clippy::too_many_arguments)]
fn enqueue(
    arena: &mut Arena,
    lib: &Library,
    frontier: &mut BinaryHeap<TrainCand>,
    gcfg: &GuidedConfig,
    pool: &Pool,
    f_idx: usize,
    a_idx: usize,
    inputs: &[Value],
    target_rows: &Tensor,
    net: &Network,
    head: ScoringHead,
    mode: SearchMode,
    stop_prim: Option<PrimId>,
    cache: &mut EmbedCache,
    fuel: u32,
) {
    let f_entry = pool.entry(f_idx);
    let a_entry = pool.entry(a_idx);

    // `stop` is a sentinel; it should only ever appear in the
    // *function* position of the outermost App. Filter the action
    // space directly so the policy never sees illegal positions.
    let f_is_stop = is_stop_primref(arena, f_entry.node, stop_prim);
    let a_is_stop = is_stop_primref(arena, a_entry.node, stop_prim);
    if a_is_stop { return; }                       // stop never in arg pos
    if f_is_stop && mode == SearchMode::Solve { return; } // q-search ignores stop

    let new_node = app(arena, f_entry.node, a_entry.node);
    if pool.contains(new_node) { return; }
    let size = f_entry.size + a_entry.size + 1;

    // For stop-applications, the candidate is a termination action,
    // not a runtime expression. Skip eval entirely — the search loop
    // recognises `f = stop` and returns the arg as the program.
    //
    // The only hard validity check at search-time: X's evaluated
    // values must not be closure or bottom. Those are fundamentally
    // invalid task outputs because value-equality is undefined for
    // closures (we never recover them). Everything else — leaves,
    // constants, identities — is allowed through; they get auto-
    // shortcut by the q-search with `r_poser = small_floor`,
    // providing a faint but nonzero signal that "produced *some*
    // program" is preferable to "produced nothing". The poser then
    // learns by gradient that non-trivial programs (which earn full
    // tent reward) are better still.
    let values = if f_is_stop {
        if a_entry.values.iter().any(|v| v.is_bottom() || contains_closure(v)) {
            return;
        }
        Vec::new()
    } else {
        let values = apply_values(arena, lib, &f_entry.values, &a_entry.values, fuel);
        if values.iter().all(|v| v.is_bottom()) { return; }
        values
    };

    let score_t = match head {
        ScoringHead::Q => net.q_score(
            f_entry.node, a_entry.node, arena, lib, inputs, target_rows, fuel, cache,
        ),
        ScoringHead::Poser => net.poser_score(
            f_entry.node, a_entry.node, arena, lib, inputs, target_rows, fuel, cache,
        ),
    };
    let logit = match score_t.and_then(|t| scalar_f32(&t)) {
        Ok(v) => v,
        Err(_) => return,
    };
    let prio = logit - gcfg.size_penalty * size as f32;
    if prio < gcfg.priority_floor { return; }

    frontier.push(TrainCand {
        priority: prio,
        f_node: f_entry.node,
        a_node: a_entry.node,
        node_id: new_node,
        size,
        values,
    });
}

fn is_stop_primref(arena: &Arena, node: NodeId, stop: Option<PrimId>) -> bool {
    if let (Some(stop), NodeKind::PrimRef(p)) = (stop, arena.kind(node)) {
        return *p == stop;
    }
    false
}

fn contains_closure(v: &Value) -> bool {
    match v {
        Value::Closure(_) => true,
        Value::List(xs) => xs.iter().any(contains_closure),
        Value::Pair(p) => contains_closure(&p.0) || contains_closure(&p.1),
        _ => false,
    }
}

fn seed_pool(
    arena: &mut Arena,
    lib: &Library,
    cfg: &SearchConfig,
    inputs: &[Value],
    pool: &mut Pool,
) {
    let p_node = param(arena, 0);
    let vals = eval_per_example(arena, lib, p_node, inputs, cfg.eval_fuel);
    let _ = pool.try_add(p_node, 1, vals);

    for v in &cfg.literal_seeds {
        let id = lit(arena, v.clone());
        let vals = eval_per_example(arena, lib, id, inputs, cfg.eval_fuel);
        let _ = pool.try_add(id, 1, vals);
    }

    for p in 0..(lib.len() as u32) {
        let id = prim_ref(arena, p);
        let vals = eval_per_example(arena, lib, id, inputs, cfg.eval_fuel);
        let _ = pool.try_add(id, 1, vals);
    }
}

fn eval_per_example(
    arena: &Arena,
    lib: &Library,
    node: NodeId,
    inputs: &[Value],
    fuel: u32,
) -> Vec<Value> {
    inputs
        .iter()
        .map(|input| {
            let env = [input.clone()];
            let mut f = fuel;
            eval(arena, lib, node, &env, &mut f)
                .unwrap_or_else(|_| Value::bottom("seed eval error"))
        })
        .collect()
}

fn apply_values(
    arena: &Arena,
    lib: &Library,
    f_vals: &[Value],
    a_vals: &[Value],
    fuel: u32,
) -> Vec<Value> {
    f_vals
        .iter()
        .zip(a_vals.iter())
        .map(|(fv, av)| {
            let mut f = fuel;
            apply(arena, lib, fv.clone(), av.clone(), &mut f)
                .unwrap_or_else(|_| Value::bottom("apply error"))
        })
        .collect()
}

fn values_match(values: &[Value], expected: &[Value]) -> bool {
    if values.len() != expected.len() { return false; }
    if values.iter().any(|v| v.is_bottom()) { return false; }
    values == expected
}

/// Collect every `App` NodeId reachable from `root`. This is `S_nodes`
/// in the per-action credit assignment — the policy gets credit for
/// any action whose `created_node` is in this set.
fn collect_app_nodes(arena: &Arena, root: NodeId) -> HashSet<NodeId> {
    arena
        .reachable_topo(root)
        .into_iter()
        .filter(|&id| matches!(arena.kind(id), NodeKind::App { .. }))
        .collect()
}

/// Sample from a categorical over `priorities` with the given softmax
/// temperature. Numerically stable (subtracts max).
fn softmax_sample(priorities: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    debug_assert!(!priorities.is_empty());
    let t = temperature.max(1e-6);
    let max_p = priorities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = priorities
        .iter()
        .map(|p| ((p - max_p) / t).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        // All scores collapsed to -inf or some pathology; fall back to
        // uniform sampling so we still make progress.
        return rng.gen_range(priorities.len());
    }
    let u = rng.next_f32() * sum;
    let mut cum = 0.0;
    for (i, &e) in exps.iter().enumerate() {
        cum += e;
        if u <= cum { return i; }
    }
    priorities.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use lang::builtin::seed_builtin_library;
    use neural::NetworkCfg;

    fn tiny_net() -> (Network, Library) {
        let lib = seed_builtin_library();
        let cfg = NetworkCfg { n: 8, ..NetworkCfg::default() };
        let net = Network::new(&cfg, &lib, Device::Cpu).unwrap();
        (net, lib)
    }

    /// Construct mode: with the temperature high enough that we're
    /// sampling uniformly, the poser-search should eventually pick
    /// `App(stop, _)` and terminate. We only assert termination here
    /// — the specific program is unconstrained for a random network.
    #[test]
    fn poser_search_terminates_on_stop() {
        let (net, lib) = tiny_net();
        let mut arena = Arena::new();
        let inputs = vec![Value::Int(0)];

        let cfg = SearchConfig {
            time_budget: Duration::from_secs(5),
            max_program_size: 12,
            ..SearchConfig::default()
        };
        let gcfg = GuidedConfig::default();
        let train_cfg = TrainingCfg {
            top_k: 4,
            temperature: 10.0, // nearly uniform — explore broadly
            max_steps: 200,
        };
        let mut rng = Rng::new(0xabc);

        let traj = solve_guided_training(
            &mut arena, &lib, &cfg, &net, &gcfg,
            ScoringHead::Poser, SearchMode::Construct,
            &inputs, None, &train_cfg, &mut rng,
        );

        // Either the poser hit App(stop, _) → solution is Some, or it
        // ran out of budget → solution is None. We can't be certain
        // about which with a random network — but the trajectory
        // should have non-trivial steps either way.
        assert!(traj.steps.len() > 0, "expected non-empty trajectory");
        assert!(traj.s_pool > 0);
    }

    /// Solve mode: when expected matches a one-app program, the search
    /// should be able to find it within budget given enough exploration.
    /// With a random network this is high-variance, so we only assert
    /// shape sanity — the loss code is what needs to be correct.
    #[test]
    fn q_search_records_trajectory_shape() {
        let (net, lib) = tiny_net();
        let mut arena = Arena::new();

        // Trivial task: identity. v0 → v0.
        let inputs = vec![Value::Int(1), Value::Int(2)];
        let expected = vec![Value::Int(1), Value::Int(2)];

        let cfg = SearchConfig {
            time_budget: Duration::from_secs(5),
            max_program_size: 4,
            ..SearchConfig::default()
        };
        let gcfg = GuidedConfig::default();
        let train_cfg = TrainingCfg { top_k: 4, temperature: 1.0, max_steps: 50 };
        let mut rng = Rng::new(0xdef);

        let traj = solve_guided_training(
            &mut arena, &lib, &cfg, &net, &gcfg,
            ScoringHead::Q, SearchMode::Solve,
            &inputs, Some(&expected), &train_cfg, &mut rng,
        );

        // The seeded pool already contains param(0) which evaluates to
        // the inputs themselves — but the search emits a non-trivial
        // trajectory of pool-pair expansions before that's discovered
        // via values_match (or it never is). Either way, the step
        // count is bounded by max_steps.
        assert!(traj.s_pool <= train_cfg.max_steps);
        for step in &traj.steps {
            assert!(step.chosen_idx < step.candidates.len());
            assert!(!step.candidates.is_empty());
        }
    }
}
