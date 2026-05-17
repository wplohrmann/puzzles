//! Convert a dream task into a sequence of training samples.
//!
//! Bottom-up trajectory: the dream's DAG is topo-sorted; each non-seed
//! node is materialised in order, and at each step the policy target is
//! the App-pair that constructs the next node, while negatives come from
//! every other admissible pair already in the pool.
//!
//! See `decisions/05-curriculum.md` for the design.

use rustc_hash::FxHashMap;

use lang::arena::{Arena, NodeId};
use lang::builtin::ALL_BUILTINS;
use lang::construct::{lit, param, prim_ref};
use lang::eval::{apply, eval, Value};
use lang::ir::{LitValue, NodeKind};
use lang::library::Library;

use neural::{
    cand_features, state_features, task_features, PolicySample, ValueSample,
    CAND_FEAT_DIM,
};
use neural::Rng;

use crate::dream::DreamTask;

/// All training samples extracted from one dream.
#[derive(Default)]
pub struct DreamSamples {
    pub policy: Vec<PolicySample>,
    pub value: Vec<ValueSample>,
}

/// Cap on negatives per step. Each step gets the positive plus up to
/// this many negative candidates.
pub const MAX_NEGATIVES_PER_STEP: usize = 32;

/// Build training samples from a dream by replaying its bottom-up
/// construction order.
pub fn dream_to_samples(
    arena: &mut Arena,
    lib: &Library,
    cfg_fuel: u32,
    dream: &DreamTask,
    extra_literal_seeds: &[LitValue],
    rng: &mut Rng,
) -> DreamSamples {
    let mut out = DreamSamples::default();

    let inputs: Vec<Value> = dream.examples.iter().map(|(x, _)| x.clone()).collect();
    let expected: Vec<Value> = dream.examples.iter().map(|(_, y)| y.clone()).collect();
    let task_f = task_features(&expected);

    // Initial pool: param(0) + every primref + literal seeds.
    let mut pool_nodes: Vec<NodeId> = Vec::new();
    let mut pool_values: Vec<Vec<Value>> = Vec::new();
    let mut pool_cand_feats: Vec<[f32; CAND_FEAT_DIM]> = Vec::new();
    let mut id_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();

    let try_add_seed = |arena: &mut Arena,
                            id: NodeId,
                            pool_nodes: &mut Vec<NodeId>,
                            pool_values: &mut Vec<Vec<Value>>,
                            pool_cand_feats: &mut Vec<[f32; CAND_FEAT_DIM]>,
                            id_to_idx: &mut FxHashMap<NodeId, usize>| {
        if id_to_idx.contains_key(&id) { return; }
        let vals = eval_per_example(arena, lib, id, &inputs, cfg_fuel);
        let f = cand_features(&vals, &expected);
        id_to_idx.insert(id, pool_nodes.len());
        pool_nodes.push(id);
        pool_values.push(vals);
        pool_cand_feats.push(f);
    };

    let p0 = param(arena, 0);
    try_add_seed(arena, p0, &mut pool_nodes, &mut pool_values, &mut pool_cand_feats, &mut id_to_idx);
    for v in extra_literal_seeds {
        let id = lit(arena, v.clone());
        try_add_seed(arena, id, &mut pool_nodes, &mut pool_values, &mut pool_cand_feats, &mut id_to_idx);
    }
    for &b in ALL_BUILTINS {
        let pid = lib.lookup(b.name()).expect("builtin in seed lib");
        let id = prim_ref(arena, pid);
        try_add_seed(arena, id, &mut pool_nodes, &mut pool_values, &mut pool_cand_feats, &mut id_to_idx);
    }
    // Also seed any leaf nodes that appear in the dream but aren't in
    // the standard seed set (e.g. literal `2`).
    let topo = arena.reachable_topo(dream.program);
    for &node in &topo {
        match arena.kind(node) {
            NodeKind::Literal(_) | NodeKind::PrimRef(_) | NodeKind::Param { .. } => {
                // safe to add if missing
                if !id_to_idx.contains_key(&node) {
                    let vals = eval_per_example(arena, lib, node, &inputs, cfg_fuel);
                    let f = cand_features(&vals, &expected);
                    id_to_idx.insert(node, pool_nodes.len());
                    pool_nodes.push(node);
                    pool_values.push(vals);
                    pool_cand_feats.push(f);
                }
            }
            _ => {}
        }
    }

    // Walk the topological order; for every App node in the dream we
    // emit a training sample.
    for (step_idx, &node) in topo.iter().enumerate() {
        let _ = step_idx;
        match arena.kind(node).clone() {
            NodeKind::App { func, arg } => {
                if id_to_idx.contains_key(&node) { continue; }

                let f_idx = match id_to_idx.get(&func) {
                    Some(&i) => i,
                    // The dream's parent referenced a node not yet in
                    // the pool — should not happen given topo order.
                    None => continue,
                };
                let a_idx = match id_to_idx.get(&arg) {
                    Some(&i) => i,
                    None => continue,
                };

                let pool_size = pool_nodes.len();
                let state = state_features(&pool_cand_feats);

                // Build the candidate list. The positive is (f_idx, a_idx);
                // negatives are uniformly sampled (f', a') pairs that are
                // not the positive and don't already exist in the pool
                // (since pool entries can't be re-added).
                let mut candidates: Vec<[f32; CAND_FEAT_DIM]> = Vec::new();
                let pos_values = apply_per_example(
                    arena, lib, &pool_values[f_idx], &pool_values[a_idx], cfg_fuel,
                );
                let pos_cf = cand_features(&pos_values, &expected);
                candidates.push(pos_cf);
                let positive_idx = 0;

                let max_pairs = pool_size * pool_size;
                let mut tried = 0usize;
                let target_neg = MAX_NEGATIVES_PER_STEP.min(max_pairs.saturating_sub(1));
                // Hard-negative mixing: half share the positive's f_idx
                // (different a), a quarter share the positive's a_idx
                // (different f), the rest are fully random. Sharing one
                // half forces the network to discriminate on the
                // non-shared component instead of latching on to a
                // single coarse "right-vs-wrong shape" signal.
                while candidates.len() - 1 < target_neg && tried < target_neg * 12 {
                    tried += 1;
                    let mode = rng.gen_range(4);
                    let (g, h) = match mode {
                        0 | 1 => (f_idx, rng.gen_range(pool_size)),       // share f
                        2 => (rng.gen_range(pool_size), a_idx),            // share a
                        _ => (rng.gen_range(pool_size), rng.gen_range(pool_size)),
                    };
                    if g == f_idx && h == a_idx { continue; }
                    let neg_values = apply_per_example(
                        arena, lib, &pool_values[g], &pool_values[h], cfg_fuel,
                    );
                    // Skip negatives that Bottom on every example — they
                    // give the network a trivially-easy "non-Bottom good"
                    // signal that doesn't transfer to the search-time
                    // problem (the search already drops Bottoms).
                    if neg_values.iter().all(|v| v.is_bottom()) { continue; }
                    let neg_cf = cand_features(&neg_values, &expected);
                    candidates.push(neg_cf);
                }

                out.policy.push(PolicySample {
                    task_feat: task_f,
                    state_feat: state,
                    candidates,
                    positive_idx,
                });
                // Value head: every prefix on the dream's winning path
                // is "going to solve" → 1.0. To break the degenerate
                // "always predict 1" fit, also emit a negative built by
                // hypothetically extending the pool with a random wrong
                // App. This shifts the aggregated state features by the
                // wrong node's cand_features, giving the value head
                // something to discriminate against.
                out.value.push(ValueSample {
                    task_feat: task_f,
                    state_feat: state,
                    target: 1.0,
                });
                if pool_size >= 2 {
                    let g = rng.gen_range(pool_size);
                    let h = rng.gen_range(pool_size);
                    if !(g == f_idx && h == a_idx) {
                        let neg_v = apply_per_example(
                            arena, lib, &pool_values[g], &pool_values[h], cfg_fuel,
                        );
                        let neg_cf = cand_features(&neg_v, &expected);
                        // Build a hypothetical state that has the wrong
                        // node appended. We don't actually mutate the
                        // pool; we just compute state_features on the
                        // augmented list.
                        let mut augmented = pool_cand_feats.clone();
                        augmented.push(neg_cf);
                        let neg_state = state_features(&augmented);
                        out.value.push(ValueSample {
                            task_feat: task_f,
                            state_feat: neg_state,
                            target: 0.0,
                        });
                    }
                }

                // Add the new node to the pool.
                let new_idx = pool_nodes.len();
                let _ = new_idx;
                id_to_idx.insert(node, pool_nodes.len());
                pool_nodes.push(node);
                pool_values.push(pos_values);
                pool_cand_feats.push(pos_cf);
            }
            // Lambda nodes: the dream samples don't produce them (sampler
            // doesn't emit Lambdas). Skip just in case.
            NodeKind::Lambda { .. } => continue,
            _ => continue,
        }
    }

    out
}

fn eval_per_example(arena: &Arena, lib: &Library, node: NodeId, inputs: &[Value], fuel: u32) -> Vec<Value> {
    inputs.iter().map(|input| {
        let env = [input.clone()];
        let mut f = fuel;
        match eval(arena, lib, node, &env, &mut f) {
            Ok(v) => v,
            Err(_) => Value::bottom("dream eval err"),
        }
    }).collect()
}

fn apply_per_example(
    arena: &Arena, lib: &Library,
    f_vals: &[Value], a_vals: &[Value], fuel: u32,
) -> Vec<Value> {
    f_vals.iter().zip(a_vals.iter()).map(|(fv, av)| {
        let mut f = fuel;
        match apply(arena, lib, fv.clone(), av.clone(), &mut f) {
            Ok(v) => v,
            Err(_) => Value::bottom("dream apply err"),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dream::{sample_dream, DreamCfg};
    use lang::builtin::seed_builtin_library;

    #[test]
    fn samples_emerge_from_dream() {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let mut rng = Rng::new(7);
        let cfg = DreamCfg { max_size: 5, ..DreamCfg::default() };
        let dream = loop {
            if let Some(d) = sample_dream(&mut arena, &lib, &mut rng, &cfg) {
                break d;
            }
        };
        let s = dream_to_samples(
            &mut arena, &lib, cfg.fuel, &dream,
            &[LitValue::Int(0), LitValue::Int(1)],
            &mut rng,
        );
        assert!(!s.policy.is_empty(), "no policy samples emitted");
        for sample in &s.policy {
            assert!(sample.candidates.len() >= 1);
            assert!(sample.positive_idx < sample.candidates.len());
        }
    }
}
