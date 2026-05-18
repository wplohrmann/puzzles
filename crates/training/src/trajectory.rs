//! Convert a dream task into a sequence of training samples.
//!
//! Walks the dream's DAG bottom-up; at each `App` node, the positive
//! pair `(f_t, a_t)` is paired with `K` sampled negative pairs drawn
//! from `pool_t × pool_t`. See `docs/02-neural.md` § Training.

use rustc_hash::FxHashSet;

use lang::arena::{Arena, NodeId};
use lang::builtin::ALL_BUILTINS;
use lang::construct::{lit, param, prim_ref};
use lang::ir::{LitValue, NodeKind};
use lang::library::Library;

use neural::Rng;
use neural::TrainSample;

use crate::dream::DreamTask;

/// Cap on negatives per step. The positive is always included.
pub const MAX_NEGATIVES_PER_STEP: usize = 32;

/// All training samples extracted from one dream.
#[derive(Default)]
pub struct DreamSamples {
    pub samples: Vec<TrainSample>,
}

pub fn dream_to_samples(
    arena: &mut Arena,
    lib: &Library,
    dream: &DreamTask,
    extra_literal_seeds: &[LitValue],
    rng: &mut Rng,
    max_negatives: usize,
) -> DreamSamples {
    let inputs: Vec<_> = dream.examples.iter().map(|(x, _)| x.clone()).collect();
    let targets: Vec<_> = dream.examples.iter().map(|(_, y)| y.clone()).collect();

    let mut pool: Vec<NodeId> = Vec::new();
    let mut in_pool: FxHashSet<NodeId> = FxHashSet::default();

    let add = |id: NodeId, pool: &mut Vec<NodeId>, set: &mut FxHashSet<NodeId>| {
        if set.insert(id) { pool.push(id); }
    };

    let p0 = param(arena, 0);
    add(p0, &mut pool, &mut in_pool);
    for v in extra_literal_seeds {
        let id = lit(arena, v.clone());
        add(id, &mut pool, &mut in_pool);
    }
    for &b in ALL_BUILTINS {
        let pid = lib.lookup(b.name()).expect("builtin in seed lib");
        let id = prim_ref(arena, pid);
        add(id, &mut pool, &mut in_pool);
    }
    // Seed any extra leaf literals the dream uses.
    let topo = arena.reachable_topo(dream.program);
    for &node in &topo {
        match arena.kind(node) {
            NodeKind::Literal(_) | NodeKind::PrimRef(_) | NodeKind::Param { .. } => {
                add(node, &mut pool, &mut in_pool);
            }
            _ => {}
        }
    }

    let mut out = DreamSamples::default();

    for &node in &topo {
        match arena.kind(node).clone() {
            NodeKind::App { func, arg } => {
                if in_pool.contains(&node) { continue; }
                if !in_pool.contains(&func) || !in_pool.contains(&arg) { continue; }

                let pool_size = pool.len();
                let mut candidates: Vec<(NodeId, NodeId)> = Vec::new();
                candidates.push((func, arg));
                let positive_idx = 0;

                let target_neg = max_negatives.min(pool_size * pool_size - 1);
                let mut tried = 0usize;
                let max_attempts = target_neg * 16 + 16;
                let mut seen: FxHashSet<(NodeId, NodeId)> = FxHashSet::default();
                seen.insert((func, arg));
                while candidates.len() - 1 < target_neg && tried < max_attempts {
                    tried += 1;
                    let mode = rng.gen_range(4);
                    let (g_id, h_id) = match mode {
                        0 | 1 => (func, pool[rng.gen_range(pool_size)]),
                        2 => (pool[rng.gen_range(pool_size)], arg),
                        _ => (pool[rng.gen_range(pool_size)], pool[rng.gen_range(pool_size)]),
                    };
                    if !seen.insert((g_id, h_id)) { continue; }
                    candidates.push((g_id, h_id));
                }

                out.samples.push(TrainSample {
                    candidates,
                    positive_idx,
                    inputs: inputs.clone(),
                    targets: targets.clone(),
                });

                add(node, &mut pool, &mut in_pool);
            }
            NodeKind::Lambda { .. } => continue,
            _ => continue,
        }
    }

    out
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
            if let Some(d) = sample_dream(&mut arena, &lib, &mut rng, &cfg) { break d; }
        };
        let s = dream_to_samples(
            &mut arena, &lib, &dream,
            &[LitValue::Int(0), LitValue::Int(1)],
            &mut rng, 4,
        );
        assert!(!s.samples.is_empty(), "no samples emitted");
        for sample in &s.samples {
            assert!(sample.candidates.len() >= 1);
            assert!(sample.positive_idx < sample.candidates.len());
        }
    }
}
