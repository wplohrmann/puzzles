//! Dream-driven training loop.
//!
//! For each iteration:
//! 1. Sample `dreams_per_iter` dreams of the current curriculum stage.
//! 2. Convert each dream into a list of training samples.
//! 3. Run mini-batch optimizer steps over the flattened batch.

use candle_core::Device;
use candle_nn::AdamW;

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::ir::LitValue;
use lang::library::Library;

use neural::{make_optimizer, train_step, Network, NetworkCfg, Rng, TrainSample};

use crate::curriculum::Curriculum;
use crate::dream::{sample_dream, DreamCfg};
use crate::trajectory::dream_to_samples;

#[derive(Clone, Debug)]
pub struct TrainCfg {
    pub iterations: usize,
    pub dreams_per_iter: usize,
    pub seed: u64,
    pub fuel: u32,
    pub examples_per_dream: usize,
    pub literal_seeds: Vec<LitValue>,
    pub curriculum: Curriculum,
    pub network: NetworkCfg,
    pub steps_per_iter: usize,
    /// Negatives sampled per step.
    pub max_negatives: usize,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            iterations: 30,
            dreams_per_iter: 16,
            seed: 0xc0ffee_face,
            fuel: 100_000,
            examples_per_dream: 3,
            literal_seeds: vec![LitValue::Int(0), LitValue::Int(1)],
            curriculum: Curriculum::default(),
            network: NetworkCfg::default(),
            steps_per_iter: 1,
            max_negatives: 16,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IterMetrics {
    pub iter: usize,
    pub dreams_used: usize,
    pub samples: usize,
    pub loss: f32,
    pub top1: f32,
    pub max_size: u32,
}

pub fn train_one_iter(
    iter: usize,
    cfg: &TrainCfg,
    rng: &mut Rng,
    net: &Network,
    opt: &mut AdamW,
    lib: &Library,
) -> IterMetrics {
    let max_size = cfg.curriculum.max_size_for(iter);
    let dream_cfg = DreamCfg {
        max_size,
        examples_per_dream: cfg.examples_per_dream,
        fuel: cfg.fuel,
        ..DreamCfg::default()
    };

    let mut owned_arenas: Vec<Arena> = Vec::new();
    let mut owned_samples: Vec<TrainSample> = Vec::new();
    let mut arena_idx: Vec<usize> = Vec::new();

    for _ in 0..cfg.dreams_per_iter {
        let mut arena = Arena::new();
        let dream = match sample_dream(&mut arena, lib, rng, &dream_cfg) {
            Some(d) => d,
            None => continue,
        };
        let s = dream_to_samples(
            &mut arena, lib, &dream, &cfg.literal_seeds, rng, cfg.max_negatives,
        );
        if s.samples.is_empty() { continue; }
        let aidx = owned_arenas.len();
        owned_arenas.push(arena);
        for sample in s.samples {
            arena_idx.push(aidx);
            owned_samples.push(sample);
        }
    }

    if owned_samples.is_empty() {
        return IterMetrics {
            iter, dreams_used: 0, samples: 0, loss: 0.0, top1: 0.0, max_size,
        };
    }

    let mut perm: Vec<usize> = (0..owned_samples.len()).collect();
    fisher_yates(&mut perm, rng);

    let steps = cfg.steps_per_iter.max(1);
    let mut loss_sum = 0.0;
    let mut top1_sum = 0.0;
    let mut step_count = 0;
    for s in 0..steps {
        let lo = (s * perm.len()) / steps;
        let hi = ((s + 1) * perm.len()) / steps;
        let batch: Vec<(&TrainSample, &Arena, &Library)> = perm[lo..hi].iter()
            .map(|&i| (&owned_samples[i], &owned_arenas[arena_idx[i]], lib))
            .collect();
        if batch.is_empty() { continue; }
        let stats = match train_step(net, opt, &batch, cfg.fuel) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("train_step error: {e}");
                continue;
            }
        };
        loss_sum += stats.loss * stats.samples as f32;
        top1_sum += stats.positive_top1 * stats.samples as f32;
        step_count += stats.samples;
    }

    IterMetrics {
        iter,
        dreams_used: owned_arenas.len(),
        samples: owned_samples.len(),
        loss: if step_count > 0 { loss_sum / step_count as f32 } else { 0.0 },
        top1: if step_count > 0 { top1_sum / step_count as f32 } else { 0.0 },
        max_size,
    }
}

fn fisher_yates(v: &mut [usize], rng: &mut Rng) {
    let n = v.len();
    if n < 2 { return; }
    for i in (1..n).rev() {
        let j = rng.gen_range(i + 1);
        v.swap(i, j);
    }
}

pub fn fresh_network(cfg: &TrainCfg) -> (Network, Library, Rng, AdamW) {
    let rng = Rng::new(cfg.seed);
    let lib = seed_builtin_library();
    let net = Network::new(&cfg.network, &lib, Device::Cpu)
        .expect("network construction");
    let opt = make_optimizer(&net, cfg.network.lr, cfg.network.weight_decay)
        .expect("optimizer construction");
    (net, lib, rng, opt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_iter_runs() {
        let cfg = TrainCfg {
            iterations: 1,
            dreams_per_iter: 2,
            ..TrainCfg::default()
        };
        let (net, lib, mut rng, mut opt) = fresh_network(&cfg);
        let m = train_one_iter(0, &cfg, &mut rng, &net, &mut opt, &lib);
        assert!(m.max_size > 0);
    }
}
