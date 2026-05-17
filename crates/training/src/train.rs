//! Training loop: dream-only (no wake phase) until M5 lands.
//!
//! The loop iterates: at each step, sample a batch of dreams under the
//! current curriculum stage, convert them to (policy, value) samples,
//! and run one Adam update. Periodically evaluate the network on the
//! search-benchmark.

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::ir::LitValue;
use lang::library::Library;

use neural::{train_step, Network, NetworkCfg, Rng};
use neural::{PolicySample, ValueSample};

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
    /// Apply the optimizer this many times per iteration. Each step
    /// shuffles the batch and trains on a slice of size
    /// `ceil(batch_size / steps_per_iter)`.
    pub steps_per_iter: usize,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            iterations: 30,
            dreams_per_iter: 32,
            seed: 0xc0ffee_face,
            fuel: 200_000,
            examples_per_dream: 4,
            literal_seeds: vec![LitValue::Int(0), LitValue::Int(1)],
            curriculum: Curriculum::default(),
            network: NetworkCfg::default(),
            steps_per_iter: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IterMetrics {
    pub iter: usize,
    pub dreams_used: usize,
    pub policy_samples: usize,
    pub value_samples: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub max_size: u32,
}

/// Run one training iteration: sample dreams, build samples, train. The
/// caller owns the network and seed library.
pub fn train_one_iter(
    iter: usize,
    cfg: &TrainCfg,
    rng: &mut Rng,
    net: &mut Network,
    lib: &Library,
) -> IterMetrics {
    let max_size = cfg.curriculum.max_size_for(iter);
    let dream_cfg = DreamCfg {
        max_size,
        examples_per_dream: cfg.examples_per_dream,
        fuel: cfg.fuel,
        ..DreamCfg::default()
    };

    let mut policy_samples = Vec::new();
    let mut value_samples = Vec::new();
    let mut dreams_used = 0;

    // Use a fresh arena per dream so we don't bloat memory across the
    // training run; the arena is throwaway scratch for this dream's
    // bottom-up replay.
    for _ in 0..cfg.dreams_per_iter {
        let mut arena = Arena::new();
        let dream = match sample_dream(&mut arena, lib, rng, &dream_cfg) {
            Some(d) => d,
            None => continue,
        };
        let s = dream_to_samples(&mut arena, lib, cfg.fuel, &dream, &cfg.literal_seeds, rng);
        if s.policy.is_empty() { continue; }
        dreams_used += 1;
        policy_samples.extend(s.policy);
        value_samples.extend(s.value);
    }

    if policy_samples.is_empty() {
        return IterMetrics {
            iter, dreams_used: 0,
            policy_samples: 0, value_samples: 0,
            policy_loss: 0.0, value_loss: 0.0,
            max_size,
        };
    }

    // Shuffle the batch then split into `steps_per_iter` mini-batches and
    // do one Adam step per mini-batch.
    let mut p_idx: Vec<usize> = (0..policy_samples.len()).collect();
    let mut v_idx: Vec<usize> = (0..value_samples.len()).collect();
    fisher_yates(&mut p_idx, rng);
    fisher_yates(&mut v_idx, rng);

    let steps = cfg.steps_per_iter.max(1);
    let mut p_loss_sum = 0.0;
    let mut v_loss_sum = 0.0;
    let mut p_count = 0;
    let mut v_count = 0;

    for s in 0..steps {
        let p_lo = (s * policy_samples.len()) / steps;
        let p_hi = ((s + 1) * policy_samples.len()) / steps;
        let v_lo = (s * value_samples.len()) / steps;
        let v_hi = ((s + 1) * value_samples.len()) / steps;

        let p_batch: Vec<PolicySample> = p_idx[p_lo..p_hi].iter()
            .map(|&i| clone_policy(&policy_samples[i]))
            .collect();
        let v_batch: Vec<ValueSample> = v_idx[v_lo..v_hi].iter()
            .map(|&i| clone_value(&value_samples[i]))
            .collect();
        if p_batch.is_empty() && v_batch.is_empty() { continue; }
        let stats = train_step(net, &p_batch, &v_batch);
        p_loss_sum += stats.policy_loss * p_batch.len() as f32;
        v_loss_sum += stats.value_loss * v_batch.len() as f32;
        p_count += p_batch.len();
        v_count += v_batch.len();
    }

    IterMetrics {
        iter,
        dreams_used,
        policy_samples: policy_samples.len(),
        value_samples: value_samples.len(),
        policy_loss: if p_count > 0 { p_loss_sum / p_count as f32 } else { 0.0 },
        value_loss:  if v_count > 0 { v_loss_sum / v_count as f32 } else { 0.0 },
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

fn clone_policy(s: &PolicySample) -> PolicySample {
    PolicySample {
        task_feat: s.task_feat,
        state_feat: s.state_feat,
        candidates: s.candidates.clone(),
        positive_idx: s.positive_idx,
    }
}
fn clone_value(s: &ValueSample) -> ValueSample {
    ValueSample {
        task_feat: s.task_feat,
        state_feat: s.state_feat,
        target: s.target,
    }
}

/// Convenience: build a fresh seeded network and library to start
/// training from scratch.
pub fn fresh_network(cfg: &TrainCfg) -> (Network, Library, Rng) {
    let mut rng = Rng::new(cfg.seed);
    let net = Network::new(&cfg.network, &mut rng);
    let lib = seed_builtin_library();
    (net, lib, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_iter_runs() {
        let cfg = TrainCfg {
            iterations: 1,
            dreams_per_iter: 4,
            ..TrainCfg::default()
        };
        let (mut net, lib, mut rng) = fresh_network(&cfg);
        let m = train_one_iter(0, &cfg, &mut rng, &mut net, &lib);
        // Loose: should at least run without panicking; the network is
        // tiny so a single iter may have low diversity.
        assert!(m.max_size > 0);
    }
}
