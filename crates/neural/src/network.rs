//! Network glue: per-candidate features → policy logit, state → value logit.
//!
//! See `decisions/03-value-only-features.md` for the architecture summary.
//! In short:
//!
//!   per_example_pair_feats(value, target) → AGG → cand_feat (3 * 24 = 72)
//!   per_example_solo_feats(target)        → AGG → task_feat (3 * 12 = 36)
//!   pool aggregate (mean+max+min of cand_feat over pool nodes) → state (3 * 72 = 216)
//!
//!   policy logit = PolicyMLP([cand_feat ⊕ task_feat ⊕ state]) — dim 324 → 1
//!   value scalar = ValueMLP([task_feat ⊕ state])              — dim 252 → 1

use serde::{Deserialize, Serialize};

use lang::eval::Value;

use crate::feat::{
    aggregate_examples, value_features, value_pair_features, PAIR_FEAT_DIM, SOLO_FEAT_DIM,
};
use crate::mlp::{adam_step, sigmoid_bce, softmax_xent, AdamCfg, Mlp};
use crate::rng::Rng;

/// Aggregated feature width per task example (mean + max + min over examples).
pub const CAND_FEAT_DIM: usize = 3 * PAIR_FEAT_DIM; // 72
pub const TASK_FEAT_DIM: usize = 3 * SOLO_FEAT_DIM; // 36
/// State summary: aggregated over pool nodes' candidate features.
pub const STATE_DIM: usize = 3 * CAND_FEAT_DIM;     // 216

pub const POLICY_IN_DIM: usize = CAND_FEAT_DIM + TASK_FEAT_DIM + STATE_DIM; // 324
pub const VALUE_IN_DIM: usize = TASK_FEAT_DIM + STATE_DIM;                  // 252

/// The trainable network. Two MLPs share no weights — they're tiny.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Network {
    pub policy: Mlp,
    pub value: Mlp,
    pub adam: AdamCfgWire,
    /// 1-indexed Adam step counter.
    pub step: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AdamCfgWire {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamCfgWire {
    fn default() -> Self {
        let c = AdamCfg::default();
        Self { lr: c.lr, beta1: c.beta1, beta2: c.beta2, eps: c.eps, weight_decay: c.weight_decay }
    }
}

impl AdamCfgWire {
    pub fn to_cfg(self) -> AdamCfg {
        AdamCfg {
            lr: self.lr, beta1: self.beta1, beta2: self.beta2,
            eps: self.eps, weight_decay: self.weight_decay,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NetworkCfg {
    pub policy_hidden: usize,
    pub value_hidden: usize,
    pub adam: AdamCfg,
}

impl Default for NetworkCfg {
    fn default() -> Self {
        Self {
            policy_hidden: 128,
            value_hidden: 64,
            adam: AdamCfg {
                lr: 3e-3,
                ..AdamCfg::default()
            },
        }
    }
}

impl Network {
    pub fn new(cfg: &NetworkCfg, rng: &mut Rng) -> Self {
        let policy = Mlp::new(&[POLICY_IN_DIM, cfg.policy_hidden, cfg.policy_hidden, 1], rng);
        let value = Mlp::new(&[VALUE_IN_DIM, cfg.value_hidden, 1], rng);
        Self {
            policy, value,
            adam: AdamCfgWire {
                lr: cfg.adam.lr, beta1: cfg.adam.beta1, beta2: cfg.adam.beta2,
                eps: cfg.adam.eps, weight_decay: cfg.adam.weight_decay,
            },
            step: 0,
        }
    }

    /// Reinitialise serde-skipped scratch buffers after loading.
    pub fn rehydrate_scratch(&mut self) {
        self.policy.rehydrate_scratch();
        self.value.rehydrate_scratch();
    }

    /// Score a single candidate. Returns the policy logit; the search uses
    /// raw logits (max wins; softmax only matters at training time).
    pub fn policy_logit(
        &self,
        cand: &[f32; CAND_FEAT_DIM],
        task: &[f32; TASK_FEAT_DIM],
        state: &[f32; STATE_DIM],
    ) -> f32 {
        let mut x = Vec::with_capacity(POLICY_IN_DIM);
        x.extend_from_slice(cand);
        x.extend_from_slice(task);
        x.extend_from_slice(state);
        let (out, _) = self.policy.forward(&x);
        out[0]
    }

    /// Estimate `V(state, task) ∈ (0, 1)` (sigmoid of logit).
    pub fn value_estimate(
        &self,
        task: &[f32; TASK_FEAT_DIM],
        state: &[f32; STATE_DIM],
    ) -> f32 {
        let mut x = Vec::with_capacity(VALUE_IN_DIM);
        x.extend_from_slice(task);
        x.extend_from_slice(state);
        let (out, _) = self.value.forward(&x);
        sigmoid(out[0])
    }
}

// -- aggregation helpers used by both training and inference -------------

/// Compute per-example pair features for a node's values vs the task's
/// expected outputs, then aggregate to `CAND_FEAT_DIM`.
pub fn cand_features(node_values: &[Value], expected: &[Value]) -> [f32; CAND_FEAT_DIM] {
    debug_assert_eq!(node_values.len(), expected.len());
    let per_ex: Vec<Vec<f32>> = node_values
        .iter()
        .zip(expected.iter())
        .map(|(v, t)| value_pair_features(v, t).to_vec())
        .collect();
    let agg = aggregate_examples(&per_ex);
    let mut out = [0.0; CAND_FEAT_DIM];
    out.copy_from_slice(&agg);
    out
}

/// Encode the task itself: per-example solo features of the *target*,
/// aggregated. Constant across candidates within a search.
pub fn task_features(expected: &[Value]) -> [f32; TASK_FEAT_DIM] {
    let per_ex: Vec<Vec<f32>> = expected.iter().map(|v| value_features(v).to_vec()).collect();
    let agg = aggregate_examples(&per_ex);
    let mut out = [0.0; TASK_FEAT_DIM];
    out.copy_from_slice(&agg);
    out
}

/// State features: aggregate the pool's per-node `CAND_FEAT_DIM` features
/// across the pool. Empty pool → zero state (consistent boundary).
pub fn state_features(pool_cand_feats: &[[f32; CAND_FEAT_DIM]]) -> [f32; STATE_DIM] {
    let mut out = [0.0; STATE_DIM];
    if pool_cand_feats.is_empty() {
        return out;
    }
    let rows: Vec<Vec<f32>> = pool_cand_feats.iter().map(|r| r.to_vec()).collect();
    let agg = aggregate_examples(&rows);
    out.copy_from_slice(&agg);
    out
}

// -- training-step plumbing ---------------------------------------------

/// One sample of training data: a single state, a list of candidate
/// features (one is the positive), and the target action's index.
pub struct PolicySample {
    pub task_feat: [f32; TASK_FEAT_DIM],
    pub state_feat: [f32; STATE_DIM],
    pub candidates: Vec<[f32; CAND_FEAT_DIM]>,
    pub positive_idx: usize,
}

/// One sample of value training data: state + ground-truth in {0, 1}.
pub struct ValueSample {
    pub task_feat: [f32; TASK_FEAT_DIM],
    pub state_feat: [f32; STATE_DIM],
    pub target: f32,
}

/// Run one training step on a batch of (policy + value) samples. The
/// optimiser bookkeeping is owned by `self`.
pub fn train_step(
    net: &mut Network,
    policy_batch: &[PolicySample],
    value_batch: &[ValueSample],
) -> StepStats {
    let mut total_p_loss = 0.0;
    let mut total_v_loss = 0.0;
    let mut p_count = 0;
    let mut v_count = 0;
    let n_p = policy_batch.len() as f32;
    let n_v = value_batch.len() as f32;

    // POLICY forward+backward over each sample. Backprop scaled by 1/n_p
    // so the loss is a mean over the batch.
    for sample in policy_batch {
        let mut cand_logits: Vec<f32> = Vec::with_capacity(sample.candidates.len());
        let mut caches: Vec<(Vec<f32>, Vec<Vec<f32>>)> = Vec::with_capacity(sample.candidates.len());
        for cand in &sample.candidates {
            let mut x = Vec::with_capacity(POLICY_IN_DIM);
            x.extend_from_slice(cand);
            x.extend_from_slice(&sample.task_feat);
            x.extend_from_slice(&sample.state_feat);
            let (out, cache) = net.policy.forward(&x);
            cand_logits.push(out[0]);
            caches.push((x, cache));
        }
        let (loss, grads) = softmax_xent(&cand_logits, sample.positive_idx);
        total_p_loss += loss;
        p_count += 1;
        for (k, _) in caches.iter().enumerate() {
            let g = grads[k] / n_p;
            // backward expects dy of width 1
            let _ = net.policy.backward(&caches[k].1, vec![g]);
        }
    }

    // VALUE forward+backward.
    for sample in value_batch {
        let mut x = Vec::with_capacity(VALUE_IN_DIM);
        x.extend_from_slice(&sample.task_feat);
        x.extend_from_slice(&sample.state_feat);
        let (out, cache) = net.value.forward(&x);
        let (loss, dz) = sigmoid_bce(out[0], sample.target);
        total_v_loss += loss;
        v_count += 1;
        let _ = net.value.backward(&cache, vec![dz / n_v]);
    }

    // Apply Adam.
    if p_count > 0 {
        net.step += 1;
        adam_step(&mut net.policy, &net.adam.to_cfg(), net.step);
    }
    if v_count > 0 {
        // Re-use same step counter for `value` so the schedule shares.
        // The two MLPs are independent, but the bias-correction divisor
        // tracking the same step keeps things simple.
        let s = net.step.max(1);
        adam_step(&mut net.value, &net.adam.to_cfg(), s);
    }

    StepStats {
        policy_loss: if p_count > 0 { total_p_loss / p_count as f32 } else { 0.0 },
        value_loss: if v_count > 0 { total_v_loss / v_count as f32 } else { 0.0 },
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub policy_loss: f32,
    pub value_loss: f32,
}

fn sigmoid(z: f32) -> f32 { 1.0 / (1.0 + (-z).exp()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_logit_runs() {
        let mut rng = Rng::new(1);
        let net = Network::new(&NetworkCfg::default(), &mut rng);
        let cand = [0.0; CAND_FEAT_DIM];
        let task = [0.0; TASK_FEAT_DIM];
        let state = [0.0; STATE_DIM];
        let _ = net.policy_logit(&cand, &task, &state);
    }
}
