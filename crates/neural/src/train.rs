//! One gradient step: softmax-CE over candidate `(f, a)` pairs.
//!
//! See `docs/02-neural.md` § Training. The loss for a single step is
//! `-log( exp(q(f+, a+)) / Σ_C exp(q(f, a)) )` over a candidate set
//! `C` that contains the positive pair plus K negatives.

use candle_core::{Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use lang::arena::{Arena, NodeId};
use lang::eval::Value;
use lang::library::Library;

use crate::embed::EmbedCache;
use crate::network::Network;

/// One training datum: the candidates and the index of the positive
/// within them. All candidate pairs reference nodes interned in `arena`.
#[derive(Clone)]
pub struct TrainSample {
    /// `(f, a)` pairs to score. Index `positive_idx` is the dream's
    /// actual next pair.
    pub candidates: Vec<(NodeId, NodeId)>,
    pub positive_idx: usize,
    /// The task's example inputs (used for the value walk).
    pub inputs: Vec<Value>,
    /// The task's example outputs.
    pub targets: Vec<Value>,
}

#[derive(Clone, Debug, Default)]
pub struct StepStats {
    pub loss: f32,
    pub samples: usize,
    pub positive_top1: f32,
    /// Per-sample correctness (parallel to the input `samples` slice).
    /// `true` if the positive pair had the highest logit. Callers that
    /// want bucketed top-1 (e.g. only on samples from C=k dreams)
    /// filter this against their own per-sample tags.
    pub per_sample_correct: Vec<bool>,
}

/// One optimizer step over a batch of samples. Each sample contributes
/// one softmax CE term (positive vs candidates). Losses are summed,
/// `.backward()` runs once, then Adam steps.
pub fn train_step(
    net: &Network,
    opt: &mut AdamW,
    samples: &[(&TrainSample, &Arena, &Library)],
    fuel: u32,
) -> Result<StepStats> {
    if samples.is_empty() {
        return Ok(StepStats::default());
    }
    let dev = &net.device;
    let mut total_loss: Option<Tensor> = None;
    let mut sample_count = 0usize;
    let mut correct_top1 = 0usize;
    let mut per_sample_correct: Vec<bool> = Vec::with_capacity(samples.len());

    for (sample, arena, lib) in samples {
        if sample.candidates.is_empty() {
            per_sample_correct.push(false);
            continue;
        }
        // Per-sample cache: a fresh one each time so old graph nodes
        // don't accidentally cross sample boundaries.
        let mut cache = EmbedCache::default();
        let target_rows = net.target_stack(&sample.targets)?;
        let mut logits: Vec<Tensor> = Vec::with_capacity(sample.candidates.len());
        for (f_node, a_node) in &sample.candidates {
            let q = net.q_score(
                *f_node, *a_node, arena, lib, &sample.inputs, &target_rows, fuel, &mut cache,
            )?;
            logits.push(q);
        }
        let stacked = Tensor::cat(&logits.iter().collect::<Vec<_>>(), 0)?; // (C,)

        // Track top-1 accuracy.
        let logit_values: Vec<f32> = stacked.to_vec1()?;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logit_values.iter().enumerate() {
            if v > best_val { best_val = v; best_idx = i; }
        }
        let is_correct = best_idx == sample.positive_idx;
        if is_correct { correct_top1 += 1; }
        per_sample_correct.push(is_correct);

        // Numerically-stable log-sum-exp.
        let stacked2 = stacked.unsqueeze(0)?; // (1, C)
        let log_softmax = candle_nn::ops::log_softmax(&stacked2, 1)?;
        let pos = sample.positive_idx;
        let pos_log_prob = log_softmax.narrow(1, pos, 1)?.flatten_all()?;
        let loss = pos_log_prob.neg()?;
        total_loss = Some(match total_loss {
            None => loss,
            Some(t) => (t + loss)?,
        });
        sample_count += 1;
    }

    if sample_count == 0 {
        return Ok(StepStats::default());
    }

    let n_t = sample_count as f64;
    let loss_t = total_loss.unwrap().affine(1.0 / n_t, 0.0)?;
    let loss_val: Vec<f32> = loss_t.to_vec1().or_else(|_| {
        loss_t.flatten_all()?.to_vec1::<f32>()
    })?;
    let scalar_loss = loss_val[0];

    let grads = loss_t.backward()?;
    opt.step(&grads)?;
    let _ = dev;

    Ok(StepStats {
        loss: scalar_loss,
        samples: sample_count,
        positive_top1: correct_top1 as f32 / sample_count as f32,
        per_sample_correct,
    })
}

/// Build a fresh Adam(W) optimizer over the network's params.
pub fn make_optimizer(net: &Network, lr: f64, weight_decay: f64) -> Result<AdamW> {
    let params = ParamsAdamW {
        lr,
        weight_decay,
        ..ParamsAdamW::default()
    };
    AdamW::new(net.all_vars(), params)
}
