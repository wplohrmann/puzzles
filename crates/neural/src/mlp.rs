//! A small MLP with explicit per-layer forward/backward and Adam.
//!
//! Pure Rust, hand-derived gradients. Activations are ReLU between hidden
//! layers; the head returns its raw pre-activation (logit / score) and
//! the loss layer is responsible for any nonlinearity (sigmoid for value,
//! softmax for policy).
//!
//! See `decisions/02-pure-rust-nn.md` for why we're not using a framework.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// A linear (Dense) layer: `out = x · W + b`.
///
/// Weights are stored in row-major order; `weights[i*out_dim + j]` is the
/// connection from input `i` to output `j`. Forward is `out_j = b_j +
/// sum_i x_i * w_{i,j}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    /// Adam moments — kept inside the layer so train_step is a single
    /// pass over the layer list.
    #[serde(skip, default)]
    pub m_w: Vec<f32>,
    #[serde(skip, default)]
    pub v_w: Vec<f32>,
    #[serde(skip, default)]
    pub m_b: Vec<f32>,
    #[serde(skip, default)]
    pub v_b: Vec<f32>,
    /// Accumulated gradients (zeroed each train_step after applying).
    #[serde(skip, default)]
    pub g_w: Vec<f32>,
    #[serde(skip, default)]
    pub g_b: Vec<f32>,
}

impl Linear {
    /// Kaiming-uniform init: `U[-sqrt(6/in_dim), sqrt(6/in_dim)]`.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        let bound = (6.0 / in_dim.max(1) as f32).sqrt();
        let mut weights = Vec::with_capacity(in_dim * out_dim);
        for _ in 0..in_dim * out_dim {
            weights.push(rng.uniform_centered(bound));
        }
        let bias = vec![0.0; out_dim];
        Self {
            in_dim, out_dim,
            weights, bias,
            m_w: vec![0.0; in_dim * out_dim],
            v_w: vec![0.0; in_dim * out_dim],
            m_b: vec![0.0; out_dim],
            v_b: vec![0.0; out_dim],
            g_w: vec![0.0; in_dim * out_dim],
            g_b: vec![0.0; out_dim],
        }
    }

    /// Restore Adam/grad scratch buffers after a deserialise.
    pub fn rehydrate_scratch(&mut self) {
        self.m_w = vec![0.0; self.in_dim * self.out_dim];
        self.v_w = vec![0.0; self.in_dim * self.out_dim];
        self.m_b = vec![0.0; self.out_dim];
        self.v_b = vec![0.0; self.out_dim];
        self.g_w = vec![0.0; self.in_dim * self.out_dim];
        self.g_b = vec![0.0; self.out_dim];
    }

    /// `y = x · W + b`. Returns a fresh `out` vector.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut out = self.bias.clone();
        // out[j] += sum_i x[i] * W[i*out + j]
        for i in 0..self.in_dim {
            let xi = x[i];
            if xi == 0.0 { continue; }
            let row = &self.weights[i * self.out_dim..(i + 1) * self.out_dim];
            for j in 0..self.out_dim {
                out[j] += xi * row[j];
            }
        }
        out
    }

    /// Backward pass. Given the input `x` used in forward and the gradient
    /// w.r.t. the output `dy`, accumulate weight/bias grads and return the
    /// gradient w.r.t. `x`.
    pub fn backward(&mut self, x: &[f32], dy: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(dy.len(), self.out_dim);
        // Bias grad: dy
        for j in 0..self.out_dim {
            self.g_b[j] += dy[j];
        }
        // Weight grad: x ⊗ dy, dx = dy · W^T
        let mut dx = vec![0.0; self.in_dim];
        for i in 0..self.in_dim {
            let xi = x[i];
            let row_g = &mut self.g_w[i * self.out_dim..(i + 1) * self.out_dim];
            let row_w = &self.weights[i * self.out_dim..(i + 1) * self.out_dim];
            let mut accum = 0.0;
            for j in 0..self.out_dim {
                row_g[j] += xi * dy[j];
                accum += row_w[j] * dy[j];
            }
            dx[i] = accum;
        }
        dx
    }
}

/// A multi-layer perceptron with ReLU between hidden layers and a linear
/// head.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Mlp {
    /// Build an MLP with the given dimensions. `dims = [in, h1, ..., out]`.
    pub fn new(dims: &[usize], rng: &mut Rng) -> Self {
        assert!(dims.len() >= 2);
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for w in dims.windows(2) {
            layers.push(Linear::new(w[0], w[1], rng));
        }
        Self { layers }
    }

    pub fn rehydrate_scratch(&mut self) {
        for l in &mut self.layers { l.rehydrate_scratch(); }
    }

    /// Forward pass. Returns the output and the activation cache (one
    /// entry per Linear layer's *input*, i.e. `cache[k]` is the input
    /// to `layers[k]`). The post-ReLU activations are returned via the
    /// next layer's input cache.
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut cache: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        let mut h = x.to_vec();
        for (k, layer) in self.layers.iter().enumerate() {
            cache.push(h.clone());
            let mut z = layer.forward(&h);
            // ReLU between hidden layers; the final layer is linear.
            if k + 1 < self.layers.len() {
                for v in &mut z { if *v < 0.0 { *v = 0.0; } }
            }
            h = z;
        }
        (h, cache)
    }

    /// Backward pass given the gradient w.r.t. the head's pre-activation
    /// output. Returns the gradient w.r.t. the MLP input.
    pub fn backward(&mut self, cache: &[Vec<f32>], mut dy: Vec<f32>) -> Vec<f32> {
        // Walk layers in reverse.
        for k in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[k];
            let x = &cache[k];
            // ReLU was applied *after* every layer except the last. So if
            // we're upstream of a ReLU, the gradient comes through the
            // ReLU mask of the next layer's input. The next layer's input
            // (in `cache[k+1]`) is exactly that post-ReLU activation.
            // We mask `dy` here only when this layer is *not* the last.
            //
            // Wait — the order: head's pre-activation has no ReLU, so the
            // initial dy is correct. After we backward through the head,
            // we've got dx = grad w.r.t. cache[head]. cache[head] is the
            // post-ReLU activation of the previous layer. So the *next*
            // backward step (the ReLU mask) needs to apply *before*
            // calling backward on the previous layer.
            let dx = layer.backward(x, &dy);
            if k > 0 {
                // Apply ReLU mask using layers[k-1]'s output, which is
                // cache[k] but pre-ReLU — we don't store pre-ReLU. We
                // recompute it from cache[k-1].
                // The standard trick: ReLU mask is `cache[k] > 0`
                // (post-ReLU == 0 iff pre-ReLU ≤ 0).
                let post = &cache[k];
                let mut masked = dx;
                for i in 0..masked.len() {
                    if post[i] <= 0.0 {
                        masked[i] = 0.0;
                    }
                }
                dy = masked;
            } else {
                return dx;
            }
        }
        unreachable!("MLP has at least one layer");
    }
}

/// Adam optimiser hyperparameters. Step count is owned by the caller (so
/// multiple modules can share an optimiser schedule).
#[derive(Clone, Copy, Debug)]
pub struct AdamCfg {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamCfg {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Apply the accumulated gradients with Adam, then zero them. `step` is
/// the 1-indexed global step counter.
pub fn adam_step(mlp: &mut Mlp, cfg: &AdamCfg, step: u64) {
    let bc1 = 1.0 - cfg.beta1.powi(step as i32);
    let bc2 = 1.0 - cfg.beta2.powi(step as i32);
    for layer in &mut mlp.layers {
        // Weights.
        for k in 0..layer.weights.len() {
            let g = layer.g_w[k] + cfg.weight_decay * layer.weights[k];
            layer.m_w[k] = cfg.beta1 * layer.m_w[k] + (1.0 - cfg.beta1) * g;
            layer.v_w[k] = cfg.beta2 * layer.v_w[k] + (1.0 - cfg.beta2) * g * g;
            let m_hat = layer.m_w[k] / bc1;
            let v_hat = layer.v_w[k] / bc2;
            layer.weights[k] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
            layer.g_w[k] = 0.0;
        }
        // Biases (no weight decay on bias).
        for k in 0..layer.bias.len() {
            let g = layer.g_b[k];
            layer.m_b[k] = cfg.beta1 * layer.m_b[k] + (1.0 - cfg.beta1) * g;
            layer.v_b[k] = cfg.beta2 * layer.v_b[k] + (1.0 - cfg.beta2) * g * g;
            let m_hat = layer.m_b[k] / bc1;
            let v_hat = layer.v_b[k] / bc2;
            layer.bias[k] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
            layer.g_b[k] = 0.0;
        }
    }
}

// -- Loss helpers ---------------------------------------------------------

/// Sigmoid binary cross-entropy. `logit` is pre-activation; `target` ∈ {0, 1}.
/// Returns `(loss, dL/dlogit)`.
pub fn sigmoid_bce(logit: f32, target: f32) -> (f32, f32) {
    // numerically-stable log-sum-exp version:
    // loss = max(z, 0) - z*y + log(1 + exp(-|z|))
    let z = logit;
    let abs_z = z.abs();
    let loss = z.max(0.0) - z * target + (1.0 + (-abs_z).exp()).ln();
    let p = 1.0 / (1.0 + (-z).exp());
    let dz = p - target;
    (loss, dz)
}

/// Softmax cross-entropy with one-hot target on `target_idx`. Returns
/// `(loss, dL/dlogits)`. `logits.len()` must be > 0.
pub fn softmax_xent(logits: &[f32], target_idx: usize) -> (f32, Vec<f32>) {
    debug_assert!(target_idx < logits.len());
    // log-sum-exp for stability
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0;
    for &l in logits {
        sum_exp += (l - max_l).exp();
    }
    let log_z = max_l + sum_exp.ln();
    let loss = log_z - logits[target_idx];
    let mut grad = Vec::with_capacity(logits.len());
    for &l in logits {
        grad.push((l - max_l).exp() / sum_exp);
    }
    grad[target_idx] -= 1.0;
    (loss, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numeric-gradient check on a tiny MLP + a scalar (sigmoid-BCE) head.
    /// If forward/backward are consistent, analytic and numeric grads
    /// agree to a few ulps.
    #[test]
    fn numeric_grad_check_sigmoid() {
        let mut rng = Rng::new(42);
        let mlp = Mlp::new(&[3, 4, 1], &mut rng);
        let x: Vec<f32> = vec![0.5, -0.3, 0.7];
        let target = 1.0;

        // Forward + analytic backward.
        let (out, cache) = mlp.forward(&x);
        let (_loss, dz) = sigmoid_bce(out[0], target);
        let mut mlp_grad = mlp.clone();
        let _dx = mlp_grad.backward(&cache, vec![dz]);

        // Numeric grad on a single weight in layer 0.
        let eps = 1e-3;
        let mut perturbed = mlp.clone();
        let idx = 5;
        perturbed.layers[0].weights[idx] += eps;
        let (out_p, _) = perturbed.forward(&x);
        let (loss_p, _) = sigmoid_bce(out_p[0], target);
        perturbed.layers[0].weights[idx] -= 2.0 * eps;
        let (out_m, _) = perturbed.forward(&x);
        let (loss_m, _) = sigmoid_bce(out_m[0], target);
        let numeric = (loss_p - loss_m) / (2.0 * eps);
        let analytic = mlp_grad.layers[0].g_w[idx];
        assert!(
            (numeric - analytic).abs() < 1e-2,
            "numeric={:.6} analytic={:.6}", numeric, analytic
        );
    }

    #[test]
    fn adam_drives_loss_down() {
        let mut rng = Rng::new(7);
        let mut mlp = Mlp::new(&[2, 8, 1], &mut rng);
        let cfg = AdamCfg { lr: 1e-2, ..AdamCfg::default() };
        // y = 1 if x[0]+x[1] > 0 else 0.
        let dataset: Vec<(Vec<f32>, f32)> = (0..200)
            .map(|i| {
                let a = (i as f32 - 100.0) / 100.0;
                let b = ((i * 31) as f32 % 200.0 - 100.0) / 100.0;
                (vec![a, b], if a + b > 0.0 { 1.0 } else { 0.0 })
            })
            .collect();

        let mut step: u64 = 0;
        let mut last_loss = f32::INFINITY;
        for epoch in 0..200 {
            let mut total = 0.0;
            for (x, y) in &dataset {
                let (out, cache) = mlp.forward(x);
                let (loss, dz) = sigmoid_bce(out[0], *y);
                total += loss;
                let _ = mlp.backward(&cache, vec![dz]);
                step += 1;
                adam_step(&mut mlp, &cfg, step);
            }
            if epoch == 199 { last_loss = total / dataset.len() as f32; }
        }
        assert!(last_loss < 0.2, "final loss too high: {}", last_loss);
    }
}
