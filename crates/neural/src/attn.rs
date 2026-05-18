//! Cross-attention pooler over the task's examples.
//!
//! Query is built from the candidate pair's structural embedding;
//! keys/values are the per-example projections from `phi`.

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

pub struct CrossAttn {
    pub n: usize,
    pub w_q: Linear, // 2N -> N
    pub w_k: Linear, // N -> N
    pub w_v: Linear, // N -> N
}

impl CrossAttn {
    pub fn build(vb: &VarBuilder, n: usize) -> Result<Self> {
        let w_q = linear_no_bias(2 * n, n, vb.pp("w_q"))?;
        let w_k = linear_no_bias(n, n, vb.pp("w_k"))?;
        let w_v = linear_no_bias(n, n, vb.pp("w_v"))?;
        Ok(Self { n, w_q, w_k, w_v })
    }

    /// Inputs:
    /// - `struct_pair`: `(1, 2N)`
    /// - `per_ex`: `(K, N)`
    ///
    /// Output: `(1, N)` context, attention-pooled over the K examples.
    pub fn forward(&self, struct_pair: &Tensor, per_ex: &Tensor) -> Result<Tensor> {
        let n = self.n as f64;
        let scale = 1.0 / n.sqrt();
        let query = self.w_q.forward(struct_pair)?;          // (1, N)
        let keys = self.w_k.forward(per_ex)?;                // (K, N)
        let vals = self.w_v.forward(per_ex)?;                // (K, N)
        // scores[i] = (keys[i] · query) / sqrt(N)
        let prod = keys.broadcast_mul(&query)?;              // (K, N)
        let scores = prod.sum_keepdim(1)?.affine(scale, 0.0)?; // (K, 1)
        let scores = scores.flatten_all()?;                   // (K,)
        let alphas = candle_nn::ops::softmax(&scores, 0)?;    // (K,)
        let weighted = vals.broadcast_mul(&alphas.unsqueeze(1)?)?; // (K, N)
        let ctx = weighted.sum_keepdim(0)?;                   // (1, N)
        Ok(ctx)
    }
}
