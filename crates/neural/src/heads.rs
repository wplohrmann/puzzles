//! Per-example projection (`phi`) and the `q_head`.
//!
//! See `docs/02-neural.md`. `phi: R^{3N} → R^N` projects the concatenated
//! `(h_value(f), h_value(a), h_target)` per example. `q_head` consumes
//! the attention context plus the candidate's structural embeddings.

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// `phi: R^{3N} → R^N`. Two-layer MLP, ReLU between, linear out.
pub struct PhiMlp {
    pub n: usize,
    pub w1: Linear,
    pub w2: Linear,
}

impl PhiMlp {
    pub fn build(vb: &VarBuilder, n: usize, hidden: usize) -> Result<Self> {
        let w1 = linear(3 * n, hidden, vb.pp("w1"))?;
        let w2 = linear(hidden, n, vb.pp("w2"))?;
        Ok(Self { n, w1, w2 })
    }

    /// Forward over `(K, 3N) → (K, N)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w1.forward(x)?.relu()?;
        self.w2.forward(&h)
    }
}

/// `q_head: R^{3N} → R^1`. Two-layer MLP.
pub struct QHead {
    pub n: usize,
    pub w1: Linear,
    pub w2: Linear,
}

impl QHead {
    pub fn build(vb: &VarBuilder, n: usize, hidden: usize) -> Result<Self> {
        let w1 = linear(3 * n, hidden, vb.pp("w1"))?;
        let w2 = linear(hidden, 1, vb.pp("w2"))?;
        Ok(Self { n, w1, w2 })
    }

    /// Forward over `(1, 3N) → (1, 1)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w1.forward(x)?.relu()?;
        self.w2.forward(&h)
    }
}
