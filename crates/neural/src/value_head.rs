//! Value head: per-node scalar predicting the **per-action credit**
//! that would be assigned to the action creating this node.
//!
//! Input:  `h_struct(node)` — a single node's structural embedding,
//!         shape `(1, N)`.
//! Output: scalar `V(node)` (returned as a `(1, 1)` tensor, caller can
//!         flatten if needed).
//!
//! This is an **action-conditional baseline** for the q-head's
//! actor-critic loss: at each search step with action `a_t` creating
//! node `n_t`, advantage is `C_t − V(n_t).detach()`. Strictly speaking
//! an action-conditional baseline introduces a small policy-gradient
//! bias, but in practice it's a standard and effective variance
//! reduction. Per-node, no pool aggregation — keeps things simple and
//! cheap.
//!
//! Trained by regression: `loss_V(n_t) = (V(n_t) − C_t)²` over every
//! search step on every trajectory (solved or not). V learns "given
//! the trunk thinks node `n` looks like *this*, what was the credit
//! assigned to its creation action?".

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// `value_head: R^N → R^1`. Two-layer MLP.
pub struct ValueHead {
    pub n: usize,
    pub w1: Linear,
    pub w2: Linear,
}

impl ValueHead {
    pub fn build(vb: &VarBuilder, n: usize, hidden: usize) -> Result<Self> {
        let w1 = linear(n, hidden, vb.pp("w1"))?;
        let w2 = linear(hidden, 1, vb.pp("w2"))?;
        Ok(Self { n, w1, w2 })
    }

    /// Forward over `(B, N) → (B, 1)`. Caller may pass batched node
    /// embeddings stacked along dim 0 to score many nodes in one shot.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w1.forward(x)?.relu()?;
        self.w2.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn value_head_shapes() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let n = 16;
        let hidden = 32;
        let head = ValueHead::build(&vb.pp("v"), n, hidden).unwrap();

        // Batched eval over 5 nodes.
        let x = Tensor::randn(0.0f32, 1.0f32, (5, n), &dev).unwrap();
        let y = head.forward(&x).unwrap();
        assert_eq!(y.dims2().unwrap(), (5, 1));
    }
}
