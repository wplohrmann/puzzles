//! Poser head: scores `(f, a)` candidates for the poser-search, with
//! the same input shape as the q-head.
//!
//! Input:  `[ctx, h_struct(f), h_struct(a)]`, shape `(1, 3N)` — exactly
//!         the same tensor the q-head consumes. (The `ctx` here is
//!         computed by the same cross-attention pooler over the task's
//!         examples; in the poser's case the "examples" are the
//!         currently-evaluated values of the pool's seed nodes on a
//!         set of probe inputs.)
//! Output: scalar `q_poser(f, a)`, used as priority in the
//!         poser-guided search.
//!
//! Trained by REINFORCE against the shaped poser tent reward (peak at
//! `β · N_poser` frontier expansions), with an EMA baseline and
//! **stop-grad at the trunk**. The poser's adversarial objective could
//! corrupt the trunk's representation if its gradient flowed in, so
//! only the poser-head's MLP weights get gradient from the poser loss.

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// `poser_head: R^{3N} → R^1`. Two-layer MLP, same shape as `QHead`
/// but trained against a different objective.
pub struct PoserHead {
    pub n: usize,
    pub w1: Linear,
    pub w2: Linear,
}

impl PoserHead {
    pub fn build(vb: &VarBuilder, n: usize, hidden: usize) -> Result<Self> {
        let w1 = linear(3 * n, hidden, vb.pp("w1"))?;
        let w2 = linear(hidden, 1, vb.pp("w2"))?;
        Ok(Self { n, w1, w2 })
    }

    /// Forward over `(1, 3N) → (1, 1)`. Caller stacks batched inputs
    /// along dim 0 to score multiple frontier candidates in one shot.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w1.forward(x)?.relu()?;
        self.w2.forward(&h)
    }

    /// Trainable parameters as `Var`s, for plumbing the poser's
    /// stop-grad-at-trunk update (the poser loss should only update
    /// the head's params, not the trunk). The optimizer wires this up
    /// at the `Network` level.
    pub fn params<'a>(&'a self) -> [&'a Linear; 2] {
        [&self.w1, &self.w2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn poser_head_shapes() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let n = 16;
        let hidden = 32;
        let head = PoserHead::build(&vb.pp("poser"), n, hidden).unwrap();
        // Score 4 frontier candidates in one batch.
        let x = Tensor::randn(0.0f32, 1.0f32, (4, 3 * n), &dev).unwrap();
        let y = head.forward(&x).unwrap();
        assert_eq!(y.dims2().unwrap(), (4, 1));
    }
}
