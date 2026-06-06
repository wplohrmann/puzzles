//! Forward-prediction head: predict the per-example value embedding of
//! a composite node from its children's embeddings.
//!
//! Input per example `i`:
//!   `[h_struct(f),  h_struct(a),  h_value(f, i),  h_value(a, i)]`
//! shape `(K, 4N)` for a batch of K examples.
//!
//! Output: predicted `h_value(App(f, a), i)`, shape `(K, N)`.
//!
//! Trained against the *actual* `h_value(App(f, a), i)` computed by the
//! existing trunk walk — with the target side **detached**. The
//! stop-grad on the target prevents the cheap solution "let the trunk
//! collapse all `h_value`s to a constant and the predictor learn to
//! match it". SIGReg provides a second, independent anti-collapse
//! pressure.
//!
//! Why this head matters: the q-head only trains the trunk when a
//! search succeeds. The forward head trains the trunk on *every*
//! sampled program — every internal `App` node contributes one
//! supervised sample. So the embeddings stay sharp even when the
//! actor is silent (e.g. cold-start, when the searcher solves nothing
//! and the policy-gradient term is zero).

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// `forward_head: R^{4N} → R^N`. Two-layer MLP, ReLU between, linear
/// out. Mirrors `PhiMlp` in shape but takes one extra `(1, N)` block
/// of input (the parent's `h_value` for the example is what we
/// predict).
pub struct ForwardHead {
    pub n: usize,
    pub w1: Linear,
    pub w2: Linear,
}

impl ForwardHead {
    pub fn build(vb: &VarBuilder, n: usize, hidden: usize) -> Result<Self> {
        let w1 = linear(4 * n, hidden, vb.pp("w1"))?;
        let w2 = linear(hidden, n, vb.pp("w2"))?;
        Ok(Self { n, w1, w2 })
    }

    /// Forward over `(K, 4N) → (K, N)`.
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
    fn forward_head_shapes() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let n = 16;
        let hidden = 32;
        let head = ForwardHead::build(&vb.pp("fwd"), n, hidden).unwrap();

        let k = 3;
        let x = Tensor::randn(0.0f32, 1.0f32, (k, 4 * n), &dev).unwrap();
        let y = head.forward(&x).unwrap();
        assert_eq!(y.dims2().unwrap(), (k, n));
    }
}
