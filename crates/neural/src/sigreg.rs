//! SIGReg — Sketched Isotropic-Gaussian Regularization.
//!
//! Project a batch of embeddings onto many random unit-norm 1-D
//! directions, then apply the **Epps–Pulley** characteristic-function
//! test to each projection to score "how far is this 1-D sample from
//! `N(0, 1)`". Mean across slices = the SIGReg loss.
//!
//! Why this works and why we use it: see Balestriero & LeCun,
//! arXiv:2511.08544 (LeJEPA). Three properties that matter for us:
//! - It pushes the embedding distribution toward `N(0, I)` — not just
//!   matching first two moments (like VICReg or Barlow Twins) but the
//!   full distribution. Prevents dimensional / topological collapse.
//! - It's a pure auxiliary regularizer — no target encoder, no EMA, no
//!   projector. Plays cleanly alongside a downstream task head on the
//!   same trunk via `total_loss = task_loss + λ · sigreg_loss(emb)`.
//! - Bounded gradient norm `O(1/B)` (Thm 4 in the paper), so it won't
//!   dominate the task gradient at λ ≈ 0.05.
//!
//! Caveat: do **not** L2-normalize or LayerNorm the embeddings just
//! before passing them to SIGReg — SIGReg targets `R^N` geometry, not
//! a sphere, and a normalized embedding can never become `N(0, I)`.

use candle_core::{Device, Result, Tensor};

/// Configuration for SIGReg. Defaults match the LeJEPA paper.
#[derive(Clone, Copy, Debug)]
pub struct SigRegCfg {
    /// Number of random 1-D projections per call. Resampled every call.
    /// 1024 is the paper default; 512 also works at smaller embedding
    /// widths.
    pub num_slices: usize,
    /// Number of trapezoidal quadrature nodes for the Epps–Pulley
    /// integral on `[-5, 5]`. 17 is the paper default.
    pub num_quad_points: usize,
}

impl Default for SigRegCfg {
    fn default() -> Self {
        Self { num_slices: 1024, num_quad_points: 17 }
    }
}

/// Compute the SIGReg loss on a `(B, N)` batch of embeddings.
///
/// `B` = number of samples (rows), `N` = embedding width (cols).
/// Output is a scalar `Tensor` (shape `()`). The caller multiplies by
/// the regularization weight `λ` and adds it to the total loss.
pub fn sigreg_loss(z: &Tensor, cfg: &SigRegCfg) -> Result<Tensor> {
    let (b, n) = z.dims2()?;
    let device = z.device();
    let m = cfg.num_slices;
    let q = cfg.num_quad_points;
    assert!(q >= 2, "need at least 2 quadrature points");

    // Sample `(N, M)` Gaussian projection matrix; normalize columns to
    // unit L2 norm so each is a uniform direction on `S^{N-1}`.
    // Resampled every call — the per-step direction shuffle is what
    // gives Sobolev-rate coverage over many steps without needing
    // `M = O(2^N)`.
    let raw_a = Tensor::randn(0.0f32, 1.0f32, (n, m), device)?;
    let col_norms = mul(&raw_a, &raw_a)?
        .sum_keepdim(0)?
        .sqrt()?; // (1, M)
    let a = raw_a.broadcast_div(&col_norms)?;

    // Project: (B, N) @ (N, M) -> (B, M). The random A has no grad,
    // but gradient flows through `z`.
    let p = z.matmul(&a)?;

    // Quadrature points and trapezoidal weights on [-5, 5].
    let (t, trap_w) = quadrature(q, device)?; // both (Q,)

    // Outer product `t * p[j, m]` → (B, M, Q). We compute it as
    //   p_e: (B, M, 1)  ·  t_e: (1, 1, Q)
    let p_e = p.unsqueeze(2)?;
    let t_e = t.unsqueeze(0)?.unsqueeze(0)?;
    let tz = p_e.broadcast_mul(&t_e)?;

    // Empirical characteristic function components, averaged over B.
    //   φ̂_re(t, m) = (1/B) Σ_j cos(t · p[j, m])
    //   φ̂_im(t, m) = (1/B) Σ_j sin(t · p[j, m])
    let mean_cos = tz.cos()?.mean(0)?; // (M, Q)
    let mean_sin = tz.sin()?.mean(0)?; // (M, Q)

    // Target: standard-normal characteristic function `exp(-t²/2)`,
    // purely real. So Im target = 0.
    let t_sq = mul(&t, &t)?;
    let target_re = t_sq.affine(-0.5, 0.0)?.exp()?; // (Q,)
    let re_diff = mean_cos.broadcast_sub(&target_re.unsqueeze(0)?)?;
    let im_diff = mean_sin;

    // |φ̂ - target|² = Re² + Im².
    let mag_sq = (mul(&re_diff, &re_diff)? + mul(&im_diff, &im_diff)?)?; // (M, Q)

    // Gaussian weight `w(t) = exp(-t²/σ²)` with σ = 1 → `exp(-t²)`.
    // Multiply in the trapezoidal weights here too — both are (Q,).
    let w_t = t_sq.affine(-1.0, 0.0)?.exp()?;
    let combined_w = mul(&w_t, &trap_w)?; // (Q,)

    // Apply weight, sum over Q to get the EP value per slice.
    let weighted = mag_sq.broadcast_mul(&combined_w.unsqueeze(0)?)?; // (M, Q)
    let ep_per_slice = weighted.sum(1)?; // (M,)

    // EP statistic is `B · ∫ |φ̂ - target|² w(t) dt`. We've already
    // computed `∫ |φ̂ - target|² w(t) dt` per slice; multiply by B.
    let scaled = ep_per_slice.affine(b as f64, 0.0)?;

    // Mean across slices = the SIGReg loss (scalar).
    scaled.mean(0)
}

/// `t * t` — candle 0.10 has no `.sqr()` shorthand, so inline it.
fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.mul(b)
}

/// Trapezoidal quadrature nodes `t_k` on `[-5, 5]` and weights such
/// that `Σ_k f(t_k) · w_k ≈ ∫_{-5}^{5} f(t) dt` for smooth `f`.
fn quadrature(q: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let span = 10.0f32;
    let h = span / (q - 1) as f32;
    let mut t = Vec::with_capacity(q);
    let mut w = vec![h; q];
    w[0] = h / 2.0;
    w[q - 1] = h / 2.0;
    for k in 0..q {
        t.push(-5.0 + k as f32 * h);
    }
    let t_t = Tensor::from_vec(t, (q,), device)?;
    let w_t = Tensor::from_vec(w, (q,), device)?;
    Ok((t_t, w_t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// A batch sampled from `N(0, I)` should produce a small SIGReg
    /// loss; a degenerate (collapsed) batch should produce a much
    /// larger one. We don't compare against an absolute threshold —
    /// just the relative ordering.
    #[test]
    fn collapsed_loss_exceeds_gaussian_loss() {
        let dev = Device::Cpu;
        let cfg = SigRegCfg::default();

        // Gaussian batch (B=64, N=32).
        let z_gauss = Tensor::randn(0.0f32, 1.0f32, (64, 32), &dev).unwrap();
        let loss_gauss = sigreg_loss(&z_gauss, &cfg).unwrap();
        let v_gauss: f32 = loss_gauss.to_scalar().unwrap();

        // Collapsed batch: every row is the all-ones vector → all
        // projections are constants, far from Gaussian.
        let z_collapse = Tensor::ones((64, 32), candle_core::DType::F32, &dev).unwrap();
        let loss_collapse = sigreg_loss(&z_collapse, &cfg).unwrap();
        let v_collapse: f32 = loss_collapse.to_scalar().unwrap();

        assert!(
            v_collapse > v_gauss * 2.0,
            "expected collapsed loss ({v_collapse}) to be much larger than \
             Gaussian loss ({v_gauss}) but it wasn't",
        );
    }

    /// Gradient should flow through `z`. Build a Var, take the loss,
    /// backward, check the var got a gradient.
    #[test]
    fn loss_backprops_through_z() {
        use candle_core::{DType, Var};
        let dev = Device::Cpu;
        let cfg = SigRegCfg { num_slices: 64, num_quad_points: 9 };
        let v = Var::from_tensor(&Tensor::randn(0.0f32, 1.0f32, (32, 16), &dev).unwrap()).unwrap();
        let z = v.as_tensor();
        let loss = sigreg_loss(z, &cfg).unwrap();
        let grads = loss.backward().unwrap();
        let g = grads.get(&v).expect("z should have gradient");
        assert_eq!(g.dims2().unwrap(), (32, 16));
        // Gradient should be finite (no NaNs) for a Gaussian batch.
        let g_flat: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        for x in &g_flat {
            assert!(x.is_finite(), "non-finite grad component: {}", x);
        }
        let _ = DType::F32;
    }
}
