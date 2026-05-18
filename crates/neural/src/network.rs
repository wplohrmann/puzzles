//! Top-level network: leaf tables + `app_net` + `phi` + cross-attention
//! + `q_head`. One scalar `q(f, a | task)` per candidate pair.
//!
//! See `docs/02-neural.md`. All embedding tensors are rank-2 with shape
//! `(1, N)`; the cross-attention pooler vectorises across examples.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

use lang::arena::{Arena, NodeId};
use lang::eval::Value;
use lang::library::Library;

use crate::attn::CrossAttn;
use crate::embed::{
    h_struct, h_target, h_value, AppNet, EmbedCache, LeafTables, ListPairIds,
};
use crate::heads::{PhiMlp, QHead};

/// Hyperparameters for the network.
#[derive(Clone, Copy, Debug)]
pub struct NetworkCfg {
    pub n: usize,
    pub app_hidden: usize,
    pub phi_hidden: usize,
    pub q_hidden: usize,
    pub lr: f64,
    pub weight_decay: f64,
}

impl Default for NetworkCfg {
    fn default() -> Self {
        Self {
            n: 32,
            app_hidden: 128,
            phi_hidden: 64,
            q_hidden: 64,
            lr: 3e-3,
            weight_decay: 0.0,
        }
    }
}

pub struct Network {
    pub cfg: NetworkCfg,
    pub device: Device,
    pub varmap: VarMap,
    pub leaves: LeafTables,
    pub app_net: AppNet,
    pub phi: PhiMlp,
    pub attn: CrossAttn,
    pub q_head: QHead,
    pub lp: ListPairIds,
    pub lib_size: usize,
    pub model_version: u64,
}

impl Network {
    pub fn new(cfg: &NetworkCfg, lib: &Library, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lib_size = lib.len();
        let leaves = LeafTables::build(&vb.pp("leaves"), cfg.n, lib_size)?;
        let app_net = AppNet::build(&vb.pp("app"), cfg.n, cfg.app_hidden)?;
        let phi = PhiMlp::build(&vb.pp("phi"), cfg.n, cfg.phi_hidden)?;
        let attn = CrossAttn::build(&vb.pp("attn"), cfg.n)?;
        let q_head = QHead::build(&vb.pp("q"), cfg.n, cfg.q_hidden)?;
        let lp = ListPairIds::from_library(lib);
        Ok(Self {
            cfg: *cfg,
            device,
            varmap,
            leaves,
            app_net,
            phi,
            attn,
            q_head,
            lp,
            lib_size,
            model_version: 0,
        })
    }

    /// Convenience: get all trainable parameters for the optimizer.
    pub fn all_vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    /// Compute `q(f, a)` for one candidate pair given a per-task
    /// pre-computed embedding cache.
    ///
    /// `inputs` are the task's example inputs (used for h_value's
    /// closure case); `targets` are the example outputs already
    /// embedded as a `(K, N)` stack.
    pub fn q_score(
        &self,
        f_node: NodeId,
        a_node: NodeId,
        arena: &Arena,
        lib: &Library,
        inputs: &[Value],
        target_rows: &Tensor,
        fuel: u32,
        cache: &mut EmbedCache,
    ) -> Result<Tensor> {
        let n = self.cfg.n;
        let k = inputs.len();
        debug_assert_eq!(target_rows.dims2()?, (k, n));

        // Structural embeddings.
        let hf_struct = h_struct(f_node, arena, &self.leaves, &self.app_net, cache)?;
        let ha_struct = h_struct(a_node, arena, &self.leaves, &self.app_net, cache)?;
        let struct_pair = Tensor::cat(&[&hf_struct, &ha_struct], 1)?; // (1, 2N)

        // Per-example value embeddings → (K, N) stacks.
        let mut hf_rows = Vec::with_capacity(k);
        let mut ha_rows = Vec::with_capacity(k);
        for (i, x_i) in inputs.iter().enumerate() {
            let hf_i = h_value(
                f_node, i, x_i, arena, lib, &self.leaves, &self.app_net, self.lp, fuel, cache,
            )?;
            let ha_i = h_value(
                a_node, i, x_i, arena, lib, &self.leaves, &self.app_net, self.lp, fuel, cache,
            )?;
            hf_rows.push(hf_i);
            ha_rows.push(ha_i);
        }
        let hf_stack = Tensor::cat(&hf_rows.iter().collect::<Vec<_>>(), 0)?; // (K, N)
        let ha_stack = Tensor::cat(&ha_rows.iter().collect::<Vec<_>>(), 0)?;

        // per_ex[i] = phi([hf_i, ha_i, target_i]).
        let phi_in = Tensor::cat(&[&hf_stack, &ha_stack, target_rows], 1)?; // (K, 3N)
        let per_ex = self.phi.forward(&phi_in)?;

        // Cross-attention pool over examples.
        let ctx = self.attn.forward(&struct_pair, &per_ex)?; // (1, N)

        // q_head([ctx, h_struct(f), h_struct(a)]) → (1, 1)
        let q_in = Tensor::cat(&[&ctx, &hf_struct, &ha_struct], 1)?;
        let out = self.q_head.forward(&q_in)?;
        Ok(out.flatten_all()?) // (1,)
    }

    /// Pre-compute the target stack `(K, N)` once per task.
    pub fn target_stack(&self, targets: &[Value]) -> Result<Tensor> {
        let mut rows = Vec::with_capacity(targets.len());
        for y in targets {
            let row = h_target(y, &self.leaves, &self.app_net, self.lp)?;
            rows.push(row);
        }
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)
    }

    /// Save trainable params to a safetensors file.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.varmap.save(path)
    }

    /// Load trainable params from a safetensors file. Network must be
    /// constructed with the same shapes first.
    pub fn load(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.varmap.load(path)
    }
}

/// Helper: extract f32 scalar from a 0-d / 1-element tensor.
pub fn scalar_f32(t: &Tensor) -> Result<f32> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(v[0])
}
