//! Top-level network: leaf tables + `app_net` + `phi` + cross-attention
//! + `q_head`. One scalar `q(f, a | task)` per candidate pair.
//!
//! See `docs/02-neural.md`. All embedding tensors are rank-2 with shape
//! `(1, N)`; the cross-attention pooler vectorises across examples.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

use lang::arena::{Arena, NodeId};
use lang::builtin::BuiltinId;
use lang::eval::Value;
use lang::ir::NodeKind;
use lang::library::{Library, PrimId};

use crate::attn::CrossAttn;
use crate::embed::{
    h_struct, h_target, h_value, AppNet, EmbedCache, LeafTables, ListPairIds,
};
use crate::forward_head::ForwardHead;
use crate::heads::{PhiMlp, QHead};
use crate::poser_head::PoserHead;
use crate::value_head::ValueHead;

/// Hyperparameters for the network.
#[derive(Clone, Copy, Debug)]
pub struct NetworkCfg {
    pub n: usize,
    pub app_hidden: usize,
    pub phi_hidden: usize,
    pub q_hidden: usize,
    /// Hidden width of the forward-prediction head (predicts
    /// `h_value(App(f, a), i)` from children embeddings).
    pub forward_hidden: usize,
    /// Hidden width of the value head (predicts per-action credit
    /// from `h_struct(node)`).
    pub value_hidden: usize,
    /// Hidden width of the poser head (scores `(f, a)` candidates
    /// for the poser-search).
    pub poser_hidden: usize,
    /// Static logit bias added to the poser-head's output when the
    /// candidate's `f_node` is the `stop` primitive. Biases the poser
    /// toward terminating with small programs at cold-start —
    /// otherwise random init makes `App(stop, _)` exponentially
    /// unlikely to come up in the search's budget. The bias is
    /// constant (not trained); the poser head's MLP can learn to
    /// emit a compensating negative score if it wants to override it.
    pub poser_stop_bias: f32,
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
            forward_hidden: 64,
            value_hidden: 64,
            poser_hidden: 64,
            poser_stop_bias: 3.0,
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
    pub forward_head: ForwardHead,
    pub value_head: ValueHead,
    pub poser_head: PoserHead,
    pub lp: ListPairIds,
    pub lib_size: usize,
    pub model_version: u64,
    /// PrimId of the `stop` builtin in the library, if present.
    /// Used by `poser_score` to apply the cold-start logit bias.
    pub stop_prim: Option<PrimId>,
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
        let forward_head = ForwardHead::build(&vb.pp("forward"), cfg.n, cfg.forward_hidden)?;
        let value_head = ValueHead::build(&vb.pp("value"), cfg.n, cfg.value_hidden)?;
        let poser_head = PoserHead::build(&vb.pp("poser"), cfg.n, cfg.poser_hidden)?;
        let lp = ListPairIds::from_library(lib);
        let stop_prim = lib.lookup(BuiltinId::Stop.name());
        Ok(Self {
            cfg: *cfg,
            device,
            varmap,
            leaves,
            app_net,
            phi,
            attn,
            q_head,
            forward_head,
            value_head,
            poser_head,
            lp,
            lib_size,
            model_version: 0,
            stop_prim,
        })
    }

    /// Convenience: get all trainable parameters for the optimizer.
    pub fn all_vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    /// Compute the shared per-candidate embedding tuple:
    /// - `hf_struct`, `ha_struct`: structural embeddings (1, N) each
    /// - `ctx`: cross-attention pooled context (1, N) over examples
    /// - `hf_stack`, `ha_stack`: per-example value embeddings (K, N)
    ///
    /// The q-head and poser-head both consume `[ctx, hf_struct,
    /// ha_struct]`, and the forward-head consumes children's value
    /// embeddings — so collecting all of this once per (f, a)
    /// candidate lets a caller score multiple heads without redoing
    /// the trunk work.
    pub fn embed_pair(
        &self,
        f_node: NodeId,
        a_node: NodeId,
        arena: &Arena,
        lib: &Library,
        inputs: &[Value],
        target_rows: &Tensor,
        fuel: u32,
        cache: &mut EmbedCache,
    ) -> Result<EmbeddedPair> {
        let n = self.cfg.n;
        let k = inputs.len();
        debug_assert_eq!(target_rows.dims2()?, (k, n));

        let hf_struct = h_struct(f_node, arena, &self.leaves, &self.app_net, cache)?;
        let ha_struct = h_struct(a_node, arena, &self.leaves, &self.app_net, cache)?;
        let struct_pair = Tensor::cat(&[&hf_struct, &ha_struct], 1)?; // (1, 2N)

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

        let phi_in = Tensor::cat(&[&hf_stack, &ha_stack, target_rows], 1)?; // (K, 3N)
        let per_ex = self.phi.forward(&phi_in)?;
        let ctx = self.attn.forward(&struct_pair, &per_ex)?; // (1, N)

        Ok(EmbeddedPair {
            hf_struct,
            ha_struct,
            hf_stack,
            ha_stack,
            ctx,
        })
    }

    /// Compute `q(f, a)` for one candidate pair given a per-task
    /// pre-computed embedding cache.
    ///
    /// `inputs` are the task's example inputs (used for h_value's
    /// closure case); `target_rows` are the example outputs already
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
        let ep = self.embed_pair(f_node, a_node, arena, lib, inputs, target_rows, fuel, cache)?;
        let q_in = Tensor::cat(&[&ep.ctx, &ep.hf_struct, &ep.ha_struct], 1)?;
        let out = self.q_head.forward(&q_in)?;
        out.flatten_all() // (1,)
    }

    /// Compute `q_poser(f, a)` — scores `(f, a)` as a poser-search
    /// candidate. Same input shape as `q_score`; just different head.
    /// Applies the constant `poser_stop_bias` when `f_node` is the
    /// `stop` primitive — biases the poser toward producing short
    /// `App(stop, _)` programs at cold-start.
    pub fn poser_score(
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
        let ep = self.embed_pair(f_node, a_node, arena, lib, inputs, target_rows, fuel, cache)?;
        let q_in = Tensor::cat(&[&ep.ctx, &ep.hf_struct, &ep.ha_struct], 1)?;
        let out = self.poser_head.forward(&q_in)?.flatten_all()?;

        // Static stop-bias: detect f_node == PrimRef(stop_prim) and
        // add the bias constant. Same logic at sample-time (in
        // solve_guided_training's enqueue) and at loss-time (when
        // re-scoring candidates), so log π(a|s) is consistent.
        if let Some(stop) = self.stop_prim {
            if let NodeKind::PrimRef(p) = arena.kind(f_node) {
                if *p == stop {
                    return out.affine(1.0, self.cfg.poser_stop_bias as f64);
                }
            }
        }
        Ok(out)
    }

    /// Compute the value head output `V(node)` — a scalar predicting
    /// the per-action credit that would be assigned to the action
    /// creating this node. Input = `h_struct(node)`.
    pub fn value_score(
        &self,
        node: NodeId,
        arena: &Arena,
        cache: &mut EmbedCache,
    ) -> Result<Tensor> {
        let h = h_struct(node, arena, &self.leaves, &self.app_net, cache)?;
        let out = self.value_head.forward(&h)?;
        out.flatten_all() // (1,)
    }

    /// Run the forward-prediction head on one `(f, a, example_idx)`
    /// triple: predict `h_value(App(f, a), i)` from children embeddings.
    /// Returns the prediction `(1, N)`; the caller computes the loss
    /// against the *detached* actual `h_value(App(f, a), i)`.
    pub fn forward_predict(
        &self,
        f_node: NodeId,
        a_node: NodeId,
        example_idx: usize,
        input: &Value,
        arena: &Arena,
        lib: &Library,
        fuel: u32,
        cache: &mut EmbedCache,
    ) -> Result<Tensor> {
        let hf_s = h_struct(f_node, arena, &self.leaves, &self.app_net, cache)?;
        let ha_s = h_struct(a_node, arena, &self.leaves, &self.app_net, cache)?;
        let hf_v = h_value(
            f_node, example_idx, input, arena, lib,
            &self.leaves, &self.app_net, self.lp, fuel, cache,
        )?;
        let ha_v = h_value(
            a_node, example_idx, input, arena, lib,
            &self.leaves, &self.app_net, self.lp, fuel, cache,
        )?;
        let x = Tensor::cat(&[&hf_s, &ha_s, &hf_v, &ha_v], 1)?; // (1, 4N)
        self.forward_head.forward(&x) // (1, N)
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

/// Cached per-`(f, a)` embedding tuple used by the heads that share
/// the 3N input format (q-head, poser-head). Returned by
/// `Network::embed_pair` so a caller can score multiple heads on the
/// same candidate without redoing the trunk work.
#[derive(Clone)]
pub struct EmbeddedPair {
    pub hf_struct: Tensor, // (1, N)
    pub ha_struct: Tensor, // (1, N)
    pub hf_stack: Tensor,  // (K, N)
    pub ha_stack: Tensor,  // (K, N)
    pub ctx: Tensor,       // (1, N)
}

/// Helper: extract f32 scalar from a 0-d / 1-element tensor.
pub fn scalar_f32(t: &Tensor) -> Result<f32> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(v[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang::arena::Arena;
    use lang::builtin::seed_builtin_library;
    use lang::construct::{app, lit, param, prim_ref};
    use lang::ir::LitValue;

    /// All four heads forward-pass cleanly on a fresh network, on the
    /// same `(f, a)` candidate, with consistent shapes.
    #[test]
    fn all_heads_forward() {
        let lib = seed_builtin_library();
        let cfg = NetworkCfg { n: 16, ..NetworkCfg::default() };
        let net = Network::new(&cfg, &lib, Device::Cpu).expect("build");

        // Tiny candidate: App(add, 1).
        let mut arena = Arena::new();
        let add_id = lib.lookup("add").unwrap();
        let add_ref = prim_ref(&mut arena, add_id);
        let one = lit(&mut arena, LitValue::Int(1));
        let p = param(&mut arena, 0);
        let _app_add_one = app(&mut arena, add_ref, one);

        let inputs = vec![Value::Int(2), Value::Int(3), Value::Int(4)];
        let targets = vec![Value::Int(3), Value::Int(4), Value::Int(5)];
        let target_rows = net.target_stack(&targets).unwrap();

        let mut cache = EmbedCache::default();
        let q = net.q_score(add_ref, p, &arena, &lib, &inputs, &target_rows, 1_000, &mut cache).unwrap();
        let pos = net.poser_score(add_ref, p, &arena, &lib, &inputs, &target_rows, 1_000, &mut cache).unwrap();
        let val = net.value_score(add_ref, &arena, &mut cache).unwrap();
        let pred = net.forward_predict(add_ref, p, 0, &Value::Int(2), &arena, &lib, 1_000, &mut cache).unwrap();

        assert_eq!(q.flatten_all().unwrap().dims1().unwrap(), 1);
        assert_eq!(pos.flatten_all().unwrap().dims1().unwrap(), 1);
        assert_eq!(val.flatten_all().unwrap().dims1().unwrap(), 1);
        assert_eq!(pred.dims2().unwrap(), (1, cfg.n));
    }
}
