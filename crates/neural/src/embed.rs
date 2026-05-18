//! Embedding the DAG: leaf tables, app_net composition, structural and
//! value walks.
//!
//! See `docs/02-neural.md`. Every tensor in this module is rank 2 with
//! shape `(1, N)` — a single embedding vector. The leading "batch" dim
//! is preserved so candle's `Linear::forward` works without special
//! casing.

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, layer_norm, Linear, LayerNorm, VarBuilder};
use rustc_hash::FxHashMap;

use lang::arena::{Arena, NodeId};
use lang::builtin::BuiltinId;
use lang::eval::{eval, Value};
use lang::ir::{LitValue, NodeKind};
use lang::library::{Library, PrimId};

/// Trainable leaf-embedding tables.
pub struct LeafTables {
    pub n: usize,
    pub lib_size: usize,
    /// `(lib_size, N)`. Row `p` is the embedding for primitive `p`.
    pub prim_emb: Tensor,
    /// `(1, N)`.
    pub param_emb: Tensor,
    pub lit_int_emb: Tensor,
    pub lit_bool_emb: Tensor,
    pub lit_float_emb: Tensor,
    pub lit_char_emb: Tensor,
    pub bottom_emb: Tensor,
}

impl LeafTables {
    pub fn build(vb: &VarBuilder, n: usize, lib_size: usize) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_UNIFORM;
        let prim_emb = vb.get_with_hints((lib_size, n), "prim_emb", init)?;
        let param_emb = vb.get_with_hints((1, n), "param_emb", init)?;
        let lit_int_emb = vb.get_with_hints((1, n), "lit_int_emb", init)?;
        let lit_bool_emb = vb.get_with_hints((1, n), "lit_bool_emb", init)?;
        let lit_float_emb = vb.get_with_hints((1, n), "lit_float_emb", init)?;
        let lit_char_emb = vb.get_with_hints((1, n), "lit_char_emb", init)?;
        let bottom_emb = vb.get_with_hints((1, n), "bottom_emb", init)?;
        Ok(Self {
            n,
            lib_size,
            prim_emb,
            param_emb,
            lit_int_emb,
            lit_bool_emb,
            lit_float_emb,
            lit_char_emb,
            bottom_emb,
        })
    }

    /// Row `p` of `prim_emb` as a `(1, N)` tensor.
    pub fn prim_row(&self, p: PrimId) -> Result<Tensor> {
        let row = self.prim_emb.narrow(0, p as usize, 1)?;
        Ok(row)
    }
}

/// Replace the last slot of a `(1, N)` tensor with `value`. The first
/// `N-1` slots are taken from `base` (so gradient flows through them).
pub fn set_literal_dim(base: &Tensor, value: f32) -> Result<Tensor> {
    let n = base.dim(1)?;
    debug_assert!(n >= 1);
    let prefix = base.narrow(1, 0, n - 1)?;
    let last = Tensor::full(value, (1, 1), base.device())?;
    Tensor::cat(&[&prefix, &last], 1)
}

/// `signum(x) * ln(1 + |x|)`.
pub fn signed_log1p(x: f64) -> f32 {
    let s = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
    (s * (1.0 + x.abs()).ln()) as f32
}

/// `app_net: R^{2N} → R^N`, layer-norm on the output.
pub struct AppNet {
    pub n: usize,
    pub w1: Linear,
    pub w2: Linear,
    pub ln: LayerNorm,
}

impl AppNet {
    pub fn build(vb: &VarBuilder, n: usize, hidden: usize) -> Result<Self> {
        let w1 = linear(2 * n, hidden, vb.pp("w1"))?;
        let w2 = linear(hidden, n, vb.pp("w2"))?;
        let ln = layer_norm(n, 1e-5, vb.pp("ln"))?;
        Ok(Self { n, w1, w2, ln })
    }

    /// Compose two `(1, N)` embeddings via `app_net`.
    pub fn compose(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let x = Tensor::cat(&[a, b], 1)?;
        let h = self.w1.forward(&x)?.relu()?;
        let z = self.w2.forward(&h)?;
        self.ln.forward(&z)
    }
}

/// Identifiers for the cons / nil / pair primitives used by `embed_value`.
#[derive(Clone, Copy, Debug)]
pub struct ListPairIds {
    pub cons: PrimId,
    pub nil: PrimId,
    pub pair: PrimId,
}

impl ListPairIds {
    pub fn from_library(lib: &Library) -> Self {
        let cons = lib.lookup(BuiltinId::Cons.name()).expect("cons in seed lib");
        let nil = lib.lookup(BuiltinId::Nil.name()).expect("nil in seed lib");
        let pair = lib.lookup(BuiltinId::Pair.name()).expect("pair in seed lib");
        Self { cons, nil, pair }
    }
}

/// Embed a concrete `Value` (Int/Bool/Float/Char/List/Pair/Bottom).
/// `(1, N)` output.
pub fn embed_value(
    value: &Value,
    leaves: &LeafTables,
    app_net: &AppNet,
    lp: ListPairIds,
) -> Result<Tensor> {
    match value {
        Value::Int(v) => {
            let slot = signed_log1p(*v as f64);
            set_literal_dim(&leaves.lit_int_emb, slot)
        }
        Value::Bool(b) => {
            let slot = if *b { 1.0 } else { 0.0 };
            set_literal_dim(&leaves.lit_bool_emb, slot)
        }
        Value::Float(f) => {
            let slot = signed_log1p(*f);
            set_literal_dim(&leaves.lit_float_emb, slot)
        }
        Value::Char(c) => {
            let slot = (*c as u32) as f32 / 128.0;
            set_literal_dim(&leaves.lit_char_emb, slot)
        }
        Value::Bottom(_) => Ok(leaves.bottom_emb.clone()),
        Value::Closure(_) => {
            // Closures are runtime objects without a concrete value;
            // fall back to `bottom_emb` here. The structural walk in
            // `h_value_walk` is what really handles closures.
            Ok(leaves.bottom_emb.clone())
        }
        Value::List(xs) => {
            if xs.is_empty() {
                return leaves.prim_row(lp.nil);
            }
            // Right-fold via cons.
            let cons = leaves.prim_row(lp.cons)?;
            let mut acc = leaves.prim_row(lp.nil)?;
            for x in xs.iter().rev() {
                let xv = embed_value(x, leaves, app_net, lp)?;
                let inner = app_net.compose(&cons, &xv)?;
                acc = app_net.compose(&inner, &acc)?;
            }
            Ok(acc)
        }
        Value::Pair(p) => {
            let a = embed_value(&p.0, leaves, app_net, lp)?;
            let b = embed_value(&p.1, leaves, app_net, lp)?;
            let pair = leaves.prim_row(lp.pair)?;
            let inner = app_net.compose(&pair, &a)?;
            app_net.compose(&inner, &b)
        }
    }
}

/// Per-(model-step) caches for structural and value walks.
#[derive(Default)]
pub struct EmbedCache {
    pub h_struct: FxHashMap<NodeId, Tensor>,
    /// `(NodeId, example_idx) -> tensor`.
    pub h_value: FxHashMap<(NodeId, usize), Tensor>,
    /// `eval(node, input_i)` per example index. Caches eval results so
    /// `h_value` can take the case-2 fast path when possible.
    pub node_eval: FxHashMap<(NodeId, usize), Value>,
}

/// Compute `h_struct(N)` for a `NodeId`, recursively composing via
/// `app_net` over its child structures. Cached per `NodeId`.
pub fn h_struct(
    node: NodeId,
    arena: &Arena,
    leaves: &LeafTables,
    app_net: &AppNet,
    cache: &mut EmbedCache,
) -> Result<Tensor> {
    if let Some(t) = cache.h_struct.get(&node) {
        return Ok(t.clone());
    }
    let out = match arena.kind(node) {
        NodeKind::Literal(LitValue::Int(v)) => {
            set_literal_dim(&leaves.lit_int_emb, signed_log1p(*v as f64))?
        }
        NodeKind::Literal(LitValue::Bool(b)) => {
            set_literal_dim(&leaves.lit_bool_emb, if *b { 1.0 } else { 0.0 })?
        }
        NodeKind::Literal(LitValue::Float(f)) => {
            set_literal_dim(&leaves.lit_float_emb, signed_log1p(*f))?
        }
        NodeKind::Literal(LitValue::Char(c)) => {
            set_literal_dim(&leaves.lit_char_emb, (*c as u32) as f32 / 128.0)?
        }
        NodeKind::Param { .. } => leaves.param_emb.clone(),
        NodeKind::PrimRef(p) => leaves.prim_row(*p)?,
        NodeKind::App { func, arg } => {
            let hf = h_struct(*func, arena, leaves, app_net, cache)?;
            let ha = h_struct(*arg, arena, leaves, app_net, cache)?;
            app_net.compose(&hf, &ha)?
        }
        NodeKind::Lambda { body } => {
            // The search doesn't propose bare lambdas; treat as bottom
            // for embedding purposes.
            let _ = body;
            leaves.bottom_emb.clone()
        }
    };
    cache.h_struct.insert(node, out.clone());
    Ok(out)
}

/// Compute the value of `node` on `input` (single example) with caching
/// keyed by `(node, example_idx)`.
pub fn eval_at_example(
    node: NodeId,
    example_idx: usize,
    input: &Value,
    arena: &Arena,
    lib: &Library,
    fuel: u32,
    cache: &mut EmbedCache,
) -> Value {
    if let Some(v) = cache.node_eval.get(&(node, example_idx)) {
        return v.clone();
    }
    let env = [input.clone()];
    let mut f = fuel;
    let v = match eval(arena, lib, node, &env, &mut f) {
        Ok(v) => v,
        Err(_) => Value::bottom("eval err"),
    };
    cache.node_eval.insert((node, example_idx), v.clone());
    v
}

/// Compute `h_value(N, i)` — the per-example value embedding.
///
/// 1. If the node evaluates to a concrete value (Int/Bool/Float/Char/
///    List/Pair), `embed_value(value)`.
/// 2. Closure → walk the DAG via `app_net`, substituting `Param(0)` by
///    `embed_value(input)`.
/// 3. Bottom → `bottom_emb`.
pub fn h_value(
    node: NodeId,
    example_idx: usize,
    input: &Value,
    arena: &Arena,
    lib: &Library,
    leaves: &LeafTables,
    app_net: &AppNet,
    lp: ListPairIds,
    fuel: u32,
    cache: &mut EmbedCache,
) -> Result<Tensor> {
    if let Some(t) = cache.h_value.get(&(node, example_idx)) {
        return Ok(t.clone());
    }
    let v = eval_at_example(node, example_idx, input, arena, lib, fuel, cache);
    let out = match &v {
        Value::Closure(_) => {
            // Walk N's DAG via app_net, substituting Param(0) for the
            // input value. (`h_value_walk` is cached within this call.)
            let input_emb = embed_value(input, leaves, app_net, lp)?;
            h_value_walk(node, &input_emb, arena, leaves, app_net, cache)?
        }
        Value::Bottom(_) => leaves.bottom_emb.clone(),
        // Concrete value case.
        _ => embed_value(&v, leaves, app_net, lp)?,
    };
    cache.h_value.insert((node, example_idx), out.clone());
    Ok(out)
}

/// Structural walk that substitutes `Param(0)` by `input_emb`. Used by
/// `h_value` for closures. Walk-local cache to avoid recomputation
/// within one (node, input_emb) walk.
fn h_value_walk(
    node: NodeId,
    input_emb: &Tensor,
    arena: &Arena,
    leaves: &LeafTables,
    app_net: &AppNet,
    cache: &mut EmbedCache,
) -> Result<Tensor> {
    match arena.kind(node) {
        NodeKind::Literal(LitValue::Int(v)) => {
            set_literal_dim(&leaves.lit_int_emb, signed_log1p(*v as f64))
        }
        NodeKind::Literal(LitValue::Bool(b)) => {
            set_literal_dim(&leaves.lit_bool_emb, if *b { 1.0 } else { 0.0 })
        }
        NodeKind::Literal(LitValue::Float(f)) => {
            set_literal_dim(&leaves.lit_float_emb, signed_log1p(*f))
        }
        NodeKind::Literal(LitValue::Char(c)) => {
            set_literal_dim(&leaves.lit_char_emb, (*c as u32) as f32 / 128.0)
        }
        NodeKind::Param { .. } => Ok(input_emb.clone()),
        NodeKind::PrimRef(p) => leaves.prim_row(*p),
        NodeKind::App { func, arg } => {
            let hf = h_value_walk(*func, input_emb, arena, leaves, app_net, cache)?;
            let ha = h_value_walk(*arg, input_emb, arena, leaves, app_net, cache)?;
            app_net.compose(&hf, &ha)
        }
        NodeKind::Lambda { .. } => Ok(leaves.bottom_emb.clone()),
    }
}

/// Standalone helper: embed a value (the target). Same as `embed_value`.
pub fn h_target(
    value: &Value,
    leaves: &LeafTables,
    app_net: &AppNet,
    lp: ListPairIds,
) -> Result<Tensor> {
    embed_value(value, leaves, app_net, lp)
}

/// Stack a list of `(1, N)` tensors along dim 0, producing `(K, N)`.
pub fn stack_rows(rows: &[Tensor]) -> Result<Tensor> {
    let refs: Vec<&Tensor> = rows.iter().collect();
    Tensor::cat(&refs, 0)
}
