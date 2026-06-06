//! Dream sampler: PCFG-driven random programs evaluated against random
//! inputs to produce synthetic supervised tasks.
//!
//! See `decisions/05-curriculum.md` and `decisions/06-pcfg-prior.md`.

use std::rc::Rc;

use lang::arena::{Arena, NodeId};
use lang::builtin::BuiltinId;
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{eval, Value};
use lang::ir::LitValue;
use lang::library::Library;

use neural::Rng;

/// One synthetic task produced by sampling a program from the prior.
#[derive(Clone, Debug)]
pub struct DreamTask {
    pub program: NodeId,
    pub examples: Vec<(Value, Value)>,
}

/// PCFG hyper-parameters for sampling. Defaults are eyeballed from the
/// trivial-list bench programs (sum, head, length, add-one-to-each).
#[derive(Clone, Debug)]
pub struct DreamCfg {
    pub max_size: u32,
    /// Probability of forming an `App` at each non-leaf step.
    pub p_app: f32,
    /// Number of (input, output) examples per dream.
    pub examples_per_dream: usize,
    /// Max retries finding a non-Bottom output set before giving up on
    /// the sampled program.
    pub max_resample: usize,
    /// Fuel for evaluating sampled programs.
    pub fuel: u32,
    /// `examples_per_dream` * this fraction of outputs may Bottom; above
    /// that we resample. Some bottoms are fine — they teach the network
    /// that the candidate is bad.
    pub max_bottom_frac: f32,
    /// Probability that a sampled dream uses a structural template
    /// (`((fold B) Z) p0`-shaped) instead of a flat PCFG sample. See
    /// `decisions/10-template-dreams.md`. The bodies/accumulators
    /// inside the template are sampled by the PCFG so the network sees
    /// many fold-shape examples without us hand-writing the bench
    /// programs.
    pub p_template: f32,
}

impl Default for DreamCfg {
    fn default() -> Self {
        Self {
            max_size: 7,
            p_app: 0.85,
            examples_per_dream: 4,
            max_resample: 200,
            fuel: 200_000,
            max_bottom_frac: 0.5,
            p_template: 0.7,
        }
    }
}

/// Sample a random program. The DAG-size of the result is bounded by
/// `size_budget`; hash-cons sharing may make it smaller.
///
/// `depth` is the current AST depth from the root. At depth 0 we sample
/// the program's root and we bias toward left-leaning App-chains
/// terminating in `param(0)` — the standard shape for list-mapping
/// programs (`λxs. f xs`), shared by every bench task.
pub fn sample_program(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    size_budget: u32,
    p_app: f32,
) -> NodeId {
    sample_program_inner(arena, lib, rng, size_budget, p_app, 0)
}

fn sample_program_inner(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    size_budget: u32,
    p_app: f32,
    depth: u32,
) -> NodeId {
    if size_budget <= 1 {
        return sample_leaf(arena, lib, rng, depth);
    }
    if rng.next_f32() < p_app && size_budget >= 3 {
        let total = size_budget - 1;
        // Bias the split: prefer the function side to be larger (the
        // bench programs all left-lean — `App(App(App(fold, x), y), p0)`
        // grows the function position deepest while the arg position
        // stays shallow). Specifically, sample `k_a` from a half-decay
        // distribution so the arg is usually small.
        let k_a = sample_small_split(rng, total);
        let k_f = total - k_a;
        let f = sample_program_inner(arena, lib, rng, k_f, p_app, depth + 1);
        // The rightmost (deepest) arg of the root chain is overwhelmingly
        // `param(0)` in the bench programs. Force it when we're at the
        // outermost App's arg position (depth 0, k_a == 1).
        let a = if depth == 0 && k_a == 1 {
            param(arena, 0)
        } else {
            sample_program_inner(arena, lib, rng, k_a, p_app, depth + 1)
        };
        app(arena, f, a)
    } else {
        sample_leaf(arena, lib, rng, depth)
    }
}

/// Sample `k_a ∈ [1, total-1]` with a geometric-style decay (smaller k_a
/// more likely). For total=2, returns 1.
fn sample_small_split(rng: &mut Rng, total: u32) -> u32 {
    if total <= 2 { return 1; }
    let r = rng.next_f32();
    // Skewed: k_a=1 with prob 0.55, =2 with 0.2, geometric tail.
    let mut k_a = 1u32;
    let mut cum = 0.55f32;
    while r > cum && k_a < total - 1 {
        k_a += 1;
        cum += 0.5 * (1.0 - cum);
    }
    k_a
}

/// Sample a leaf: param(0), a small int literal, or a primref. Weights
/// biased toward primitives that feature in the bench programs. At
/// shallow depths (closer to the root) we *de-emphasize* `param(0)` —
/// param(0) almost always lives at the deepest arg position, not at a
/// function position.
pub fn sample_leaf(arena: &mut Arena, lib: &Library, rng: &mut Rng, depth: u32) -> NodeId {
    let weights = leaf_weights(lib, depth);
    let total: u32 = weights.iter().map(|(_, w)| *w).sum();
    let mut pick = rng.gen_range(total as usize) as u32;
    for (kind, w) in &weights {
        if pick < *w {
            return materialise_leaf(arena, lib, rng, *kind);
        }
        pick -= w;
    }
    unreachable!("weighted pick fell through");
}

#[derive(Clone, Copy, Debug)]
enum LeafKind {
    Param0,
    SmallInt,
    Prim(BuiltinId),
}

fn leaf_weights(lib: &Library, depth: u32) -> Vec<(LeafKind, u32)> {
    // `param(0)` belongs at the deepest right-arg position, not at the
    // function position. Down-weight param(0) at shallow depth.
    let param0_weight = if depth == 0 { 0 } else { 3 };
    let mut out = vec![
        (LeafKind::Param0, param0_weight),
        (LeafKind::SmallInt, 3),
    ];
    // The bench programs use `fold`, `cons`, `add`, `b`, `k`, `nil`,
    // and small ints. Bias toward those.
    let high = [
        BuiltinId::Fold, BuiltinId::Cons, BuiltinId::Add, BuiltinId::K,
        BuiltinId::B, BuiltinId::Nil,
    ];
    let mid = [
        BuiltinId::If, BuiltinId::Eq, BuiltinId::Lt, BuiltinId::Not,
        BuiltinId::Sub, BuiltinId::Mul,
    ];
    for builtin in [
        BuiltinId::Add, BuiltinId::Sub, BuiltinId::Mul, BuiltinId::Div,
        BuiltinId::Lt, BuiltinId::Eq, BuiltinId::Not, BuiltinId::And, BuiltinId::Or,
        BuiltinId::If, BuiltinId::Pair, BuiltinId::Fst, BuiltinId::Snd,
        BuiltinId::Nil, BuiltinId::Cons, BuiltinId::Fold, BuiltinId::Unfold,
        BuiltinId::K, BuiltinId::B,
    ] {
        if lib.lookup(builtin.name()).is_none() { continue; }
        let w = if high.contains(&builtin) {
            5
        } else if mid.contains(&builtin) {
            2
        } else {
            1
        };
        out.push((LeafKind::Prim(builtin), w));
    }
    out
}

fn materialise_leaf(arena: &mut Arena, lib: &Library, rng: &mut Rng, kind: LeafKind) -> NodeId {
    match kind {
        LeafKind::Param0 => param(arena, 0),
        LeafKind::SmallInt => {
            let v = (rng.gen_range(7) as i64) - 3; // [-3, 3]
            lit(arena, LitValue::Int(v))
        }
        LeafKind::Prim(b) => {
            let pid = lib.lookup(b.name()).expect("builtin in seed library");
            prim_ref(arena, pid)
        }
    }
}

/// Sample one input drawn from a list-of-int distribution. Length 0–8,
/// elements uniform in [-5, 5].
pub fn sample_input_list_int(rng: &mut Rng) -> Value {
    let len = rng.gen_range(9);
    let mut xs = Vec::with_capacity(len);
    for _ in 0..len {
        let i = (rng.gen_range(11) as i64) - 5;
        xs.push(Value::Int(i));
    }
    Value::List(Rc::new(xs))
}

/// Per-dream input type. Selected once, then used across all examples.
#[derive(Clone, Copy, Debug)]
pub enum InputKind {
    ListInt,
    Int,
    Bool,
    PairIntInt,
}

pub fn sample_input_kind(rng: &mut Rng) -> InputKind {
    match rng.gen_range(4) {
        0 => InputKind::ListInt,
        1 => InputKind::Int,
        2 => InputKind::Bool,
        _ => InputKind::PairIntInt,
    }
}

pub fn sample_input_of_kind(rng: &mut Rng, kind: InputKind) -> Value {
    match kind {
        InputKind::ListInt => sample_input_list_int(rng),
        InputKind::Int => Value::Int((rng.gen_range(11) as i64) - 5),
        InputKind::Bool => Value::Bool(rng.next_f32() < 0.5),
        InputKind::PairIntInt => Value::pair(
            Value::Int((rng.gen_range(11) as i64) - 5),
            Value::Int((rng.gen_range(11) as i64) - 5),
        ),
    }
}

/// Construct a synthetic task by sampling a program and a list of
/// inputs, running the program, and recording (input, output) pairs.
/// Returns `None` if too many outputs Bottom (the program is malformed
/// for our chosen input distribution).
pub fn try_make_dream(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    cfg: &DreamCfg,
) -> Option<DreamTask> {
    // Roll: structural template vs flat PCFG.
    let use_template = rng.next_f32() < cfg.p_template && cfg.max_size >= 7;
    let program = if use_template {
        match sample_fold_template(arena, lib, rng, cfg) {
            Some(p) => p,
            None => return None,
        }
    } else {
        sample_flat(arena, lib, rng, cfg)
    };

    if matches!(arena.kind(program), lang::ir::NodeKind::Param { .. }
            | lang::ir::NodeKind::Literal(_)
            | lang::ir::NodeKind::PrimRef(_))
    {
        return None;
    }
    let _ = use_template;

    // If the dream uses a fold-template, list inputs are the only thing
    // that yields a non-Bottom output. Otherwise pick a random per-dream
    // input kind so small programs that operate on Bool / Int / Pair
    // can survive.
    let input_kind = if use_template {
        InputKind::ListInt
    } else {
        sample_input_kind(rng)
    };
    let mut examples: Vec<(Value, Value)> = Vec::with_capacity(cfg.examples_per_dream);
    let mut bottom_count = 0;
    for _ in 0..cfg.examples_per_dream {
        let input = sample_input_of_kind(rng, input_kind);
        let mut fuel = cfg.fuel;
        let env = [input.clone()];
        let v = match eval(arena, lib, program, &env, &mut fuel) {
            Ok(v) => v,
            Err(_) => return None,
        };
        if v.is_bottom() { bottom_count += 1; }
        examples.push((input, v));
    }
    let max_bot = (cfg.examples_per_dream as f32 * cfg.max_bottom_frac).ceil() as usize;
    if bottom_count > max_bot { return None; }
    // Reject examples whose output is Bottom or contains a Closure anywhere
    // in its List/Pair tree. Value equality has no (Closure, Closure) arm
    // and any closure compares not-equal to anything, so even a List or
    // Pair containing a closure is permanently unsolvable by value-equality.
    examples.retain(|(_, v)| !v.is_bottom() && !contains_closure(v));
    if examples.len() < cfg.examples_per_dream {
        return None;
    }
    // Discard "trivial" dreams whose output is constant across all
    // examples (the search would solve these via a literal seed). The
    // network gets nothing from them.
    if examples.iter().all(|(_, v)| v == &examples[0].1) {
        return None;
    }
    Some(DreamTask { program, examples })
}

/// True if `v` is a `Closure`, or contains one anywhere in a List/Pair
/// tree. Used to filter dreams whose expected outputs would be
/// permanently unsolvable by value-equality.
fn contains_closure(v: &Value) -> bool {
    match v {
        Value::Closure(_) => true,
        Value::List(xs) => xs.iter().any(contains_closure),
        Value::Pair(p) => contains_closure(&p.0) || contains_closure(&p.1),
        _ => false,
    }
}

/// Sample a flat PCFG-style program, size budget concentrated near
/// `cfg.max_size`.
fn sample_flat(arena: &mut Arena, lib: &Library, rng: &mut Rng, cfg: &DreamCfg) -> NodeId {
    let lo = (cfg.max_size as i32 - 2).max(2) as u32;
    let hi = cfg.max_size.max(lo);
    let s = if lo >= hi {
        hi
    } else {
        lo + rng.gen_range((hi - lo) as usize + 1) as u32
    };
    sample_program(arena, lib, rng, s, cfg.p_app)
}

/// Build a fold-template program: `App(App(App(fold, B), Z), param(0))`
/// where B and Z are sampled by the PCFG. The constructed DAG matches
/// the canonical shape of sum / head / length / add-one-to-each from
/// the bench (and most other useful list-mapping programs).
///
/// Returns `None` if budgets don't fit a fold scaffold (size budget
/// has to leave room for `fold`, `Z`, and `param(0)`).
fn sample_fold_template(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    cfg: &DreamCfg,
) -> Option<NodeId> {
    // Total size = size(fold) + size(B) + 1 + size(Z) + 1 + size(p0) + 1
    //            = 1 + size(B) + 1 + size(Z) + 1 + 1 + 1
    //            = size(B) + size(Z) + 5
    // For max_size=13: size(B) + size(Z) ≤ 8. Practical: B ∈ [1, 7], Z ∈ [1, 3].
    if cfg.max_size < 7 { return None; }
    let lo = (cfg.max_size as i32 - 2).max(7) as u32;
    let hi = cfg.max_size.max(lo);
    let total = if lo >= hi {
        hi
    } else {
        lo + rng.gen_range((hi - lo) as usize + 1) as u32
    };
    let remaining = total.saturating_sub(5); // for B + Z
    if remaining < 2 { return None; }
    // Split remaining between B (function body) and Z (initial accumulator).
    // Bias toward bigger B since the interesting programs have small Z (0 or nil).
    let z_size: u32 = if rng.next_f32() < 0.7 { 1 } else { 1 + rng.gen_range(2) as u32 };
    let z_size = z_size.min(remaining - 1);
    let b_size = remaining - z_size;
    let b_size = b_size.max(1);

    let body = sample_program(arena, lib, rena_capacity(rng), b_size, cfg.p_app);
    let acc = if z_size == 1 {
        sample_acc_leaf(arena, lib, rng)
    } else {
        sample_program(arena, lib, rena_capacity(rng), z_size, cfg.p_app)
    };
    let p0 = param(arena, 0);
    let fold_id = lib.lookup("fold")?;
    let fold_ref = prim_ref(arena, fold_id);
    let f1 = app(arena, fold_ref, body);
    let f2 = app(arena, f1, acc);
    Some(app(arena, f2, p0))
}

/// `rng` re-borrow helper so the call sites stay terse.
fn rena_capacity<'a>(rng: &'a mut Rng) -> &'a mut Rng { rng }

/// Leaf sampler biased toward typical accumulators in the bench (0, nil,
/// 1, small ints).
fn sample_acc_leaf(arena: &mut Arena, lib: &Library, rng: &mut Rng) -> NodeId {
    let r = rng.next_f32();
    if r < 0.45 {
        lit(arena, LitValue::Int(0))
    } else if r < 0.80 {
        let nil_id = lib.lookup("nil").expect("nil in seed lib");
        prim_ref(arena, nil_id)
    } else if r < 0.92 {
        lit(arena, LitValue::Int(1))
    } else {
        let v = (rng.gen_range(7) as i64) - 3;
        lit(arena, LitValue::Int(v))
    }
}

/// Repeatedly try to sample a dream; returns the first one that passes
/// the filters. Gives up after `cfg.max_resample` attempts and returns
/// `None`.
pub fn sample_dream(
    arena: &mut Arena,
    lib: &Library,
    rng: &mut Rng,
    cfg: &DreamCfg,
) -> Option<DreamTask> {
    for _ in 0..cfg.max_resample {
        if let Some(d) = try_make_dream(arena, lib, rng, cfg) {
            return Some(d);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang::builtin::seed_builtin_library;

    #[test]
    fn dream_eventually_samples() {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let mut rng = Rng::new(0xc0ffee);
        let cfg = DreamCfg::default();
        let mut got = 0;
        for _ in 0..200 {
            if sample_dream(&mut arena, &lib, &mut rng, &cfg).is_some() {
                got += 1;
            }
        }
        // At default settings, well over half of attempts should yield a
        // dream. A weak lower bound to keep this test cheap and stable.
        assert!(got > 30, "got only {} dreams in 200 tries", got);
    }
}
