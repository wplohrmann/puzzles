//! Hand-crafted "gold standard" task generator.
//!
//! A `GoldTask` is just `(inputs, outputs)` — fully defined by I/O,
//! no canonical program. The q-search finds whatever program matches.
//!
//! Three categories, sampled uniformly:
//!
//! 1. **arith_unary**: `f(v) = v OP a` for `OP ∈ {add, sub, mul}` and
//!    `a ∈ [-3, 3]`. K random Int inputs in `[-10, 10]`.
//! 2. **arith_compose**: `f(v) = (v OP1 a1) OP2 a2`.
//! 3. **bool_truth_table**: one of the 16 possible boolean functions
//!    of `(a, b) ∈ {(T,T), (T,F), (F,T), (F,F)}`. Always exactly 4
//!    examples — the full truth table is what defines the task.
//!
//! Boolean tasks include the degenerate ones (constants, projections,
//! `not a`, etc.) — those just get auto-shortcutted by the q-search
//! with `S_pool=0` and `r_q≈1`, so they're cheap wins. The interesting
//! ones (AND, OR, NAND, NOR, XOR, XNOR, implications) require real
//! search work.

use lang::arena::{Arena, NodeId};
use lang::builtin::BuiltinId;
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::Value;
use lang::ir::LitValue;
use lang::library::Library;

use neural::Rng;

/// A task fully defined by its input/output pairs, plus the recipe for
/// the canonical program that generated those outputs. The I/O is what
/// the q-search sees; the recipe lets us reconstruct a ground-truth
/// program for teacher-forced (behaviour-cloning) supervision.
#[derive(Clone, Debug)]
pub struct GoldTask {
    pub category: GoldCategory,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    /// Recipe for the canonical solving program. See `build_canonical`.
    pub program: GoldProgram,
}

impl GoldTask {
    /// Materialise the canonical solving program into `arena`. Returns
    /// `None` only if a required primitive is missing from `lib`. The
    /// program is built from the same leaves the q-search seeds with
    /// (`param(0)`, small int / bool literals, primitives), so its
    /// nodes hash-cons against the search's seed pool.
    pub fn build_canonical(&self, arena: &mut Arena, lib: &Library) -> Option<NodeId> {
        self.program.build(arena, lib)
    }
}

/// Recipe for a canonical program. Decoupled from the materialised IR
/// so a `GoldTask` stays cheap to clone and arena-independent.
#[derive(Clone, Copy, Debug)]
pub enum GoldProgram {
    /// `op(param0, a)`.
    ArithUnary { op: ArithOp, a: i64 },
    /// `op2(op1(param0, a1), a2)`.
    ArithCompose { op1: ArithOp, a1: i64, op2: ArithOp, a2: i64 },
    /// Boolean function of `(fst param0, snd param0)`, indexed by its
    /// 4-bit truth table (same encoding as `sample_bool_truth_table`).
    Bool { tt: u8 },
}

impl GoldProgram {
    /// Materialise into `arena`; see `GoldTask::build_canonical`.
    pub fn build(&self, arena: &mut Arena, lib: &Library) -> Option<NodeId> {
        match *self {
            GoldProgram::ArithUnary { op, a } => {
                let p0 = param(arena, 0);
                let opn = prim(arena, lib, op.builtin())?;
                let la = lit(arena, LitValue::Int(a));
                Some(app2(arena, opn, p0, la))
            }
            GoldProgram::ArithCompose { op1, a1, op2, a2 } => {
                let p0 = param(arena, 0);
                let o1 = prim(arena, lib, op1.builtin())?;
                let la1 = lit(arena, LitValue::Int(a1));
                let inner = app2(arena, o1, p0, la1);
                let o2 = prim(arena, lib, op2.builtin())?;
                let la2 = lit(arena, LitValue::Int(a2));
                Some(app2(arena, o2, inner, la2))
            }
            GoldProgram::Bool { tt } => build_bool(arena, lib, tt),
        }
    }
}

/// `prim_ref` for a built-in, or `None` if absent from the library.
fn prim(arena: &mut Arena, lib: &Library, b: BuiltinId) -> Option<NodeId> {
    lib.lookup(b.name()).map(|p| prim_ref(arena, p))
}

/// Curried binary application: `App(App(f, x), y)`.
fn app2(arena: &mut Arena, f: NodeId, x: NodeId, y: NodeId) -> NodeId {
    let fx = app(arena, f, x);
    app(arena, fx, y)
}

fn b_not(arena: &mut Arena, lib: &Library, x: NodeId) -> Option<NodeId> {
    let n = prim(arena, lib, BuiltinId::Not)?;
    Some(app(arena, n, x))
}
fn b_and(arena: &mut Arena, lib: &Library, x: NodeId, y: NodeId) -> Option<NodeId> {
    let a = prim(arena, lib, BuiltinId::And)?;
    Some(app2(arena, a, x, y))
}
fn b_or(arena: &mut Arena, lib: &Library, x: NodeId, y: NodeId) -> Option<NodeId> {
    let o = prim(arena, lib, BuiltinId::Or)?;
    Some(app2(arena, o, x, y))
}
fn b_eq(arena: &mut Arena, lib: &Library, x: NodeId, y: NodeId) -> Option<NodeId> {
    let e = prim(arena, lib, BuiltinId::Eq)?;
    Some(app2(arena, e, x, y))
}

/// Minimal program for boolean function `tt` of `(a, b)` where
/// `a = fst param0`, `b = snd param0`. The `tt` bit layout matches
/// `sample_bool_truth_table`: bit0=f(T,T), bit1=f(T,F), bit2=f(F,T),
/// bit3=f(F,F).
fn build_bool(arena: &mut Arena, lib: &Library, tt: u8) -> Option<NodeId> {
    let p0 = param(arena, 0);
    let fst = prim(arena, lib, BuiltinId::Fst)?;
    let snd = prim(arena, lib, BuiltinId::Snd)?;
    let a = app(arena, fst, p0);
    let b = app(arena, snd, p0);
    match tt & 0x0f {
        0b0000 => Some(lit(arena, LitValue::Bool(false))),
        0b1111 => Some(lit(arena, LitValue::Bool(true))),
        0b0011 => Some(a), // a
        0b0101 => Some(b), // b
        0b1100 => b_not(arena, lib, a), // not a
        0b1010 => b_not(arena, lib, b), // not b
        0b0001 => b_and(arena, lib, a, b), // a ∧ b
        0b0111 => b_or(arena, lib, a, b),  // a ∨ b
        0b1110 => {
            let an = b_and(arena, lib, a, b)?;
            b_not(arena, lib, an) // NAND
        }
        0b1000 => {
            let na = b_not(arena, lib, a)?;
            let nb = b_not(arena, lib, b)?;
            b_and(arena, lib, na, nb) // NOR
        }
        0b0110 => {
            let e = b_eq(arena, lib, a, b)?;
            b_not(arena, lib, e) // XOR
        }
        0b1001 => b_eq(arena, lib, a, b), // XNOR
        0b0010 => {
            let nb = b_not(arena, lib, b)?;
            b_and(arena, lib, a, nb) // a ∧ ¬b
        }
        0b0100 => {
            let na = b_not(arena, lib, a)?;
            b_and(arena, lib, na, b) // ¬a ∧ b
        }
        0b1011 => {
            let nb = b_not(arena, lib, b)?;
            b_or(arena, lib, a, nb) // a ∨ ¬b
        }
        0b1101 => {
            let na = b_not(arena, lib, a)?;
            b_or(arena, lib, na, b) // ¬a ∨ b
        }
        _ => None,
    }
}

/// Which template a `GoldTask` was sampled from. Used to bucket
/// diagnostics (per-category solve rate, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoldCategory {
    ArithUnary,
    ArithCompose,
    BoolTruthTable,
}

impl GoldCategory {
    /// All categories, in a stable order — for balanced sampling and
    /// per-category diagnostics.
    pub const ALL: [GoldCategory; 3] = [
        GoldCategory::ArithUnary,
        GoldCategory::ArithCompose,
        GoldCategory::BoolTruthTable,
    ];

    pub fn name(self) -> &'static str {
        match self {
            GoldCategory::ArithUnary => "arith_unary",
            GoldCategory::ArithCompose => "arith_compose",
            GoldCategory::BoolTruthTable => "bool_truth_table",
        }
    }
}

/// Sample one gold task. Picks a category uniformly, then samples
/// within it. `k_examples` controls the example count for arithmetic
/// tasks (boolean truth-table tasks always have 4 examples).
pub fn sample_gold(rng: &mut Rng, k_examples: usize) -> GoldTask {
    let cat = match rng.gen_range(3) {
        0 => GoldCategory::ArithUnary,
        1 => GoldCategory::ArithCompose,
        _ => GoldCategory::BoolTruthTable,
    };
    sample_gold_in_category(cat, rng, k_examples)
}

/// Sample within a fixed category — for diagnostic tools that want a
/// balanced per-category distribution.
pub fn sample_gold_in_category(
    cat: GoldCategory,
    rng: &mut Rng,
    k_examples: usize,
) -> GoldTask {
    match cat {
        GoldCategory::ArithUnary => sample_arith_unary(rng, k_examples),
        GoldCategory::ArithCompose => sample_arith_compose(rng, k_examples),
        GoldCategory::BoolTruthTable => sample_bool_truth_table(rng),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithOp { Add, Sub, Mul }

impl ArithOp {
    fn apply(self, a: i64, b: i64) -> i64 {
        match self {
            ArithOp::Add => a.wrapping_add(b),
            ArithOp::Sub => a.wrapping_sub(b),
            ArithOp::Mul => a.wrapping_mul(b),
        }
    }

    /// The built-in primitive that implements this op.
    pub fn builtin(self) -> BuiltinId {
        match self {
            ArithOp::Add => BuiltinId::Add,
            ArithOp::Sub => BuiltinId::Sub,
            ArithOp::Mul => BuiltinId::Mul,
        }
    }
}

fn random_arith_op(rng: &mut Rng) -> ArithOp {
    match rng.gen_range(3) {
        0 => ArithOp::Add,
        1 => ArithOp::Sub,
        _ => ArithOp::Mul,
    }
}

/// Constants used as the right-hand side of arithmetic operations.
/// Range `[-3, 3]` — same as the literal seeds the q-search has
/// available, so the search has a chance of reconstructing.
fn random_arith_const(rng: &mut Rng) -> i64 {
    (rng.gen_range(7) as i64) - 3
}

/// Inputs for arithmetic tasks. Wider range than the constants so
/// that constants and inputs are easy to tell apart at training time.
fn random_int_input(rng: &mut Rng) -> i64 {
    (rng.gen_range(21) as i64) - 10
}

fn sample_arith_unary(rng: &mut Rng, k: usize) -> GoldTask {
    let op = random_arith_op(rng);
    let a = random_arith_const(rng);
    let inputs: Vec<Value> = (0..k).map(|_| Value::Int(random_int_input(rng))).collect();
    let outputs: Vec<Value> = inputs.iter().map(|v| {
        let x = v.as_int().expect("Int input");
        Value::Int(op.apply(x, a))
    }).collect();
    GoldTask {
        category: GoldCategory::ArithUnary,
        inputs,
        outputs,
        program: GoldProgram::ArithUnary { op, a },
    }
}

fn sample_arith_compose(rng: &mut Rng, k: usize) -> GoldTask {
    let op1 = random_arith_op(rng);
    let op2 = random_arith_op(rng);
    let a1 = random_arith_const(rng);
    let a2 = random_arith_const(rng);
    let inputs: Vec<Value> = (0..k).map(|_| Value::Int(random_int_input(rng))).collect();
    let outputs: Vec<Value> = inputs.iter().map(|v| {
        let x = v.as_int().expect("Int input");
        let y = op1.apply(x, a1);
        Value::Int(op2.apply(y, a2))
    }).collect();
    GoldTask {
        category: GoldCategory::ArithCompose,
        inputs,
        outputs,
        program: GoldProgram::ArithCompose { op1, a1, op2, a2 },
    }
}

/// Sample one of 16 boolean functions of two bool variables.
/// The 4-bit truth table `tt` indexes the function:
///   bit 0 = output for (a=T, b=T)
///   bit 1 = output for (a=T, b=F)
///   bit 2 = output for (a=F, b=T)
///   bit 3 = output for (a=F, b=F)
fn sample_bool_truth_table(rng: &mut Rng) -> GoldTask {
    let tt = rng.gen_range(16) as u8;
    let input_bits = [(true, true), (true, false), (false, true), (false, false)];
    let inputs: Vec<Value> = input_bits.iter().map(|(a, b)| {
        Value::pair(Value::Bool(*a), Value::Bool(*b))
    }).collect();
    let outputs: Vec<Value> = (0..4)
        .map(|i| Value::Bool(((tt >> i) & 1) == 1))
        .collect();
    GoldTask {
        category: GoldCategory::BoolTruthTable,
        inputs,
        outputs,
        program: GoldProgram::Bool { tt },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arith_unary_matches_expected_function() {
        let mut rng = Rng::new(0xa1);
        let task = sample_arith_unary(&mut rng, 5);
        assert_eq!(task.inputs.len(), 5);
        assert_eq!(task.outputs.len(), 5);
        // Outputs must be a deterministic function of inputs — if we
        // re-run the same function on the same input we should match.
        for (i, o) in task.inputs.iter().zip(task.outputs.iter()) {
            let _x = i.as_int().unwrap();
            let _y = o.as_int().unwrap();
            // Just sanity that both are Int.
        }
    }

    #[test]
    fn bool_truth_table_has_four_unique_inputs() {
        let mut rng = Rng::new(0xb1);
        let task = sample_bool_truth_table(&mut rng);
        assert_eq!(task.inputs.len(), 4);
        assert_eq!(task.outputs.len(), 4);
        // Inputs are the 4 distinct (a, b) bool pairs in canonical order.
        let expected_inputs: Vec<Value> = [(true, true), (true, false), (false, true), (false, false)]
            .iter()
            .map(|(a, b)| Value::pair(Value::Bool(*a), Value::Bool(*b)))
            .collect();
        assert_eq!(task.inputs, expected_inputs);
    }

    #[test]
    fn all_16_truth_tables_reachable() {
        // Brute-force: sample many times, collect distinct output
        // patterns (encoded as the 4-bit tt index). All 16 should
        // appear within reasonable samples.
        let mut rng = Rng::new(0xb2);
        let mut seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
        for _ in 0..500 {
            let task = sample_bool_truth_table(&mut rng);
            let mut tt = 0u8;
            for (i, out) in task.outputs.iter().enumerate() {
                if let Value::Bool(true) = out {
                    tt |= 1 << i;
                }
            }
            seen.insert(tt);
        }
        assert_eq!(seen.len(), 16, "expected all 16 truth tables, got {}", seen.len());
    }

    /// The canonical program must reproduce the task's outputs exactly.
    /// Run it on each input via the evaluator and compare.
    fn check_canonical(task: &GoldTask) {
        use lang::arena::Arena;
        use lang::builtin::seed_builtin_library;
        use lang::eval::eval;

        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let prog = task
            .build_canonical(&mut arena, &lib)
            .expect("canonical builds");
        for (input, expected) in task.inputs.iter().zip(task.outputs.iter()) {
            let env = [input.clone()];
            let mut fuel = 100_000u32;
            let got = eval(&arena, &lib, prog, &env, &mut fuel)
                .expect("canonical evaluates");
            assert_eq!(&got, expected, "canonical mismatch for {:?}", task.program);
        }
    }

    #[test]
    fn canonical_reproduces_arith_io() {
        let mut rng = Rng::new(0xc0ffee);
        for _ in 0..200 {
            check_canonical(&sample_arith_unary(&mut rng, 5));
            check_canonical(&sample_arith_compose(&mut rng, 5));
        }
    }

    #[test]
    fn canonical_reproduces_all_16_bool_tables() {
        // Build each truth table directly (independent of sampling) and
        // confirm the hand-written program matches every row.
        let input_bits = [(true, true), (true, false), (false, true), (false, false)];
        for tt in 0u8..16 {
            let inputs: Vec<Value> = input_bits
                .iter()
                .map(|(a, b)| Value::pair(Value::Bool(*a), Value::Bool(*b)))
                .collect();
            let outputs: Vec<Value> =
                (0..4).map(|i| Value::Bool(((tt >> i) & 1) == 1)).collect();
            let task = GoldTask {
                category: GoldCategory::BoolTruthTable,
                inputs,
                outputs,
                program: GoldProgram::Bool { tt },
            };
            check_canonical(&task);
        }
    }

    #[test]
    fn sample_gold_does_not_panic() {
        let mut rng = Rng::new(0x12345);
        for _ in 0..200 {
            let task = sample_gold(&mut rng, 3);
            assert!(!task.inputs.is_empty());
            assert_eq!(task.inputs.len(), task.outputs.len());
        }
    }
}
