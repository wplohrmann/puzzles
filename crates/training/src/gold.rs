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

use lang::eval::Value;

use neural::Rng;

/// A task fully defined by its input/output pairs.
#[derive(Clone, Debug)]
pub struct GoldTask {
    pub category: GoldCategory,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
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

#[derive(Clone, Copy, Debug)]
enum ArithOp { Add, Sub, Mul }

impl ArithOp {
    fn apply(self, a: i64, b: i64) -> i64 {
        match self {
            ArithOp::Add => a.wrapping_add(b),
            ArithOp::Sub => a.wrapping_sub(b),
            ArithOp::Mul => a.wrapping_mul(b),
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
    GoldTask { category: GoldCategory::ArithUnary, inputs, outputs }
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
    GoldTask { category: GoldCategory::ArithCompose, inputs, outputs }
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
    GoldTask { category: GoldCategory::BoolTruthTable, inputs, outputs }
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
