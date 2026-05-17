//! Hand-coded feature extraction over `Value`s.
//!
//! For each (value, target) pair we produce a fixed-width float vector.
//! These features are the "are we close?" signal; the policy / value MLPs
//! consume aggregates of them. See `decisions/03-value-only-features.md`.

use lang::eval::{ClosureHead, Value};

/// Number of slots reserved for one-hot encoding of a closure's head
/// primitive. Sized to accommodate the seed library (19 builtins) with a
/// little slack for learned primitives added in M5.
pub const PRIM_ONEHOT_SLOTS: usize = 32;

/// Width of `value_pair_features` output. Fixed at compile time so the
/// MLPs can be shaped statically.
///
/// Layout (offset → meaning):
///   0..8           type one-hot of value
///   8              same-variant-as-target?
///   9              deep eq?
///   10             value is_bottom
///   11..15         numeric / list overlay (Int / Float / Bool / List)
///   15..24         list-specific scalars
///   24..26         closure (args saturated, arity remaining)
///   26..58         closure prim_id one-hot (PRIM_ONEHOT_SLOTS slots)
pub const PAIR_FEAT_DIM: usize = 26 + PRIM_ONEHOT_SLOTS;

/// Width of `value_features` output (a single value, no target).
pub const SOLO_FEAT_DIM: usize = 12;

/// Encode one (value, target) pair into `PAIR_FEAT_DIM` floats. The
/// features are designed for list/scalar tasks; they degrade gracefully
/// (zeros) for types they don't understand.
pub fn value_pair_features(value: &Value, target: &Value) -> [f32; PAIR_FEAT_DIM] {
    let mut f = [0.0f32; PAIR_FEAT_DIM];

    // 0..7: type indicator one-hot for `value`.
    f[type_index(value)] = 1.0;
    f[8] = if same_variant(value, target) { 1.0 } else { 0.0 };
    f[9] = if deep_eq(value, target) { 1.0 } else { 0.0 };
    f[10] = if value.is_bottom() { 1.0 } else { 0.0 };

    // Numeric overlay (Int / Float).
    if let (Some(v), Some(t)) = (numeric(value), numeric(target)) {
        f[11] = signed_log(v - t);
        f[12] = if v == t { 1.0 } else { 0.0 };
        f[13] = signed_log(v);
        f[14] = signed_log(t);
    }

    // List features.
    if let (Value::List(vs), Value::List(ts)) = (value, target) {
        let v_len = vs.len() as f32;
        let t_len = ts.len() as f32;
        f[15] = (v_len + 1.0).ln();
        f[16] = (t_len + 1.0).ln();
        f[17] = if vs.len() == ts.len() { 1.0 } else { 0.0 };
        f[18] = signed_log(v_len - t_len);
        let n_min = vs.len().min(ts.len()).max(1);
        let mut prefix_eq = 0;
        for (a, b) in vs.iter().zip(ts.iter()) {
            if deep_eq(a, b) { prefix_eq += 1; } else { break; }
        }
        f[19] = prefix_eq as f32 / n_min as f32;
        let mut suffix_eq = 0;
        for (a, b) in vs.iter().rev().zip(ts.iter().rev()) {
            if deep_eq(a, b) { suffix_eq += 1; } else { break; }
        }
        f[20] = suffix_eq as f32 / n_min as f32;
        let mut elem_eq = 0;
        for (a, b) in vs.iter().zip(ts.iter()) {
            if deep_eq(a, b) { elem_eq += 1; }
        }
        f[21] = elem_eq as f32 / n_min as f32;
        f[22] = if is_contiguous_subseq(vs, ts) { 1.0 } else { 0.0 };
        f[23] = if is_contiguous_subseq(ts, vs) { 1.0 } else { 0.0 };
    }

    // Bool overlay.
    if let (Value::Bool(b), Value::Bool(t)) = (value, target) {
        f[11] = if *b { 1.0 } else { -1.0 };
        f[12] = if b == t { 1.0 } else { 0.0 };
    }

    // Closure overlay (24..58): args saturated, arity remaining, and
    // a one-hot for the head primitive (Lambda heads get the all-zero
    // tail and only the closure-shape features). This is the crucial
    // signal distinguishing `fold` from `add` from `cons` in the pool.
    if let Value::Closure(c) = value {
        f[24] = c.args.len() as f32;
        f[25] = (c.arity as f32) - (c.args.len() as f32);
        if let ClosureHead::Prim(pid) = c.head {
            let slot = (pid as usize) % PRIM_ONEHOT_SLOTS;
            f[26 + slot] = 1.0;
        }
    }

    f
}

/// Solo features for a Value (no target). Used for task encoding (target-
/// only). The task is List/Int-flavoured in this milestone — closure
/// detail isn't needed, since closures are pool intermediates, not task
/// targets.
pub fn value_features(value: &Value) -> [f32; SOLO_FEAT_DIM] {
    let mut f = [0.0f32; SOLO_FEAT_DIM];
    f[type_index(value)] = 1.0;          // 0..7
    f[8] = if value.is_bottom() { 1.0 } else { 0.0 };
    if let Some(v) = numeric(value) {
        f[9] = signed_log(v);
    }
    if let Value::List(xs) = value {
        f[10] = (xs.len() as f32 + 1.0).ln();
        let mut sum = 0.0;
        let mut n = 0;
        for x in xs.iter() {
            if let Some(v) = numeric(x) { sum += v; n += 1; }
        }
        f[11] = if n > 0 { signed_log(sum / n as f32) } else { 0.0 };
    }
    f
}

// -- helpers -------------------------------------------------------------

fn type_index(v: &Value) -> usize {
    match v {
        Value::Int(_) => 0,
        Value::Bool(_) => 1,
        Value::Float(_) => 2,
        Value::Char(_) => 3,
        Value::List(_) => 4,
        Value::Pair(_) => 5,
        Value::Closure(_) => 6,
        Value::Bottom(_) => 7,
    }
}

fn same_variant(a: &Value, b: &Value) -> bool {
    type_index(a) == type_index(b)
}

fn numeric(v: &Value) -> Option<f32> {
    match v {
        Value::Int(i) => Some(*i as f32),
        Value::Float(f) => Some(*f as f32),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn signed_log(x: f32) -> f32 {
    x.signum() * (1.0 + x.abs()).ln()
}

fn deep_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::List(xs), Value::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(p, q)| deep_eq(p, q))
        }
        (Value::Pair(p), Value::Pair(q)) => deep_eq(&p.0, &q.0) && deep_eq(&p.1, &q.1),
        _ => false,
    }
}

fn is_contiguous_subseq(needle: &[Value], hay: &[Value]) -> bool {
    if needle.is_empty() { return true; }
    if needle.len() > hay.len() { return false; }
    'outer: for start in 0..=hay.len() - needle.len() {
        for k in 0..needle.len() {
            if !deep_eq(&needle[k], &hay[start + k]) { continue 'outer; }
        }
        return true;
    }
    false
}

/// Aggregate per-example features into a single fixed-size vector via
/// concat(mean, max, min). Output dim = 3 * input dim.
pub fn aggregate_examples(per_example: &[Vec<f32>]) -> Vec<f32> {
    if per_example.is_empty() {
        return Vec::new();
    }
    let dim = per_example[0].len();
    let mut mean = vec![0.0; dim];
    let mut max = vec![f32::NEG_INFINITY; dim];
    let mut min = vec![f32::INFINITY; dim];
    let n = per_example.len() as f32;
    for row in per_example {
        debug_assert_eq!(row.len(), dim);
        for k in 0..dim {
            mean[k] += row[k] / n;
            if row[k] > max[k] { max[k] = row[k]; }
            if row[k] < min[k] { min[k] = row[k]; }
        }
    }
    if per_example.len() == 1 {
        // avoid -inf/+inf in the output
        for k in 0..dim {
            if !max[k].is_finite() { max[k] = mean[k]; }
            if !min[k].is_finite() { min[k] = mean[k]; }
        }
    }
    let mut out = Vec::with_capacity(3 * dim);
    out.extend(mean.iter().copied());
    out.extend(max.iter().copied());
    out.extend(min.iter().copied());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn list(xs: Vec<i64>) -> Value {
        Value::List(Rc::new(xs.into_iter().map(Value::Int).collect()))
    }

    #[test]
    fn equal_values_score_high_on_eq() {
        let v = list(vec![1, 2, 3]);
        let t = list(vec![1, 2, 3]);
        let f = value_pair_features(&v, &t);
        assert_eq!(f[9], 1.0); // deep eq
        assert_eq!(f[17], 1.0); // length match
        assert_eq!(f[21], 1.0); // elementwise eq
    }

    #[test]
    fn different_lengths_have_low_match() {
        let v = list(vec![1, 2]);
        let t = list(vec![1, 2, 3]);
        let f = value_pair_features(&v, &t);
        assert_eq!(f[9], 0.0);
        assert_eq!(f[17], 0.0);
        assert!(f[19] > 0.5); // prefix matches 2/2
    }

    #[test]
    fn bottom_marker_set() {
        let v = Value::bottom("nope");
        let t = Value::Int(0);
        let f = value_pair_features(&v, &t);
        assert_eq!(f[10], 1.0);
    }
}
