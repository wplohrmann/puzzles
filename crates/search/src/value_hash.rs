//! Stable structural hash for `Value`. Used as part of the
//! observational-equivalence dedup key.
//!
//! Closures hash to a single discriminant byte — meaning two function-
//! typed values look "equal" to this hash, which would falsely dedup
//! distinct functions. Callers must skip obs-eq entirely for entries
//! whose stored type is a function (see `pool::should_obs_eq`).

use std::hash::Hasher;

use lang::eval::Value;

pub fn hash_value<H: Hasher>(v: &Value, h: &mut H) {
    use std::hash::Hash;
    match v {
        Value::Int(i) => { 0u8.hash(h); i.hash(h); }
        Value::Bool(b) => { 1u8.hash(h); b.hash(h); }
        Value::Float(f) => { 2u8.hash(h); f.to_bits().hash(h); }
        Value::Char(c) => { 3u8.hash(h); c.hash(h); }
        Value::List(xs) => {
            4u8.hash(h);
            (xs.len() as u32).hash(h);
            for x in xs.iter() {
                hash_value(x, h);
            }
        }
        Value::Pair(p) => {
            5u8.hash(h);
            hash_value(&p.0, h);
            hash_value(&p.1, h);
        }
        Value::Closure(_) => {
            // Closures are intentionally not distinguished here — the pool
            // skips obs-eq for function-typed entries.
            6u8.hash(h);
        }
        Value::Bottom(_) => {
            // All bottoms hash equal — different bottom messages don't
            // distinguish behaviour.
            7u8.hash(h);
        }
    }
}

pub fn hash_values<H: Hasher>(vs: &[Value], h: &mut H) {
    use std::hash::Hash;
    (vs.len() as u32).hash(h);
    for v in vs {
        hash_value(v, h);
    }
}
