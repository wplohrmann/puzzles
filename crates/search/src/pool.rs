//! The pool: every candidate node the search has materialised so far,
//! indexed by NodeId, by program size (for split-by-size action
//! enumeration), and by value-tuple-hash (for observational-equivalence
//! dedup).
//!
//! Without a static type system, obs-eq is a pure function of the
//! values: two candidates are equivalent if they produce the same
//! tuple of values across every example. Closures don't compare
//! structurally — `value_obs_key` returns `None` for any tuple
//! containing a closure, and the caller decides whether to skip dedup
//! or fall back to the probe-based key.

use std::hash::Hasher;

use rustc_hash::{FxHashMap, FxHasher};

use lang::arena::NodeId;
use lang::eval::{apply, Value};
use lang::library::Library;

use crate::value_hash::hash_values;

#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub node: NodeId,
    pub size: u32,
    pub values: Vec<Value>,
}

#[derive(Default)]
pub(crate) struct Pool {
    pub entries: Vec<Entry>,
    pub by_node: FxHashMap<NodeId, usize>,
    pub by_obs: FxHashMap<u64, usize>,
    pub by_size: Vec<Vec<usize>>,
}

impl Pool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn contains(&self, node: NodeId) -> bool {
        self.by_node.contains_key(&node)
    }

    pub fn entries_with_size(&self, size: u32) -> &[usize] {
        let i = size as usize;
        if i < self.by_size.len() {
            &self.by_size[i]
        } else {
            &[]
        }
    }

    pub fn entry(&self, idx: usize) -> &Entry {
        &self.entries[idx]
    }

    pub fn try_add(
        &mut self,
        node: NodeId,
        size: u32,
        values: Vec<Value>,
        obs_key: Option<u64>,
    ) -> AddOutcome {
        if self.by_node.contains_key(&node) {
            return AddOutcome::DuplicateNode;
        }
        if let Some(key) = obs_key {
            if self.by_obs.contains_key(&key) {
                return AddOutcome::ObsEqPruned;
            }
            self.by_obs.insert(key, self.entries.len());
        }
        let idx = self.entries.len();
        self.entries.push(Entry { node, size, values });
        self.by_node.insert(node, idx);
        let i = size as usize;
        while self.by_size.len() <= i {
            self.by_size.push(Vec::new());
        }
        self.by_size[i].push(idx);
        AddOutcome::Added
    }
}

#[derive(Debug)]
pub(crate) enum AddOutcome {
    Added,
    DuplicateNode,
    ObsEqPruned,
}

/// Compute a value-tuple obs-eq key. Returns `None` if any of the
/// values is a `Closure` — closure values aren't comparable, so
/// callers must either fall back to the probe-based key or skip dedup.
pub(crate) fn value_obs_key(values: &[Value]) -> Option<u64> {
    if values.iter().any(|v| matches!(v, Value::Closure(_))) {
        return None;
    }
    let mut h = FxHasher::default();
    hash_values(values, &mut h);
    Some(h.finish())
}

/// Probe-based obs-eq key for closure-valued candidates: apply each
/// closure value to its corresponding example input and hash the
/// resulting values.
///
/// Returns `None` (skip dedup) if:
/// - any value isn't a closure;
/// - any apply yields another closure (probe didn't reach a concrete
///   value, the closure has more args to consume);
/// - any apply yields `Bottom` (the closure can't be applied directly
///   to the input, but it may still compose usefully in a higher-order
///   context — e.g. `App(add, 1)` Bottom-probes against a `List` input
///   but is essential as a fold callback);
/// - any apply errors out.
///
/// Only deduping on **concrete non-Bottom probe values** keeps the
/// soundness narrow: two closures that produce identical concrete
/// values when applied directly to the example inputs are equivalent
/// for the goal-level use `App(closure, Param(0))`. Whether they're
/// equivalent in arbitrary higher-order contexts is not guaranteed,
/// but for M2 trivial tasks the goal-level use is the only one.
pub(crate) fn probe_obs_key(
    arena: &lang::arena::Arena,
    lib: &Library,
    values: &[Value],
    inputs: &[Value],
    fuel: u32,
) -> Option<u64> {
    if values.len() != inputs.len() {
        return None;
    }
    let mut applied = Vec::with_capacity(values.len());
    for (closure, input) in values.iter().zip(inputs.iter()) {
        if !matches!(closure, Value::Closure(_)) {
            return None;
        }
        let mut f = fuel;
        match apply(arena, lib, closure.clone(), input.clone(), &mut f) {
            Ok(v) => {
                if matches!(v, Value::Closure(_)) || v.is_bottom() {
                    return None;
                }
                applied.push(v);
            }
            Err(_) => return None,
        }
    }
    let mut h = FxHasher::default();
    hash_values(&applied, &mut h);
    Some(h.finish())
}
