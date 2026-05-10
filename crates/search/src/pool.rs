//! The pool: every node the search has materialised so far, indexed by
//! NodeId, by program size (for split-by-size action enumeration), and by
//! `(canonical_type, value_tuple_hash)` (for observational-equivalence
//! dedup).
//!
//! Hash-cons in the arena guarantees structural duplicates collapse to
//! the same NodeId; the pool keeps at most one entry per NodeId.

use std::hash::Hasher;

use rustc_hash::FxHashMap;
use rustc_hash::FxHasher;

use lang::arena::{Arena, NodeId};
use lang::eval::{apply, Value};
use lang::library::Library;
use lang::ty::{unify, Subst, Ty, TyCon};

use crate::value_hash::hash_values;

/// One pool entry: a node, its evaluated values per task example, and the
/// program size of its DAG (number of *unique* sub-nodes).
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub node: NodeId,
    pub size: u32,
    pub values: Vec<Value>,
}

#[derive(Default)]
pub(crate) struct Pool {
    pub entries: Vec<Entry>,
    /// node_id -> entry index. Lookup decides whether a freshly hash-consed
    /// node is already present.
    pub by_node: FxHashMap<NodeId, usize>,
    /// (canonical type, value-tuple-hash) collapsed into one u64. Maps to
    /// entry index. Only used for entries whose type is non-functional.
    pub by_obs: FxHashMap<u64, usize>,
    /// `by_size[k]` holds entry indices of all size-k entries. Index 0 is
    /// always empty (no entries have size 0).
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

    /// Try to add a new entry. Callers pre-compute the obs-eq key:
    /// `Some(key)` means "this entry is observationally equivalent to any
    /// other entry with the same key — drop if one already exists";
    /// `None` means "skip obs-eq dedup for this entry".
    ///
    /// `node`-level dedup is always applied (hash-cons collapses
    /// structural duplicates).
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
            if let Some(&existing) = self.by_obs.get(&key) {
                let _ = existing;
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

/// Compute an obs-eq key for a non-function-typed entry. Returns `None`
/// if the type is functional (caller should consider the extended /
/// probe-based key, or skip).
pub(crate) fn value_obs_key(ty: &Ty, values: &[Value]) -> Option<u64> {
    if !should_obs_eq(ty) {
        return None;
    }
    let mut h = FxHasher::default();
    hash_type(ty, &mut h);
    hash_values(values, &mut h);
    Some(h.finish())
}

/// Compute the *extended* obs-eq key for a function-typed entry whose
/// type is `arg_ty -> R` with `R` concrete. The key hashes (R type,
/// applied_values) — i.e. what the closure produces when applied to the
/// task's example inputs.
///
/// Returns `None` if the entry's type isn't of the right shape, or if
/// any apply failed.
pub(crate) fn applied_obs_key(
    arena: &Arena,
    lib: &Library,
    ty: &Ty,
    values: &[Value],
    inputs: &[Value],
    arg_ty: &Ty,
    fuel: u32,
) -> Option<u64> {
    let (a_ty, r_ty) = ty.as_func()?;
    if !should_obs_eq(r_ty) {
        return None;
    }
    // Require a_ty unifies with arg_ty (i.e., the closure's first arg is
    // shaped like the task's input).
    let mut s = Subst::default();
    if unify(a_ty, arg_ty, &mut s).is_err() {
        return None;
    }
    if values.len() != inputs.len() {
        return None;
    }
    let mut applied = Vec::with_capacity(values.len());
    for (closure, input) in values.iter().zip(inputs.iter()) {
        let mut f = fuel;
        match apply(arena, lib, closure.clone(), input.clone(), &mut f) {
            Ok(v) => applied.push(v),
            Err(_) => return None,
        }
    }
    let mut h = FxHasher::default();
    hash_type(r_ty, &mut h);
    hash_values(&applied, &mut h);
    Some(h.finish())
}

#[derive(Debug)]
pub(crate) enum AddOutcome {
    Added,
    DuplicateNode,
    ObsEqPruned,
}

/// Whether obs-eq dedup applies to a value of this type. We dedup
/// concrete data types (Int, Bool, List<concrete>, Pair<concrete>, ...) but
/// skip function types — closures don't compare structurally.
fn should_obs_eq(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true, // a polymorphic value is concrete enough at runtime
        Ty::Con(TyCon::Fn, _) => false,
        Ty::Con(_, args) => args.iter().all(should_obs_eq),
    }
}

fn hash_type<H: Hasher>(ty: &Ty, h: &mut H) {
    use std::hash::Hash;
    ty.hash(h);
}
