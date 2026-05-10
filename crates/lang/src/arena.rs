//! Hash-consed node arena.
//!
//! Every node is uniquely identified by its structural content (kind +
//! children). `Arena::intern` is the only path that creates ids;
//! reinserting a structurally-equal node returns the existing id. This
//! means **every program is in canonical form by construction**:
//! semantically-equal subterms share storage and equality is
//! `NodeId == NodeId`.

use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};

pub use crate::ir::{Node, NodeId, NodeKind};

#[derive(Debug, Default)]
pub struct Arena {
    nodes: Vec<Node>,
    intern: FxHashMap<u64, Vec<NodeId>>,
}

impl Arena {
    pub fn new() -> Self { Self::default() }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    pub fn node(&self, id: NodeId) -> &Node { &self.nodes[id.0 as usize] }
    pub fn kind(&self, id: NodeId) -> &NodeKind { &self.node(id).kind }

    pub(crate) fn intern(&mut self, kind: NodeKind) -> NodeId {
        let hash = structural_hash(&kind);
        if let Some(bucket) = self.intern.get(&hash) {
            for &id in bucket {
                let n = &self.nodes[id.0 as usize];
                if n.hash == hash && n.kind == kind {
                    return id;
                }
            }
        }
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node { kind, hash });
        self.intern.entry(hash).or_default().push(id);
        id
    }

    /// Iterate all interned nodes in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().enumerate().map(|(i, n)| (NodeId(i as u32), n))
    }

    /// Children of `id`, in the order they appear in the kind.
    pub fn children(&self, id: NodeId) -> Vec<NodeId> {
        match &self.node(id).kind {
            NodeKind::Literal(_) | NodeKind::Param { .. } | NodeKind::PrimRef(_) => Vec::new(),
            NodeKind::Lambda { body } => vec![*body],
            NodeKind::App { func, arg } => vec![*func, *arg],
        }
    }

    /// Topologically-sorted set of all transitively-reachable nodes from
    /// `root`, in dependency order (children before parents). Useful for
    /// serialisation and bottom-up traversal.
    pub fn reachable_topo(&self, root: NodeId) -> Vec<NodeId> {
        let mut seen = FxHashMap::<NodeId, bool>::default();
        let mut order = Vec::new();
        fn dfs(
            arena: &Arena, id: NodeId,
            seen: &mut FxHashMap<NodeId, bool>, out: &mut Vec<NodeId>,
        ) {
            if seen.contains_key(&id) { return; }
            seen.insert(id, true);
            for c in arena.children(id) {
                dfs(arena, c, seen, out);
            }
            out.push(id);
        }
        dfs(self, root, &mut seen, &mut order);
        order
    }
}

fn structural_hash(kind: &NodeKind) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    discriminant(kind).hash(&mut h);
    match kind {
        NodeKind::Literal(v) => v.hash(&mut h),
        NodeKind::Param { index } => index.hash(&mut h),
        NodeKind::Lambda { body } => body.hash(&mut h),
        NodeKind::App { func, arg } => {
            func.hash(&mut h);
            arg.hash(&mut h);
        }
        NodeKind::PrimRef(p) => p.hash(&mut h),
    }
    h.finish()
}

fn discriminant(k: &NodeKind) -> u8 {
    match k {
        NodeKind::Literal(_) => 0,
        NodeKind::Param { .. } => 1,
        NodeKind::Lambda { .. } => 2,
        NodeKind::App { .. } => 3,
        NodeKind::PrimRef(_) => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::LitValue;

    #[test]
    fn intern_idempotent() {
        let mut a = Arena::new();
        let id1 = a.intern(NodeKind::Literal(LitValue::Int(42)));
        let id2 = a.intern(NodeKind::Literal(LitValue::Int(42)));
        assert_eq!(id1, id2);
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn intern_distinguishes_values() {
        let mut a = Arena::new();
        let i1 = a.intern(NodeKind::Literal(LitValue::Int(1)));
        let i2 = a.intern(NodeKind::Literal(LitValue::Int(2)));
        assert_ne!(i1, i2);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn intern_collapses_primrefs() {
        // Without static types, PrimRef(p) is one node regardless of how
        // it would have been "instantiated". This is the right behaviour:
        // a primitive ref is a primitive ref.
        let mut a = Arena::new();
        let i1 = a.intern(NodeKind::PrimRef(0));
        let i2 = a.intern(NodeKind::PrimRef(0));
        assert_eq!(i1, i2);
    }
}
