//! The pool: every candidate node the search has materialised so far,
//! indexed by `NodeId` (for hash-cons-canonical dedup) and by program
//! size (for split-by-size action enumeration).
//!
//! M2 has no behavioural pruning beyond hash-cons. Every (f, a) split
//! is enumerated, evaluated, and added unless its `NodeId` is already
//! present. Pruning candidates by their runtime values is left to the
//! M4 neural prior.

use rustc_hash::FxHashMap;

use lang::arena::NodeId;
use lang::eval::Value;

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

    pub fn try_add(&mut self, node: NodeId, size: u32, values: Vec<Value>) -> AddOutcome {
        if self.by_node.contains_key(&node) {
            return AddOutcome::DuplicateNode;
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
}
