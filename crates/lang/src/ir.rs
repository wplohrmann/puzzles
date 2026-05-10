//! Internal representation: nodes, kinds, literal values.
//!
//! There is no static type system: nodes carry only structural
//! information (kind + children). Types are runtime concepts, encoded by
//! the `Value` variants in `eval.rs`. Primitives that receive
//! mismatched runtime types return `Value::Bottom` rather than failing
//! at construction.

use serde::{Deserialize, Serialize};

use crate::library::PrimId;

/// Literal values storable in the program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LitValue {
    Int(i64),
    Bool(bool),
    Float(f64),
    Char(char),
}

impl Eq for LitValue {}
impl std::hash::Hash for LitValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            LitValue::Int(i) => { 0u8.hash(state); i.hash(state); }
            LitValue::Bool(b) => { 1u8.hash(state); b.hash(state); }
            LitValue::Float(f) => { 2u8.hash(state); f.to_bits().hash(state); }
            LitValue::Char(c) => { 3u8.hash(state); c.hash(state); }
        }
    }
}

/// Node kinds. Children of `Lambda`/`App` are `NodeId`s into the same arena.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    Literal(LitValue),
    /// de-Bruijn-indexed parameter. `index = 0` is the innermost lambda's
    /// parameter; index `N` reaches `N` lambdas outward.
    Param { index: u16 },
    Lambda { body: NodeId },
    App { func: NodeId, arg: NodeId },
    PrimRef(PrimId),
}

/// Hash-cons-canonical id into an `Arena`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn raw(self) -> u32 { self.0 }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
    /// Structural hash for the intern table.
    pub hash: u64,
}
