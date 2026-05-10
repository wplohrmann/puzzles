//! The library: a list of primitives (built-in or learned).
//!
//! Each `PrimKind::Learned` body is a `NodeId` into the library's own
//! `arena` — *not* the caller's program arena. The evaluator routes
//! Learned bodies through `&lib.arena`. The arena is deliberately not
//! part of the serialized form yet (M2 has no Learned primitives); when
//! abstraction sleep lands in M3 the Library serializer will need to
//! emit the arena's reachable subgraph alongside `primitives`.

use serde::{Deserialize, Serialize};

use crate::arena::{Arena, NodeId};
use crate::builtin::BuiltinId;

pub type PrimId = u32;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Primitive {
    pub name: String,
    /// Number of curried arguments before the primitive executes.
    pub arity: u8,
    pub kind: PrimKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimKind {
    /// Implemented in the interpreter directly.
    Builtin(BuiltinId),
    /// A closed program living in the enclosing `Library::arena`.
    Learned { body: NodeId, body_size: u32 },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Library {
    pub primitives: Vec<Primitive>,
    /// Storage for `Learned` primitive bodies. Empty in M2 (no
    /// abstraction sleep yet). Skipped on serde — M3 will need to wire
    /// arena round-tripping at the same time as the first Learned
    /// primitive is created.
    #[serde(skip, default)]
    pub arena: Arena,
}

impl Library {
    pub fn new() -> Self { Self::default() }

    pub fn add(&mut self, p: Primitive) -> PrimId {
        let id = self.primitives.len() as PrimId;
        self.primitives.push(p);
        id
    }

    pub fn get(&self, id: PrimId) -> &Primitive { &self.primitives[id as usize] }

    pub fn arity(&self, id: PrimId) -> usize {
        self.primitives[id as usize].arity as usize
    }

    pub fn lookup(&self, name: &str) -> Option<PrimId> {
        self.primitives.iter().position(|p| p.name == name).map(|i| i as PrimId)
    }

    pub fn len(&self) -> usize { self.primitives.len() }
    pub fn is_empty(&self) -> bool { self.primitives.is_empty() }
}
