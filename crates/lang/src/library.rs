//! The library: a list of primitives (built-in or learned).

use serde::{Deserialize, Serialize};

use crate::arena::NodeId;
use crate::builtin::BuiltinId;
use crate::ty::TypeScheme;

pub type PrimId = u32;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Primitive {
    pub name: String,
    pub ty: TypeScheme,
    pub kind: PrimKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimKind {
    /// Implemented in the interpreter directly.
    Builtin(BuiltinId),
    /// A closed program in the library's own arena.
    Learned { body: NodeId, body_size: u32 },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Library {
    pub primitives: Vec<Primitive>,
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
        // Arity = number of leading function arrows in the polytype body.
        let mut t = &self.primitives[id as usize].ty.body;
        let mut a = 0;
        while let Some((_, ret)) = t.as_func() {
            a += 1;
            t = ret;
        }
        a
    }

    pub fn lookup(&self, name: &str) -> Option<PrimId> {
        self.primitives.iter().position(|p| p.name == name).map(|i| i as PrimId)
    }

    pub fn len(&self) -> usize { self.primitives.len() }
    pub fn is_empty(&self) -> bool { self.primitives.is_empty() }
}
