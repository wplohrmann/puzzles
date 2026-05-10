//! Program serialisation: topo-ordered list of nodes, children as indices.
//!
//! Use this to round-trip programs to/from JSON or any other serde format.
//! `serialize` walks the reachable nodes from a root in topological order;
//! `deserialize` interns them back into an arena, returning the new root.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::arena::{Arena, NodeId};
use crate::ir::{LitValue, NodeKind};
use crate::library::PrimId;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProgramSerial {
    pub nodes: Vec<NodeSerial>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NodeSerial {
    pub kind: KindSerial,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum KindSerial {
    Literal(LitValue),
    Param { index: u16 },
    Lambda { body: u32 },
    App { func: u32, arg: u32 },
    PrimRef(PrimId),
}

/// Serialise the program rooted at `root` in `arena` to a portable form.
pub fn serialize(arena: &Arena, root: NodeId) -> ProgramSerial {
    let topo = arena.reachable_topo(root);
    let mut id_to_idx: FxHashMap<NodeId, u32> = FxHashMap::default();
    for (i, id) in topo.iter().enumerate() {
        id_to_idx.insert(*id, i as u32);
    }
    let nodes = topo
        .iter()
        .map(|id| {
            let n = arena.node(*id);
            let kind = match &n.kind {
                NodeKind::Literal(v) => KindSerial::Literal(v.clone()),
                NodeKind::Param { index } => KindSerial::Param { index: *index },
                NodeKind::Lambda { body } => KindSerial::Lambda { body: id_to_idx[body] },
                NodeKind::App { func, arg } => KindSerial::App {
                    func: id_to_idx[func],
                    arg: id_to_idx[arg],
                },
                NodeKind::PrimRef(p) => KindSerial::PrimRef(*p),
            };
            NodeSerial { kind }
        })
        .collect();
    ProgramSerial { nodes }
}

/// Reconstruct a program in `arena`, returning the new root id.
pub fn deserialize(repr: &ProgramSerial, arena: &mut Arena) -> NodeId {
    let mut idx_to_id: Vec<NodeId> = Vec::with_capacity(repr.nodes.len());
    for n in &repr.nodes {
        let kind = match &n.kind {
            KindSerial::Literal(v) => NodeKind::Literal(v.clone()),
            KindSerial::Param { index } => NodeKind::Param { index: *index },
            KindSerial::Lambda { body } => NodeKind::Lambda { body: idx_to_id[*body as usize] },
            KindSerial::App { func, arg } => NodeKind::App {
                func: idx_to_id[*func as usize],
                arg: idx_to_id[*arg as usize],
            },
            KindSerial::PrimRef(p) => NodeKind::PrimRef(*p),
        };
        let id = arena.intern(kind);
        idx_to_id.push(id);
    }
    *idx_to_id.last().expect("program has at least one node")
}
