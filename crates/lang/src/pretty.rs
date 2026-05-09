//! Debug pretty-printer for nodes. Not optimised for speed; for tests and
//! the future inspect CLI.

use crate::arena::{Arena, NodeId};
use crate::ir::{LitValue, NodeKind};
use crate::library::Library;

pub fn pretty(arena: &Arena, lib: &Library, root: NodeId) -> String {
    let mut s = String::new();
    fmt_node(arena, lib, root, &mut s, 0);
    s
}

fn fmt_node(arena: &Arena, lib: &Library, id: NodeId, out: &mut String, depth: u16) {
    match arena.kind(id) {
        NodeKind::Literal(v) => match v {
            LitValue::Int(i) => out.push_str(&format!("{}", i)),
            LitValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            LitValue::Float(f) => out.push_str(&format!("{}", f)),
            LitValue::Char(c) => out.push_str(&format!("{:?}", c)),
        },
        NodeKind::Param { index } => out.push_str(&format!("v{}", depth.saturating_sub(*index + 1))),
        NodeKind::Lambda { body, .. } => {
            out.push_str(&format!("(λv{}. ", depth));
            fmt_node(arena, lib, *body, out, depth + 1);
            out.push(')');
        }
        NodeKind::App { func, arg } => {
            out.push('(');
            fmt_node(arena, lib, *func, out, depth);
            out.push(' ');
            fmt_node(arena, lib, *arg, out, depth);
            out.push(')');
        }
        NodeKind::PrimRef(p) => out.push_str(&lib.get(*p).name),
    }
}
