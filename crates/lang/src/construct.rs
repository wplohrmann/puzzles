//! Constructors for nodes. The only public path to building IR.
//!
//! Without a static type system there is no construction-time
//! type-checking: any `App(f, a)` is admissible. Mismatched runtime
//! types surface as `Value::Bottom` during evaluation.

use crate::arena::{Arena, NodeId};
use crate::ir::{LitValue, NodeKind};
use crate::library::{Library, PrimId};

pub fn lit(arena: &mut Arena, v: LitValue) -> NodeId {
    arena.intern(NodeKind::Literal(v))
}

pub fn param(arena: &mut Arena, index: u16) -> NodeId {
    arena.intern(NodeKind::Param { index })
}

pub fn lambda(arena: &mut Arena, body: NodeId) -> NodeId {
    arena.intern(NodeKind::Lambda { body })
}

pub fn prim_ref(arena: &mut Arena, _lib: &Library, p: PrimId) -> NodeId {
    arena.intern(NodeKind::PrimRef(p))
}

pub fn app(arena: &mut Arena, func: NodeId, arg: NodeId) -> NodeId {
    arena.intern(NodeKind::App { func, arg })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::{seed_builtin_library, BuiltinId};

    fn lib() -> Library {
        seed_builtin_library()
    }

    fn lookup(lib: &Library, b: BuiltinId) -> PrimId {
        lib.lookup(b.name()).expect("builtin present")
    }

    #[test]
    fn add_1_2_constructs() {
        let lib = lib();
        let mut a = Arena::new();
        let add_ref = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Add));
        let one = lit(&mut a, LitValue::Int(1));
        let two = lit(&mut a, LitValue::Int(2));
        let app1 = app(&mut a, add_ref, one);
        let _ = app(&mut a, app1, two);
    }

    #[test]
    fn dedup_via_hash_cons() {
        let lib = lib();
        let mut a = Arena::new();
        let cons = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Cons));
        let one_a = lit(&mut a, LitValue::Int(1));
        let one_b = lit(&mut a, LitValue::Int(1));
        assert_eq!(one_a, one_b, "literals dedup");
        let app1_a = app(&mut a, cons, one_a);
        let app1_b = app(&mut a, cons, one_b);
        assert_eq!(app1_a, app1_b, "apps dedup");
    }
}
