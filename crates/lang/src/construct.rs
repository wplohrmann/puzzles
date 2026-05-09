//! Typed constructors for nodes. The only public path to building IR.
//!
//! Every constructor goes through the arena's hash-cons interner, so two
//! semantically-equal constructions yield the same `NodeId`. `app` does
//! unification at the call site, instantiating polytypes with globally-fresh
//! type variables.

use crate::arena::{Arena, NodeId};
use crate::error::{Error, Result};
use crate::ir::{LitValue, NodeKind};
use crate::library::{Library, PrimId};
use crate::ty::{unify, Subst, Ty, TyVar, TyVarGen};

pub fn lit(arena: &mut Arena, v: LitValue) -> NodeId {
    let ty = v.ty();
    arena.intern(NodeKind::Literal(v), ty)
}

pub fn param(arena: &mut Arena, index: u16, ty: Ty) -> NodeId {
    arena.intern(NodeKind::Param { index }, ty)
}

pub fn lambda(arena: &mut Arena, param_ty: Ty, body: NodeId) -> NodeId {
    let body_ty = arena.ty(body).clone();
    let ty = Ty::func(param_ty.clone(), body_ty);
    arena.intern(NodeKind::Lambda { param_ty, body }, ty)
}

pub fn prim_ref(arena: &mut Arena, lib: &Library, p: PrimId) -> NodeId {
    let body = lib.get(p).ty.body.clone();
    arena.intern(NodeKind::PrimRef(p), body)
}

/// Apply `func` to `arg`. Both children's stored types are instantiated
/// with globally-fresh vars before unification, so two uses of the same
/// polymorphic node at different concrete types produce different App nodes.
pub fn app(
    arena: &mut Arena,
    gen: &mut TyVarGen,
    func: NodeId,
    arg: NodeId,
) -> Result<NodeId> {
    let func_ty = instantiate(arena.ty(func), gen);
    let arg_ty = instantiate(arena.ty(arg), gen);

    let result_var = Ty::Var(gen.fresh());
    let expected = Ty::func(arg_ty.clone(), result_var.clone());

    let mut subst = Subst::default();
    unify(&func_ty, &expected, &mut subst).map_err(|e| Error::ApplyMismatch {
        param: format!("{}", func_ty),
        arg: format!("{}", arg_ty),
        source: e,
    })?;

    let ty = subst.apply(&result_var);
    Ok(arena.intern(NodeKind::App { func, arg }, ty))
}

/// Replace every free variable in `t` with a fresh one drawn from `gen`.
/// Two distinct free vars produce two distinct fresh vars.
fn instantiate(t: &Ty, gen: &mut TyVarGen) -> Ty {
    let frees = t.free_vars();
    if frees.is_empty() {
        return t.clone();
    }
    let mut subst = Subst::default();
    for v in frees {
        subst.insert(v as TyVar, Ty::Var(gen.fresh()));
    }
    subst.apply(t)
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
    fn lit_int_is_int() {
        let mut a = Arena::new();
        let id = lit(&mut a, LitValue::Int(7));
        assert_eq!(*a.ty(id), Ty::int());
    }

    #[test]
    fn add_1_2_is_int() {
        let lib = lib();
        let mut a = Arena::new();
        let mut g = TyVarGen::new();
        let add_ref = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Add));
        let one = lit(&mut a, LitValue::Int(1));
        let two = lit(&mut a, LitValue::Int(2));
        let app1 = app(&mut a, &mut g, add_ref, one).unwrap();
        // After (add 1), the type should be Int -> Int.
        assert_eq!(*a.ty(app1), Ty::func(Ty::int(), Ty::int()));
        let app2 = app(&mut a, &mut g, app1, two).unwrap();
        assert_eq!(*a.ty(app2), Ty::int());
    }

    #[test]
    fn cons_int_into_nil_is_list_int() {
        let lib = lib();
        let mut a = Arena::new();
        let mut g = TyVarGen::new();
        let cons = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Cons));
        let nil = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Nil));
        let one = lit(&mut a, LitValue::Int(1));
        let app1 = app(&mut a, &mut g, cons, one).unwrap();
        let app2 = app(&mut a, &mut g, app1, nil).unwrap();
        assert_eq!(*a.ty(app2), Ty::list(Ty::int()));
    }

    #[test]
    fn type_mismatch_rejected() {
        let lib = lib();
        let mut a = Arena::new();
        let mut g = TyVarGen::new();
        let add_ref = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Add));
        let true_ = lit(&mut a, LitValue::Bool(true));
        // add expects Int, not Bool — should error.
        let r = app(&mut a, &mut g, add_ref, true_);
        assert!(matches!(r, Err(Error::ApplyMismatch { .. })));
    }

    #[test]
    fn pair_at_two_uses_keeps_polymorphism() {
        // pair nil nil should have type Pair (List a) (List b) — two
        // independent List types.
        let lib = lib();
        let mut a = Arena::new();
        let mut g = TyVarGen::new();
        let pair_ = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Pair));
        let nil = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Nil));
        let app1 = app(&mut a, &mut g, pair_, nil).unwrap();
        let app2 = app(&mut a, &mut g, app1, nil).unwrap();
        // The type should be Pair<List<a>, List<b>>, canonicalised to a=0, b=1.
        assert_eq!(
            *a.ty(app2),
            Ty::pair(Ty::list(Ty::Var(0)), Ty::list(Ty::Var(1))),
        );
    }

    #[test]
    fn lambda_id_int() {
        // λx:Int. x — type Int -> Int
        let mut a = Arena::new();
        let p = param(&mut a, 0, Ty::int());
        let l = lambda(&mut a, Ty::int(), p);
        assert_eq!(*a.ty(l), Ty::func(Ty::int(), Ty::int()));
    }

    #[test]
    fn dedup_via_hash_cons() {
        let lib = lib();
        let mut a = Arena::new();
        let mut g = TyVarGen::new();
        let cons = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Cons));
        let nil = prim_ref(&mut a, &lib, lookup(&lib, BuiltinId::Nil));
        let one_a = lit(&mut a, LitValue::Int(1));
        let one_b = lit(&mut a, LitValue::Int(1));
        assert_eq!(one_a, one_b, "literals dedup");
        let app1_a = app(&mut a, &mut g, cons, one_a).unwrap();
        let app1_b = app(&mut a, &mut g, cons, one_b).unwrap();
        assert_eq!(app1_a, app1_b, "apps dedup");
        let _ = nil;
    }
}
