//! Type system: monotypes, polytypes, unification, instantiation, canonicalisation.
//!
//! The type system is "HM-lite": every primitive carries a hand-written
//! polytype, and `Apply` does unification at every call site. There is no
//! general type *inference* because every node's type is determined when it
//! is constructed. See `docs/01-language.md` § "Types".
//!
//! ## Invariants
//! - Free type variables in a stored `Ty` are *implicitly universally
//!   quantified at the use site*: every time a node is used as a child in
//!   `App`, its free vars are renamed to globally-unique fresh ids before
//!   unification.
//! - Stored `Ty`s are in **canonical form**: free vars renumbered to 0, 1,
//!   2, ... in left-to-right traversal order. This makes structural hashing
//!   and equality work for polymorphic types.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A type variable. Globally unique within a session.
pub type TyVar = u32;

/// Ground type constructors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TyCon {
    Int,
    Bool,
    Float,
    Char,
    /// `List<T>` — sequences of `T`.
    List,
    /// `Pair<A, B>` — 2-tuples; n-tuples nest.
    Pair,
    /// `Fn<A, B>` — curried function type.
    Fn,
}

impl TyCon {
    pub fn arity(self) -> usize {
        match self {
            TyCon::Int | TyCon::Bool | TyCon::Float | TyCon::Char => 0,
            TyCon::List => 1,
            TyCon::Pair | TyCon::Fn => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TyCon::Int => "Int",
            TyCon::Bool => "Bool",
            TyCon::Float => "Float",
            TyCon::Char => "Char",
            TyCon::List => "List",
            TyCon::Pair => "Pair",
            TyCon::Fn => "Fn",
        }
    }
}

/// A type: either a variable or a type constructor applied to arguments.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    Var(TyVar),
    Con(TyCon, Vec<Ty>),
}

impl Ty {
    // -- Constructors -------------------------------------------------------

    pub fn int() -> Self { Ty::Con(TyCon::Int, vec![]) }
    pub fn bool() -> Self { Ty::Con(TyCon::Bool, vec![]) }
    pub fn float() -> Self { Ty::Con(TyCon::Float, vec![]) }
    pub fn char() -> Self { Ty::Con(TyCon::Char, vec![]) }
    pub fn list(t: Ty) -> Self { Ty::Con(TyCon::List, vec![t]) }
    pub fn pair(a: Ty, b: Ty) -> Self { Ty::Con(TyCon::Pair, vec![a, b]) }
    pub fn func(a: Ty, b: Ty) -> Self { Ty::Con(TyCon::Fn, vec![a, b]) }
    pub fn var(v: TyVar) -> Self { Ty::Var(v) }

    /// Build `a -> b -> ... -> ret`.
    pub fn func_chain(args: &[Ty], ret: Ty) -> Self {
        args.iter().rev().fold(ret, |acc, a| Ty::func(a.clone(), acc))
    }

    // -- Queries ------------------------------------------------------------

    /// Free type variables in left-to-right traversal order, deduplicated.
    pub fn free_vars(&self) -> Vec<TyVar> {
        let mut seen = FxHashMap::default();
        let mut out = Vec::new();
        fn walk(t: &Ty, seen: &mut FxHashMap<TyVar, ()>, out: &mut Vec<TyVar>) {
            match t {
                Ty::Var(v) => {
                    if !seen.contains_key(v) {
                        seen.insert(*v, ());
                        out.push(*v);
                    }
                }
                Ty::Con(_, args) => {
                    for a in args {
                        walk(a, seen, out);
                    }
                }
            }
        }
        walk(self, &mut seen, &mut out);
        out
    }

    pub fn contains(&self, v: TyVar) -> bool {
        match self {
            Ty::Var(u) => *u == v,
            Ty::Con(_, args) => args.iter().any(|a| a.contains(v)),
        }
    }

    /// If this type is `Fn<A, B>`, return `(A, B)`.
    pub fn as_func(&self) -> Option<(&Ty, &Ty)> {
        match self {
            Ty::Con(TyCon::Fn, args) if args.len() == 2 => Some((&args[0], &args[1])),
            _ => None,
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Var(v) => write!(f, "t{}", v),
            Ty::Con(TyCon::Fn, args) if args.len() == 2 => {
                // Right-associative arrow.
                let needs_parens = matches!(&args[0], Ty::Con(TyCon::Fn, _));
                if needs_parens {
                    write!(f, "({}) -> {}", args[0], args[1])
                } else {
                    write!(f, "{} -> {}", args[0], args[1])
                }
            }
            Ty::Con(c, args) if args.is_empty() => write!(f, "{}", c.name()),
            Ty::Con(c, args) => {
                write!(f, "{}<", c.name())?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ">")
            }
        }
    }
}

// -- Type schemes ---------------------------------------------------------

/// A polytype: forall qs. body.
///
/// Convention: `body` uses canonical TyVars 0..qs.len(). Stored verbatim;
/// each *use* re-instantiates with globally-unique fresh TyVars.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeScheme {
    pub n_vars: u32,
    pub body: Ty,
}

impl TypeScheme {
    pub fn mono(t: Ty) -> Self {
        TypeScheme { n_vars: 0, body: t }
    }

    /// Construct from explicit quantified vars + body. The body is rewritten
    /// so its quantified vars are 0..n_vars.
    pub fn forall(qs: Vec<TyVar>, body: Ty) -> Self {
        let mut map: FxHashMap<TyVar, TyVar> = FxHashMap::default();
        for (i, q) in qs.iter().enumerate() {
            map.insert(*q, i as TyVar);
        }
        let body = rename(&body, &map);
        TypeScheme { n_vars: qs.len() as u32, body }
    }

    /// Instantiate the polytype with globally-fresh vars from `gen`.
    pub fn instantiate(&self, gen: &mut TyVarGen) -> Ty {
        if self.n_vars == 0 {
            return self.body.clone();
        }
        let mut subst = Subst::default();
        for i in 0..self.n_vars {
            subst.insert(i as TyVar, Ty::Var(gen.fresh()));
        }
        subst.apply(&self.body)
    }
}

impl fmt::Display for TypeScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.n_vars == 0 {
            write!(f, "{}", self.body)
        } else {
            write!(f, "forall ")?;
            for i in 0..self.n_vars {
                if i > 0 { write!(f, " ")?; }
                write!(f, "t{}", i)?;
            }
            write!(f, ". {}", self.body)
        }
    }
}

// -- Fresh-variable generator --------------------------------------------

#[derive(Debug, Default)]
pub struct TyVarGen { next: u32 }

impl TyVarGen {
    pub fn new() -> Self { Self { next: 1_000_000 } }
    pub fn fresh(&mut self) -> TyVar {
        let v = self.next;
        self.next += 1;
        v
    }
}

// -- Substitutions and unification ----------------------------------------

/// A type substitution. Composition is via `apply`.
#[derive(Clone, Debug, Default)]
pub struct Subst {
    map: FxHashMap<TyVar, Ty>,
}

impl Subst {
    pub fn insert(&mut self, v: TyVar, t: Ty) {
        self.map.insert(v, t);
    }

    pub fn get(&self, v: TyVar) -> Option<&Ty> { self.map.get(&v) }

    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    /// Walk `t`, replacing every `Var(v)` with `subst(v)`. Substitution
    /// targets are followed (chain semantics) — this is what unification
    /// needs. Cycle protection is the caller's responsibility (via the
    /// occurs check on `bind`); a non-cycle-protecting apply is used for
    /// pure renaming (see `rename`).
    pub fn apply(&self, t: &Ty) -> Ty {
        match t {
            Ty::Var(v) => match self.map.get(v) {
                Some(Ty::Var(u)) if u == v => Ty::Var(*v),
                Some(t2) => self.apply(t2),
                None => Ty::Var(*v),
            },
            Ty::Con(c, args) => {
                Ty::Con(*c, args.iter().map(|a| self.apply(a)).collect())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnifyError {
    #[error("cannot unify {0} with {1}")]
    Mismatch(String, String),
    #[error("infinite type: t{0} occurs in {1}")]
    Occurs(TyVar, String),
}

/// Unify `a` and `b`, extending `subst`. On error, `subst` may be partially
/// extended (caller is expected to roll back if it cares).
pub fn unify(a: &Ty, b: &Ty, subst: &mut Subst) -> Result<(), UnifyError> {
    let a = subst.apply(a);
    let b = subst.apply(b);
    match (a, b) {
        (Ty::Var(v), t) | (t, Ty::Var(v)) => bind(v, &t, subst),
        (Ty::Con(c1, a1), Ty::Con(c2, a2)) => {
            if c1 != c2 || a1.len() != a2.len() {
                return Err(UnifyError::Mismatch(
                    format!("{}", Ty::Con(c1, a1)),
                    format!("{}", Ty::Con(c2, a2)),
                ));
            }
            for (x, y) in a1.iter().zip(a2.iter()) {
                unify(x, y, subst)?;
            }
            Ok(())
        }
    }
}

fn bind(v: TyVar, t: &Ty, subst: &mut Subst) -> Result<(), UnifyError> {
    if let Ty::Var(u) = t { if *u == v { return Ok(()); } }
    if t.contains(v) {
        return Err(UnifyError::Occurs(v, format!("{}", t)));
    }
    subst.insert(v, t.clone());
    Ok(())
}

// -- Renaming and canonicalisation ---------------------------------------

/// Rename free type variables in a single pass — does *not* chase chains.
/// Use this for pure renaming (canonicalisation, alpha-renaming) where
/// `Subst::apply`'s chain semantics would form cycles.
pub fn rename(t: &Ty, map: &FxHashMap<TyVar, TyVar>) -> Ty {
    match t {
        Ty::Var(v) => Ty::Var(*map.get(v).unwrap_or(v)),
        Ty::Con(c, args) => {
            Ty::Con(*c, args.iter().map(|a| rename(a, map)).collect())
        }
    }
}

/// Renumber free vars to 0, 1, 2, ... in left-to-right traversal order.
/// Two types that are equivalent up to alpha-renaming canonicalise to the
/// same `Ty`, which makes them hash-equal.
pub fn canonicalize(t: &Ty) -> Ty {
    let frees = t.free_vars();
    let mut map: FxHashMap<TyVar, TyVar> = FxHashMap::default();
    for (i, v) in frees.iter().enumerate() {
        map.insert(*v, i as TyVar);
    }
    rename(t, &map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unify_concrete() {
        let mut s = Subst::default();
        unify(&Ty::int(), &Ty::int(), &mut s).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn unify_var() {
        let mut s = Subst::default();
        unify(&Ty::Var(0), &Ty::int(), &mut s).unwrap();
        assert_eq!(s.apply(&Ty::Var(0)), Ty::int());
    }

    #[test]
    fn unify_mismatch() {
        let mut s = Subst::default();
        let r = unify(&Ty::int(), &Ty::bool(), &mut s);
        assert!(matches!(r, Err(UnifyError::Mismatch(..))));
    }

    #[test]
    fn occurs_check() {
        let mut s = Subst::default();
        let r = unify(&Ty::Var(0), &Ty::list(Ty::Var(0)), &mut s);
        assert!(matches!(r, Err(UnifyError::Occurs(..))));
    }

    #[test]
    fn instantiate_fresh() {
        let scheme = TypeScheme::forall(
            vec![0],
            Ty::func(Ty::Var(0), Ty::list(Ty::Var(0))),
        );
        let mut g = TyVarGen::new();
        let a = scheme.instantiate(&mut g);
        let b = scheme.instantiate(&mut g);
        assert_ne!(a, b); // different fresh ids each instantiation
        // both are still of the form Var -> List Var
        for ty in &[a, b] {
            let (arg, ret) = ty.as_func().unwrap();
            assert!(matches!(arg, Ty::Var(_)));
            assert!(matches!(ret, Ty::Con(TyCon::List, _)));
        }
    }

    #[test]
    fn canonicalise_renumbers_in_order() {
        let t = Ty::pair(Ty::Var(7), Ty::pair(Ty::Var(3), Ty::Var(7)));
        assert_eq!(
            canonicalize(&t),
            Ty::pair(Ty::Var(0), Ty::pair(Ty::Var(1), Ty::Var(0))),
        );
    }

    #[test]
    fn typescheme_forall_canonicalises_body() {
        let s = TypeScheme::forall(
            vec![5, 7],
            Ty::func(Ty::Var(5), Ty::Var(7)),
        );
        assert_eq!(s.n_vars, 2);
        // body should now use canonical vars 0, 1
        assert_eq!(s.body, Ty::func(Ty::Var(0), Ty::Var(1)));
    }
}
