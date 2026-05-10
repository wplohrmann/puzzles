//! Built-in primitives: their identity, types, and a helper to seed a Library.

use serde::{Deserialize, Serialize};

use crate::library::{Library, PrimKind, Primitive};
use crate::ty::{Ty, TypeScheme};

/// Built-in primitive identifier. Kept stable so serialised programs from
/// older versions still resolve correctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuiltinId {
    // Numeric
    Add, Sub, Mul, Div,
    Lt, Eq,

    // Boolean
    Not, And, Or,
    /// `if : forall a. Bool -> a -> a -> a`
    If,

    // Pair
    Pair, Fst, Snd,

    // List
    Nil, Cons,
    /// `fold : forall a b. (a -> b -> b) -> b -> List a -> b`
    /// Right-fold semantics: `fold f z [a,b,c] = f a (f b (f c z))`.
    Fold,
    /// `unfold : forall a b. (b -> Pair (Pair a b) Bool) -> b -> List a`
    Unfold,

    // Combinators
    /// `k : forall a b. a -> b -> a` — the K combinator (`λx y. x`).
    K,
    /// `b : forall a b c. (b -> c) -> (a -> b) -> a -> c` — composition.
    B,
}

impl BuiltinId {
    pub fn name(self) -> &'static str {
        match self {
            BuiltinId::Add => "add",
            BuiltinId::Sub => "sub",
            BuiltinId::Mul => "mul",
            BuiltinId::Div => "div",
            BuiltinId::Lt => "lt",
            BuiltinId::Eq => "eq",
            BuiltinId::Not => "not",
            BuiltinId::And => "and",
            BuiltinId::Or => "or",
            BuiltinId::If => "if",
            BuiltinId::Pair => "pair",
            BuiltinId::Fst => "fst",
            BuiltinId::Snd => "snd",
            BuiltinId::Nil => "nil",
            BuiltinId::Cons => "cons",
            BuiltinId::Fold => "fold",
            BuiltinId::Unfold => "unfold",
            BuiltinId::K => "k",
            BuiltinId::B => "b",
        }
    }

    pub fn ty(self) -> TypeScheme {
        match self {
            // Int -> Int -> Int
            BuiltinId::Add | BuiltinId::Sub | BuiltinId::Mul | BuiltinId::Div => {
                TypeScheme::mono(Ty::func_chain(&[Ty::int(), Ty::int()], Ty::int()))
            }
            // Int -> Int -> Bool
            BuiltinId::Lt | BuiltinId::Eq => {
                TypeScheme::mono(Ty::func_chain(&[Ty::int(), Ty::int()], Ty::bool()))
            }
            // Bool -> Bool
            BuiltinId::Not => TypeScheme::mono(Ty::func(Ty::bool(), Ty::bool())),
            // Bool -> Bool -> Bool
            BuiltinId::And | BuiltinId::Or => {
                TypeScheme::mono(Ty::func_chain(&[Ty::bool(), Ty::bool()], Ty::bool()))
            }
            // forall a. Bool -> a -> a -> a
            BuiltinId::If => TypeScheme::forall(
                vec![0],
                Ty::func_chain(&[Ty::bool(), Ty::Var(0), Ty::Var(0)], Ty::Var(0)),
            ),
            // forall a b. a -> b -> Pair a b
            BuiltinId::Pair => TypeScheme::forall(
                vec![0, 1],
                Ty::func_chain(
                    &[Ty::Var(0), Ty::Var(1)],
                    Ty::pair(Ty::Var(0), Ty::Var(1)),
                ),
            ),
            // forall a b. Pair a b -> a
            BuiltinId::Fst => TypeScheme::forall(
                vec![0, 1],
                Ty::func(Ty::pair(Ty::Var(0), Ty::Var(1)), Ty::Var(0)),
            ),
            // forall a b. Pair a b -> b
            BuiltinId::Snd => TypeScheme::forall(
                vec![0, 1],
                Ty::func(Ty::pair(Ty::Var(0), Ty::Var(1)), Ty::Var(1)),
            ),
            // forall a. List a
            BuiltinId::Nil => TypeScheme::forall(vec![0], Ty::list(Ty::Var(0))),
            // forall a. a -> List a -> List a
            BuiltinId::Cons => TypeScheme::forall(
                vec![0],
                Ty::func_chain(
                    &[Ty::Var(0), Ty::list(Ty::Var(0))],
                    Ty::list(Ty::Var(0)),
                ),
            ),
            // forall a b. (a -> b -> b) -> b -> List a -> b
            BuiltinId::Fold => TypeScheme::forall(
                vec![0, 1],
                Ty::func_chain(
                    &[
                        Ty::func_chain(&[Ty::Var(0), Ty::Var(1)], Ty::Var(1)),
                        Ty::Var(1),
                        Ty::list(Ty::Var(0)),
                    ],
                    Ty::Var(1),
                ),
            ),
            // forall a b. (b -> Pair (Pair a b) Bool) -> b -> List a
            BuiltinId::Unfold => TypeScheme::forall(
                vec![0, 1],
                Ty::func_chain(
                    &[
                        Ty::func(
                            Ty::Var(1),
                            Ty::pair(
                                Ty::pair(Ty::Var(0), Ty::Var(1)),
                                Ty::bool(),
                            ),
                        ),
                        Ty::Var(1),
                    ],
                    Ty::list(Ty::Var(0)),
                ),
            ),
            // forall a b. a -> b -> a
            BuiltinId::K => TypeScheme::forall(
                vec![0, 1],
                Ty::func_chain(&[Ty::Var(0), Ty::Var(1)], Ty::Var(0)),
            ),
            // forall a b c. (b -> c) -> (a -> b) -> a -> c
            BuiltinId::B => TypeScheme::forall(
                vec![0, 1, 2],
                Ty::func_chain(
                    &[
                        Ty::func(Ty::Var(1), Ty::Var(2)),
                        Ty::func(Ty::Var(0), Ty::Var(1)),
                        Ty::Var(0),
                    ],
                    Ty::Var(2),
                ),
            ),
        }
    }
}

/// All built-ins, in a stable order. Index in this slice doubles as the
/// `PrimId` if you populate a fresh library with `seed_builtin_library`.
pub const ALL_BUILTINS: &[BuiltinId] = &[
    BuiltinId::Add,
    BuiltinId::Sub,
    BuiltinId::Mul,
    BuiltinId::Div,
    BuiltinId::Lt,
    BuiltinId::Eq,
    BuiltinId::Not,
    BuiltinId::And,
    BuiltinId::Or,
    BuiltinId::If,
    BuiltinId::Pair,
    BuiltinId::Fst,
    BuiltinId::Snd,
    BuiltinId::Nil,
    BuiltinId::Cons,
    BuiltinId::Fold,
    BuiltinId::Unfold,
    BuiltinId::K,
    BuiltinId::B,
];

/// Build a fresh library populated with every built-in.
pub fn seed_builtin_library() -> Library {
    let mut lib = Library::new();
    for &b in ALL_BUILTINS {
        lib.add(Primitive {
            name: b.name().to_string(),
            ty: b.ty(),
            kind: PrimKind::Builtin(b),
        });
    }
    lib
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::Library;

    #[test]
    fn seed_lib_contains_all_builtins() {
        let lib = seed_builtin_library();
        assert_eq!(lib.len(), ALL_BUILTINS.len());
        for &b in ALL_BUILTINS {
            let _id = lib.lookup(b.name()).expect("built-in present");
        }
    }

    #[test]
    fn arity_is_correct() {
        let lib = seed_builtin_library();
        assert_eq!(lib.arity(lib.lookup("add").unwrap()), 2);
        assert_eq!(lib.arity(lib.lookup("if").unwrap()), 3);
        assert_eq!(lib.arity(lib.lookup("nil").unwrap()), 0);
        assert_eq!(lib.arity(lib.lookup("cons").unwrap()), 2);
        assert_eq!(lib.arity(lib.lookup("fold").unwrap()), 3);
        assert_eq!(lib.arity(lib.lookup("unfold").unwrap()), 2);
        assert_eq!(lib.arity(lib.lookup("pair").unwrap()), 2);
        assert_eq!(lib.arity(lib.lookup("not").unwrap()), 1);
        assert_eq!(lib.arity(lib.lookup("k").unwrap()), 2);
        assert_eq!(lib.arity(lib.lookup("b").unwrap()), 3);
    }

    #[test]
    fn nil_is_polymorphic() {
        let lib = Library::default();
        let _ = lib;
        let scheme = BuiltinId::Nil.ty();
        assert_eq!(scheme.n_vars, 1);
    }
}
