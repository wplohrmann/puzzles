//! Built-in primitives: their identity, arity, and a helper to seed a Library.
//!
//! Without a static type system, primitives carry only an `arity`. Type
//! mismatches at runtime (e.g. `add` applied to a `Bool`) surface as
//! `Value::Bottom` from the evaluator.

use serde::{Deserialize, Serialize};

use crate::library::{Library, PrimKind, Primitive};

/// Built-in primitive identifier. Kept stable so serialised programs from
/// older versions still resolve correctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuiltinId {
    // Numeric
    Add, Sub, Mul, Div,
    Lt, Eq,

    // Boolean
    Not, And, Or,
    /// `if cond then else` — lazy in the unchosen branch.
    If,

    // Pair
    Pair, Fst, Snd,

    // List
    Nil, Cons,
    /// Right-fold: `fold f z [a,b,c] = f a (f b (f c z))`.
    Fold,
    /// `unfold step seed` — generates a list by repeatedly calling
    /// `step` on the running seed; stops when the second component of
    /// the returned pair is `false`.
    Unfold,

    // Combinators
    /// `K x y = x`
    K,
    /// `B f g x = f (g x)`
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

    pub fn arity(self) -> u8 {
        match self {
            BuiltinId::Add | BuiltinId::Sub | BuiltinId::Mul | BuiltinId::Div => 2,
            BuiltinId::Lt | BuiltinId::Eq => 2,
            BuiltinId::Not => 1,
            BuiltinId::And | BuiltinId::Or => 2,
            BuiltinId::If => 3,
            BuiltinId::Pair => 2,
            BuiltinId::Fst | BuiltinId::Snd => 1,
            BuiltinId::Nil => 0,
            BuiltinId::Cons => 2,
            BuiltinId::Fold => 3,
            BuiltinId::Unfold => 2,
            BuiltinId::K => 2,
            BuiltinId::B => 3,
        }
    }
}

/// All built-ins, in a stable order.
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
            arity: b.arity(),
            kind: PrimKind::Builtin(b),
        });
    }
    lib
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
