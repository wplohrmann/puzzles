//! Core language: typed DAG IR, hash-consed arena, type unification, evaluator.
//!
//! See `docs/01-language.md` for the design.

pub mod arena;
pub mod builtin;
pub mod construct;
pub mod error;
pub mod eval;
pub mod ir;
pub mod library;
pub mod pretty;
pub mod serial;
pub mod ty;

pub use arena::{Arena, NodeId};
pub use builtin::BuiltinId;
pub use error::Error;
pub use eval::{eval, Value};
pub use ir::{LitValue, Node, NodeKind};
pub use library::{Library, PrimId, Primitive};
pub use ty::{Ty, TyCon, TyVar, TypeScheme};
