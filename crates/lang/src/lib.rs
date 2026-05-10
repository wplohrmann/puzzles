//! Core language: untyped DAG IR, hash-consed arena, evaluator.
//!
//! See `docs/01-language.md` for the design. There is no static type
//! system: nodes carry only structural information, and runtime type
//! mismatches surface as `Value::Bottom`.

pub mod arena;
pub mod builtin;
pub mod construct;
pub mod error;
pub mod eval;
pub mod ir;
pub mod library;
pub mod pretty;
pub mod serial;

pub use arena::{Arena, NodeId};
pub use builtin::BuiltinId;
pub use error::Error;
pub use eval::{eval, Value};
pub use ir::{LitValue, Node, NodeKind};
pub use library::{Library, PrimId, Primitive};
