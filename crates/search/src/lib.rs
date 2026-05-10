//! Search: bottom-up size-iterative enumeration over evaluable nodes.
//!
//! See `docs/03-search.md`. M2 ships the no-NN, uniform-typed-prior
//! enumerator: at each program size N from 1 upward, every admissible
//! `Apply(f, a)` with `size(f) + size(a) + 1 == N` is constructed, evaluated
//! on the task's examples, and either (a) returned as a solution or (b)
//! added to the pool unless observational-equivalence dedup eliminates it.
//!
//! The priority-queue beam (with non-uniform priors) lands in M4 once a
//! neural policy exists.

mod config;
mod pool;
mod solve;
mod value_hash;

pub use config::{SearchConfig, SearchResult, SearchStats};
pub use solve::solve;
