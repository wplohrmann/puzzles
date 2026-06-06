//! Bottom-up size-iterative enumeration over evaluable nodes.
//!
//! See `docs/03-search.md`. At each program size N from 1 upward,
//! every `App(f, a)` with `size(f) + size(a) + 1 == N` is constructed
//! and evaluated on the task's examples, returned as a solution if it
//! matches, otherwise added to the pool unless its `NodeId` is already
//! present. Hash-cons identity is the only structural prune; speed
//! comes from the neural prior (M4+).

mod config;
mod guided;
mod pool;
mod solve;
mod training;

pub use config::{SearchConfig, SearchResult, SearchStats};
pub use guided::{solve_guided, GuidedConfig};
pub use solve::solve;
pub use training::{
    solve_guided_training, ScoringHead, SearchMode, SolutionInfo, Trajectory,
    TrajectoryStep, TrainingCfg,
};
