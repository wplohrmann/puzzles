//! Wake/sleep training loop.
//!
//! See `docs/06-training.md`. M4 ships dream-only training (no wake
//! phase): the network is trained from synthetic programs sampled from
//! the seed-library prior, with a curriculum that ramps program size
//! from 1 → 13 over many iterations.

pub mod complexity;
pub mod curriculum;
pub mod dream;
pub mod eval;
pub mod trajectory;
pub mod train;

pub use complexity::{
    eval_complexity, next_top1_target, sample_complexity_dreams, wilson_lcb,
    ComplexityCfg, ComplexityReport, ComplexityRow, CurriculumEvent,
    train_complexity_curriculum,
};
pub use curriculum::{Curriculum, CurriculumStage};
pub use dream::{sample_dream, sample_program, DreamCfg, DreamTask};
pub use eval::{bench_tasks, evaluate, BenchOutcome};
pub use trajectory::{dream_to_samples, DreamSamples, MAX_NEGATIVES_PER_STEP};
pub use train::{fresh_network, train_one_iter, IterMetrics, TrainCfg};
