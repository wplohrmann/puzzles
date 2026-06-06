//! Wake/sleep training loop.
//!
//! See `docs/06-training.md`. M4 ships dream-only training (no wake
//! phase): the network is trained from synthetic programs sampled from
//! the seed-library prior, with a curriculum that ramps program size
//! from 1 → 13 over many iterations.

pub mod actor_critic;
pub mod complexity;
pub mod curriculum;
pub mod dream;
pub mod eval;
pub mod self_play;
pub mod trajectory;
pub mod train;

pub use actor_critic::{
    actor_critic_loss, poser_reward, searcher_reward, AcLossCfg, AcLossOutputs, Baseline,
};
pub use self_play::{train_self_play_iter, EmaBaseline, SelfPlayCfg, SelfPlayStats};
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
