//! Self-play training loop.
//!
//! See `docs/09-self-play-plan.md`. A poser-search constructs dreams,
//! a q-search solves them, and an A2C-MC actor-critic plus a
//! forward-prediction auxiliary plus a SIGReg distributional
//! regularizer train the shared trunk and four heads.

pub mod actor_critic;
pub mod dream;
pub mod eval;
pub mod self_play;

pub use actor_critic::{
    actor_critic_loss, poser_reward, searcher_reward, AcLossCfg, AcLossOutputs, Baseline,
};
pub use dream::{sample_dream, sample_program, DreamCfg, DreamTask};
pub use eval::{bench_tasks, evaluate, BenchOutcome};
pub use self_play::{train_self_play_iter, EmaBaseline, SelfPlayCfg, SelfPlayStats};
