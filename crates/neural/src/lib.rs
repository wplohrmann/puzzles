//! Neural network: program embeddings + policy/value heads.
//!
//! See `docs/02-neural.md` and `decisions/02-pure-rust-nn.md`. Pure-Rust
//! hand-rolled MLP + Adam; structural recurrence stubbed for v0
//! (`decisions/03-value-only-features.md`).

pub mod feat;
pub mod mlp;
pub mod network;
pub mod rng;

pub use feat::{
    aggregate_examples, value_features, value_pair_features,
    PAIR_FEAT_DIM, SOLO_FEAT_DIM,
};
pub use mlp::{adam_step, sigmoid_bce, softmax_xent, AdamCfg, Linear, Mlp};
pub use network::{
    cand_features, state_features, task_features, train_step, Network, NetworkCfg,
    PolicySample, StepStats, ValueSample,
    CAND_FEAT_DIM, POLICY_IN_DIM, STATE_DIM, TASK_FEAT_DIM, VALUE_IN_DIM,
};
pub use rng::Rng;
