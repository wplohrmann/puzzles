//! Neural network: program embeddings + `q(f, a | task)` scoring.
//!
//! See `docs/02-neural.md`. One head, one loss, one signal: the same
//! leaf tables and `app_net` embed program structure, runtime values, and
//! task targets into a shared `R^N` space. `q(f, a)` is a scalar produced
//! by a cross-attention pooler over the task's examples followed by an
//! MLP head.

pub mod attn;
pub mod embed;
pub mod forward_head;
pub mod heads;
pub mod network;
pub mod poser_head;
pub mod rng;
pub mod sigreg;
pub mod train;
pub mod value_head;

pub use attn::CrossAttn;
pub use embed::{
    embed_value, h_struct, h_target, h_value, signed_log1p, AppNet, EmbedCache, LeafTables,
    ListPairIds,
};
pub use forward_head::ForwardHead;
pub use heads::{PhiMlp, QHead};
pub use network::{scalar_f32, Network, NetworkCfg};
pub use poser_head::PoserHead;
pub use rng::Rng;
pub use sigreg::{sigreg_loss, SigRegCfg};
pub use train::{make_optimizer, train_step, StepStats, TrainSample};
pub use value_head::ValueHead;
