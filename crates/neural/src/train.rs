//! Optimizer construction.
//!
//! With the self-play training stack, the actual training-step logic
//! lives in `training::self_play::train_self_play_iter` — it assembles
//! losses across multiple heads (forward, q, value, poser) and SIGReg,
//! then does one `backward`+`opt.step`. This module only provides the
//! helper that builds the `AdamW` over the network's parameters.

use candle_core::Result;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::network::Network;

/// Build a fresh AdamW over all of the network's trainable parameters.
pub fn make_optimizer(net: &Network, lr: f64, weight_decay: f64) -> Result<AdamW> {
    let params = ParamsAdamW {
        lr,
        weight_decay,
        ..ParamsAdamW::default()
    };
    AdamW::new(net.all_vars(), params)
}
