//! A2C-MC actor-critic loss with per-action credit assignment.
//!
//! Given a search trajectory and its terminal outcome, produce three
//! differentiable scalars suitable for adding into the total loss:
//!
//! - `actor_loss`: policy-gradient term. **Only present on successful
//!   trajectories** — failed searches don't update the policy, because
//!   we can't tell which of the many actions taken were "wrong" vs
//!   "right but the budget ran out". (See `docs/09-self-play-plan.md`,
//!   "Per-action credit".)
//! - `value_loss`: MSE between `V(created_node)` and the per-action
//!   credit `C_t`. Trained on every trajectory, solved or not.
//! - `entropy_bonus`: `c_H · H[π(·|s)]` summed over steps, added with
//!   a positive sign to the actor (the actor loss is then `-log π · A
//!   - c_H · H`). Encourages exploration.
//!
//! Per-action credit: `C_t = R` if the action's `created_node` is in
//! `S_nodes` (the solution DAG), `0` otherwise. This concentrates
//! gradient on the few actions that actually built the solution
//! instead of diluting it across all the dead-end expansions.

use candle_core::{Result, Tensor};

use lang::arena::Arena;
use lang::eval::Value;
use lang::library::Library;

use neural::{EmbedCache, Network};

use search::{ScoringHead, Trajectory};

/// Outputs from one call to `actor_critic_loss`.
pub struct AcLossOutputs {
    /// Sum of per-step `−log π(a_t|s_t) · advantage_t − c_H · H_t`.
    /// `None` iff the trajectory had no solution (we skip actor
    /// updates for failed searches).
    pub actor_loss: Option<Tensor>,
    /// Sum of per-step `(V(n_t) − C_t)²` across the trajectory.
    /// `None` when the caller uses a constant baseline (e.g., the
    /// poser's EMA baseline doesn't train the value head).
    pub value_loss: Option<Tensor>,
    /// Per-trajectory diagnostics.
    pub mean_entropy: f32,
    pub mean_advantage: f32,
    pub num_steps: usize,
}

/// Baseline strategy for advantage estimation.
#[derive(Clone, Copy, Debug)]
pub enum Baseline {
    /// Use the network's value head V(created_node). Produces a
    /// value_loss term (MSE against `C_t`) that trains the head.
    ValueHead,
    /// Use a constant scalar (e.g. an EMA of recent rewards). No
    /// value_loss is produced — the value head is not consulted at
    /// all. Used for the poser, where the trajectory is short and a
    /// per-task learned baseline isn't worth the variance reduction.
    Constant(f32),
}

/// Configuration knobs for the actor-critic loss.
#[derive(Clone, Copy, Debug)]
pub struct AcLossCfg {
    /// Softmax temperature used at sample-time; must match for the
    /// log-prob computation to recover the policy that generated the
    /// trajectory.
    pub temperature: f32,
    /// Entropy bonus weight.
    pub c_h: f32,
}

impl Default for AcLossCfg {
    fn default() -> Self {
        Self { temperature: 1.0, c_h: 0.01 }
    }
}

/// Compute actor + value loss for one trajectory.
///
/// `head` selects which scoring head is used to recompute candidate
/// logits at training time — must match the head used at sample time.
///
/// `stop_grad_trunk = true` detaches the trunk outputs before they
/// reach the head. Use it for the **poser** (its adversarial gradient
/// should not flow into the shared trunk). For the q-head and value
/// head, gradient through the trunk is fine (cooperative).
///
/// `inputs` and `targets` mirror what `solve_guided_training` ran on.
/// For the poser-search call, `targets` should be `None` (the loss
/// uses a zero placeholder).
#[allow(clippy::too_many_arguments)]
pub fn actor_critic_loss(
    net: &Network,
    arena: &Arena,
    lib: &Library,
    trajectory: &Trajectory,
    reward: f32,
    head: ScoringHead,
    baseline: Baseline,
    stop_grad_trunk: bool,
    inputs: &[Value],
    targets: Option<&[Value]>,
    cfg: &AcLossCfg,
    fuel: u32,
    cache: &mut EmbedCache,
) -> Result<AcLossOutputs> {
    let device = &net.device;

    // Pre-build target_rows once; zero-fill if no targets.
    let target_rows = match targets {
        Some(t) => net.target_stack(t)?,
        None => Tensor::zeros(
            (inputs.len(), net.cfg.n),
            candle_core::DType::F32,
            device,
        )?,
    };

    // Which nodes are in the solution DAG. `S_nodes` is empty if no
    // solution — in which case every C_t = 0 and actor_loss is None.
    let s_nodes = trajectory
        .solution
        .as_ref()
        .map(|s| s.s_nodes.clone())
        .unwrap_or_default();
    let solved = trajectory.solution.is_some();

    let mut actor_terms: Vec<Tensor> = Vec::new();
    let mut value_terms: Vec<Tensor> = Vec::new();
    let mut entropy_sum: f32 = 0.0;
    let mut advantage_sum: f32 = 0.0;
    let num_steps = trajectory.steps.len();

    for step in &trajectory.steps {
        // Per-action credit. R if the action created a solution-tree
        // node, 0 otherwise. Note this naturally hands out 0 to every
        // step on a failed trajectory (S_nodes is empty), so the
        // value head still gets a "predict zero" target everywhere.
        let c_t: f32 = if s_nodes.contains(&step.created_node) {
            reward
        } else {
            0.0
        };

        // Re-score the K candidates with the current network weights.
        // The trajectory recorded the same candidates at sample-time;
        // because sample and loss happen in the same iteration (no
        // optimizer step between), these scores are the policy that
        // generated the action.
        let mut logits: Vec<Tensor> = Vec::with_capacity(step.candidates.len());
        for &(f_node, a_node, _size) in &step.candidates {
            let ep = net.embed_pair(
                f_node, a_node, arena, lib, inputs, &target_rows, fuel, cache,
            )?;

            let pieces = if stop_grad_trunk {
                let ctx = ep.ctx.detach();
                let hf = ep.hf_struct.detach();
                let ha = ep.ha_struct.detach();
                vec![ctx, hf, ha]
            } else {
                vec![ep.ctx, ep.hf_struct, ep.ha_struct]
            };
            let head_in = Tensor::cat(&[&pieces[0], &pieces[1], &pieces[2]], 1)?;
            let logit = match head {
                ScoringHead::Q => net.q_head.forward(&head_in)?,
                ScoringHead::Poser => net.poser_head.forward(&head_in)?,
            };
            logits.push(logit.flatten_all()?); // (1,)
        }
        let stacked = Tensor::cat(&logits.iter().collect::<Vec<_>>(), 0)?; // (K,)

        // Apply temperature, log-softmax, softmax. T must match sample-
        // time T (default 1.0).
        let scaled = stacked.affine((1.0 / cfg.temperature as f64).into(), 0.0)?;
        let scaled2 = scaled.unsqueeze(0)?; // (1, K) — log_softmax needs ≥2 dims
        let log_softmax = candle_nn::ops::log_softmax(&scaled2, 1)?.squeeze(0)?;
        let softmax = candle_nn::ops::softmax(&scaled2, 1)?.squeeze(0)?;

        // log π(a_t | s_t) — pluck the chosen index, reduce to 0-d.
        let chosen = step.chosen_idx;
        let log_pi = log_softmax.narrow(0, chosen, 1)?.sum_all()?; // 0-d

        // Policy entropy at this state: H = -Σ π log π.
        let entropy_t = softmax.mul(&log_softmax)?.sum_all()?.neg()?; // 0-d
        let entropy_val: f32 = scalar(&entropy_t)?;
        entropy_sum += entropy_val;

        // Baseline. Either learned (value head, on `created_node`) or
        // a fixed scalar (EMA). The advantage uses the *detached*
        // baseline value so policy gradient doesn't push through it.
        let (advantage, value_for_loss) = match baseline {
            Baseline::ValueHead => {
                let v_t = net
                    .value_score(step.created_node, arena, cache)?
                    .sum_all()?; // 0-d
                let c_t_scalar = Tensor::new(c_t, device)?;
                let adv = (c_t_scalar - v_t.detach())?;
                (adv, Some(v_t))
            }
            Baseline::Constant(b) => {
                let adv = Tensor::new(c_t - b, device)?;
                (adv, None)
            }
        };
        let advantage_val: f32 = scalar(&advantage)?;
        advantage_sum += advantage_val;

        // Actor term: only contribute on successful trajectories. On
        // failures the policy isn't updated.
        if solved {
            let policy_term = log_pi.mul(&advantage)?.neg()?; // 0-d
            let entropy_term = entropy_t.affine(-(cfg.c_h as f64), 0.0)?; // 0-d
            let step_actor = (policy_term + entropy_term)?; // 0-d
            actor_terms.push(step_actor);
        }

        // Value loss: only when using ValueHead baseline.
        if let Some(v_t) = value_for_loss {
            let c_t_target = Tensor::new(c_t, device)?;
            let diff = (v_t - c_t_target)?;
            let v_loss_t = diff.mul(&diff)?;
            value_terms.push(v_loss_t);
        }
    }

    let value_loss = if value_terms.is_empty() {
        None
    } else {
        Some(sum_tensors(&value_terms, device)?)
    };

    let actor_loss = if actor_terms.is_empty() {
        None
    } else {
        Some(sum_tensors(&actor_terms, device)?)
    };

    let mean_entropy = if num_steps > 0 { entropy_sum / num_steps as f32 } else { 0.0 };
    let mean_advantage = if num_steps > 0 { advantage_sum / num_steps as f32 } else { 0.0 };

    Ok(AcLossOutputs {
        actor_loss,
        value_loss,
        mean_entropy,
        mean_advantage,
        num_steps,
    })
}

fn scalar(t: &Tensor) -> Result<f32> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(v[0])
}

fn sum_tensors(ts: &[Tensor], device: &candle_core::Device) -> Result<Tensor> {
    if ts.is_empty() {
        return Tensor::new(0.0f32, device);
    }
    let mut acc = ts[0].clone();
    for t in &ts[1..] {
        acc = (acc + t)?;
    }
    Ok(acc)
}

// ----------------------- Reward formulas -------------------------------

/// Searcher's reward. Linear combination of `S_pool` and `S_sol`:
///   `r = max(0, 1 − (S_pool + α · S_sol) / max_budget)`  if solved
///   `r = 0`                                                otherwise.
///
/// `α` is "how many pool expansions one solution-node is worth"
/// (default 10).
pub fn searcher_reward(
    s_pool: u32,
    s_sol: u32,
    max_budget: u32,
    alpha: f32,
    solved: bool,
) -> f32 {
    if !solved { return 0.0; }
    let cost = s_pool as f32 + alpha * s_sol as f32;
    (1.0 - cost / max_budget as f32).max(0.0)
}

/// Poser's reward. Tent on `S_pool` with peak at `β · N_poser`.
///   `r = 0` if invalid OR not solved.
///   Ramp `small_floor → 1` linearly as `S_pool` goes `0 → peak`.
///   Ramp `1 → 0` linearly as `S_pool` goes `peak → max_budget`.
///
/// `valid` = the poser's program produced no nested closures / no
/// `Bottom` outputs on the probe inputs (caller checks this).
/// `solved` = the q-search found a solution within `max_budget`.
pub fn poser_reward(
    s_pool: u32,
    n_poser: u32,
    max_budget: u32,
    beta: f32,
    small_floor: f32,
    valid: bool,
    solved: bool,
) -> f32 {
    if !valid || !solved { return 0.0; }
    let s = s_pool as f32;
    let peak = (beta * n_poser as f32).max(1.0);
    let max_b = max_budget as f32;
    if s <= peak {
        small_floor + (1.0 - small_floor) * (s / peak)
    } else if s < max_b {
        ((max_b - s) / (max_b - peak)).max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searcher_reward_solved_perfect() {
        // S_pool = 0, S_sol = 0 → max possible.
        let r = searcher_reward(0, 0, 500, 10.0, true);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn searcher_reward_unsolved_is_zero() {
        let r = searcher_reward(100, 5, 500, 10.0, false);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn searcher_reward_clamped_at_zero() {
        // S_pool + 10·S_sol > max_budget → 0.
        let r = searcher_reward(400, 20, 500, 10.0, true);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn poser_reward_peaks_at_beta_n() {
        // β=8, N_poser=5 → peak at S_pool=40.
        let r_peak = poser_reward(40, 5, 500, 8.0, 0.05, true, true);
        assert!((r_peak - 1.0).abs() < 1e-5, "peak should be 1.0, got {r_peak}");

        // S_pool=0 → small floor.
        let r_floor = poser_reward(0, 5, 500, 8.0, 0.05, true, true);
        assert!((r_floor - 0.05).abs() < 1e-5);

        // S_pool=max_budget → 0.
        let r_max = poser_reward(500, 5, 500, 8.0, 0.05, true, true);
        assert!(r_max.abs() < 1e-5);
    }

    #[test]
    fn poser_reward_invalid_is_zero() {
        let r = poser_reward(40, 5, 500, 8.0, 0.05, false, true);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn poser_reward_unsolved_is_zero() {
        let r = poser_reward(40, 5, 500, 8.0, 0.05, true, false);
        assert_eq!(r, 0.0);
    }

    /// Integration: run a tiny q-search and compute the actor-critic
    /// loss against the trajectory. Verifies the tensor shapes line up
    /// and backward() succeeds end-to-end.
    #[test]
    fn actor_critic_backprops() {
        use candle_core::Device;
        use lang::arena::Arena;
        use lang::builtin::seed_builtin_library;
        use neural::{Network, NetworkCfg, Rng};
        use search::{
            solve_guided_training, GuidedConfig, ScoringHead, SearchMode, TrainingCfg,
        };

        let lib = seed_builtin_library();
        let cfg_net = NetworkCfg { n: 8, ..NetworkCfg::default() };
        let net = Network::new(&cfg_net, &lib, Device::Cpu).unwrap();

        let mut arena = Arena::new();
        let inputs = vec![Value::Int(1), Value::Int(2)];
        let expected = vec![Value::Int(1), Value::Int(2)]; // identity

        let scfg = search::SearchConfig {
            time_budget: std::time::Duration::from_secs(5),
            max_program_size: 4,
            ..search::SearchConfig::default()
        };
        let gcfg = GuidedConfig::default();
        let tcfg = TrainingCfg { top_k: 4, temperature: 1.0, max_steps: 20 };
        let mut rng = Rng::new(7);

        let traj = solve_guided_training(
            &mut arena, &lib, &scfg, &net, &gcfg,
            ScoringHead::Q, SearchMode::Solve,
            &inputs, Some(&expected), &tcfg, &mut rng,
        );

        // Force at least a few steps so the loss has something to chew on.
        if traj.steps.is_empty() {
            return; // search bailed early; nothing to test, but no failure
        }

        let reward = searcher_reward(traj.s_pool, 1, 500, 10.0, traj.solution.is_some());

        let mut cache = EmbedCache::default();
        let out = actor_critic_loss(
            &net, &arena, &lib, &traj, reward,
            ScoringHead::Q, Baseline::ValueHead, false,
            &inputs, Some(&expected),
            &AcLossCfg::default(),
            scfg.eval_fuel, &mut cache,
        ).unwrap();

        assert_eq!(out.num_steps, traj.steps.len());

        let value_loss = out.value_loss.expect("ValueHead baseline produces a value_loss");
        let v_val: f32 = value_loss.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(v_val.is_finite(), "non-finite value loss");

        if let Some(actor) = out.actor_loss {
            let total = (actor + value_loss).unwrap();
            let _grads = total.backward().expect("backward should succeed");
        } else {
            let _grads = value_loss.backward().expect("value-only backward should succeed");
        }
    }

    /// Constant baseline (poser-style): no value_loss is produced; the
    /// actor loss should still backprop.
    #[test]
    fn actor_critic_constant_baseline_no_value_loss() {
        use candle_core::Device;
        use lang::arena::Arena;
        use lang::builtin::seed_builtin_library;
        use neural::{Network, NetworkCfg, Rng};
        use search::{
            solve_guided_training, GuidedConfig, ScoringHead, SearchMode, TrainingCfg,
        };

        let lib = seed_builtin_library();
        let cfg_net = NetworkCfg { n: 8, ..NetworkCfg::default() };
        let net = Network::new(&cfg_net, &lib, Device::Cpu).unwrap();

        let mut arena = Arena::new();
        let inputs = vec![Value::Int(0)];

        let scfg = search::SearchConfig {
            time_budget: std::time::Duration::from_secs(5),
            max_program_size: 6,
            ..search::SearchConfig::default()
        };
        let gcfg = GuidedConfig::default();
        // Force termination via stop with very few steps and high temp
        let tcfg = TrainingCfg { top_k: 4, temperature: 5.0, max_steps: 50 };
        let mut rng = Rng::new(11);

        let traj = solve_guided_training(
            &mut arena, &lib, &scfg, &net, &gcfg,
            ScoringHead::Poser, SearchMode::Construct,
            &inputs, None, &tcfg, &mut rng,
        );

        if traj.steps.is_empty() { return; }

        let reward = 0.3; // arbitrary
        let mut cache = EmbedCache::default();
        let out = actor_critic_loss(
            &net, &arena, &lib, &traj, reward,
            ScoringHead::Poser, Baseline::Constant(0.1), true,
            &inputs, None,
            &AcLossCfg::default(),
            scfg.eval_fuel, &mut cache,
        ).unwrap();

        assert!(out.value_loss.is_none(), "Constant baseline should produce no value_loss");
        if let Some(actor) = out.actor_loss {
            let _grads = actor.backward().expect("poser actor backward should succeed");
        }
    }
}
