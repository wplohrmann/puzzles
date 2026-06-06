//! `graph-seek-self-play` — train via poser/searcher self-play.
//!
//! Runs the new training stack: poser-search builds dreams, q-search
//! solves them, A2C-MC trains both heads + the value head + the
//! forward-prediction head, SIGReg keeps the trunk well-distributed.
//! See `docs/09-self-play-plan.md`.

use std::env;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::Duration;

use candle_core::Device;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use lang::builtin::seed_builtin_library;

use neural::{make_optimizer, Network, NetworkCfg, Rng, SigRegCfg};
use search::{GuidedConfig, TrainingCfg};
use training::{
    train_self_play_iter, AcLossCfg, EmaBaseline, SelfPlayCfg, SelfPlayStats,
};

#[derive(Debug, Clone)]
struct Args {
    seed: u64,
    iterations: usize,
    dreams_per_iter: usize,
    examples_per_dream: usize,
    max_poser_nodes: u32,
    max_budget: u32,
    alpha: f32,
    beta: f32,
    small_floor: f32,
    lambda_sigreg: f32,
    sigreg_slices: usize,
    sigreg_quad: usize,
    top_k: usize,
    temperature: f32,
    c_h: f32,
    poser_ema_decay: f32,
    fuel: u32,
    time_budget_secs: u64,
    n: usize,
    log_every: usize,
    out: Option<PathBuf>,
    save_model: Option<PathBuf>,
    poser_stop_bias: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            seed: 0xC0DE_C0DE,
            iterations: 500,
            dreams_per_iter: 16,
            examples_per_dream: 3,
            max_poser_nodes: 6,
            max_budget: 500,
            alpha: 10.0,
            beta: 8.0,
            small_floor: 0.05,
            lambda_sigreg: 0.05,
            sigreg_slices: 1024,
            sigreg_quad: 17,
            top_k: 16,
            temperature: 1.0,
            c_h: 0.01,
            poser_ema_decay: 0.99,
            fuel: 50_000,
            time_budget_secs: 30,
            n: 32,
            log_every: 1,
            out: None,
            save_model: None,
            poser_stop_bias: 1.0,
        }
    }
}

fn parse() -> Args {
    let raw: Vec<String> = env::args().collect();
    let mut a = Args::default();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--seed" => { i += 1; a.seed = raw[i].parse().unwrap(); }
            "--iterations" => { i += 1; a.iterations = raw[i].parse().unwrap(); }
            "--dreams-per-iter" => { i += 1; a.dreams_per_iter = raw[i].parse().unwrap(); }
            "--examples" => { i += 1; a.examples_per_dream = raw[i].parse().unwrap(); }
            "--max-poser-nodes" => { i += 1; a.max_poser_nodes = raw[i].parse().unwrap(); }
            "--max-budget" => { i += 1; a.max_budget = raw[i].parse().unwrap(); }
            "--alpha" => { i += 1; a.alpha = raw[i].parse().unwrap(); }
            "--beta" => { i += 1; a.beta = raw[i].parse().unwrap(); }
            "--small-floor" => { i += 1; a.small_floor = raw[i].parse().unwrap(); }
            "--lambda-sigreg" => { i += 1; a.lambda_sigreg = raw[i].parse().unwrap(); }
            "--sigreg-slices" => { i += 1; a.sigreg_slices = raw[i].parse().unwrap(); }
            "--sigreg-quad" => { i += 1; a.sigreg_quad = raw[i].parse().unwrap(); }
            "--top-k" => { i += 1; a.top_k = raw[i].parse().unwrap(); }
            "--temperature" => { i += 1; a.temperature = raw[i].parse().unwrap(); }
            "--c-h" => { i += 1; a.c_h = raw[i].parse().unwrap(); }
            "--poser-ema-decay" => { i += 1; a.poser_ema_decay = raw[i].parse().unwrap(); }
            "--fuel" => { i += 1; a.fuel = raw[i].parse().unwrap(); }
            "--time-budget" => { i += 1; a.time_budget_secs = raw[i].parse().unwrap(); }
            "--n" => { i += 1; a.n = raw[i].parse().unwrap(); }
            "--log-every" => { i += 1; a.log_every = raw[i].parse().unwrap(); }
            "--out" => { i += 1; a.out = Some(PathBuf::from(&raw[i])); }
            "--save-model" => { i += 1; a.save_model = Some(PathBuf::from(&raw[i])); }
            "--poser-stop-bias" => { i += 1; a.poser_stop_bias = raw[i].parse().unwrap(); }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {}", other);
                print_help();
                std::process::exit(2);
            }
        }
        i += 1;
    }
    a
}

fn print_help() {
    println!(
        "graph-seek-self-play\n\
        \n\
        Training loop:\n\
        --seed N                   RNG seed (default 0xC0DE_C0DE)\n\
        --iterations N             Number of training iterations (default 500)\n\
        --dreams-per-iter N        Batch size — dreams sampled per iter (default 16)\n\
        --examples N               I/O examples per dream (default 3)\n\
        \n\
        Search budgets:\n\
        --max-poser-nodes N        Cap on poser program node count (default 6)\n\
        --max-budget N             Q-search frontier expansion ceiling (default 500)\n\
        --fuel N                   Eval fuel per primitive call (default 50000)\n\
        --time-budget SECS         Wall-clock per search (default 30)\n\
        \n\
        Reward shaping:\n\
        --alpha F                  S_sol weight in searcher cost (default 10.0)\n\
        --beta F                   Poser peak = beta * N_poser (default 8.0)\n\
        --small-floor F            Poser floor for valid programs (default 0.05)\n\
        \n\
        Policy gradient:\n\
        --top-k N                  Top-K frontier sample width (default 16)\n\
        --temperature F            Softmax temperature (default 1.0)\n\
        --c-h F                    Entropy bonus weight (default 0.01)\n\
        --poser-ema-decay F        EMA decay for poser baseline (default 0.99)\n\
        \n\
        SIGReg:\n\
        --lambda-sigreg F          SIGReg loss weight (default 0.05)\n\
        --sigreg-slices N          Random 1D projection count (default 1024)\n\
        --sigreg-quad N            Epps-Pulley quadrature points (default 17)\n\
        \n\
        Network:\n\
        --n N                      Embedding width (default 32)\n\
        \n\
        Output:\n\
        --log-every N              Print stats every N iterations (default 1)\n\
        --out PATH                 Final markdown report (default stdout only)\n\
        --save-model PATH          Save network weights at end\n"
    );
}

fn main() {
    let args = parse();
    let lib = seed_builtin_library();
    let net_cfg = NetworkCfg {
        n: args.n,
        poser_stop_bias: args.poser_stop_bias,
        ..NetworkCfg::default()
    };
    let net = Network::new(&net_cfg, &lib, Device::Cpu).expect("net build");
    let mut opt = make_optimizer(&net, net_cfg.lr, net_cfg.weight_decay).expect("optimizer");
    let mut rng = Rng::new(args.seed);
    let mut baseline = EmaBaseline::new(0.0, args.poser_ema_decay);

    let cfg = SelfPlayCfg {
        max_poser_nodes: args.max_poser_nodes,
        max_budget: args.max_budget,
        alpha: args.alpha,
        beta: args.beta,
        small_floor: args.small_floor,
        lambda_sigreg: args.lambda_sigreg,
        sigreg: SigRegCfg {
            num_slices: args.sigreg_slices,
            num_quad_points: args.sigreg_quad,
        },
        training_search: TrainingCfg {
            top_k: args.top_k,
            temperature: args.temperature,
            max_steps: args.max_budget,
        },
        ac: AcLossCfg {
            temperature: args.temperature,
            c_h: args.c_h,
        },
        guided: GuidedConfig::default(),
        dreams_per_iter: args.dreams_per_iter,
        examples_per_dream: args.examples_per_dream,
        fuel: args.fuel,
        time_budget_secs: args.time_budget_secs,
        poser_ema_decay: args.poser_ema_decay,
    };

    println!(
        "graph-seek-self-play  iterations={}  dreams_per_iter={}  N={}  max_poser_nodes={}  max_budget={}",
        args.iterations, args.dreams_per_iter, args.n,
        args.max_poser_nodes, args.max_budget,
    );

    let is_tty = std::io::stderr().is_terminal();
    let multi = MultiProgress::new();
    let outer = multi.add(ProgressBar::new(args.iterations as u64));
    outer.set_style(
        ProgressStyle::with_template(
            "Iter [{pos}/{len}] {bar:30.cyan/blue} {msg}",
        )
        .unwrap()
        .progress_chars("━━╸ "),
    );
    let inner = multi.add(ProgressBar::new_spinner());
    inner.set_style(
        ProgressStyle::with_template("  {spinner:.green} {msg}").unwrap(),
    );
    if is_tty {
        inner.enable_steady_tick(Duration::from_millis(100));
    }

    let mut all_stats: Vec<SelfPlayStats> = Vec::with_capacity(args.iterations);
    for iter in 0..args.iterations {
        outer.set_message(format!("loss=… solve_rate=…"));
        inner.set_message(format!("iter {}", iter + 1));

        let stats = match train_self_play_iter(
            iter, &net, &mut opt, &lib, &mut rng, &mut baseline, &cfg,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("iter {} failed: {}", iter, e);
                break;
            }
        };

        if args.log_every > 0 && (iter + 1) % args.log_every == 0 {
            let line = format_stats_line(&stats);
            if is_tty {
                multi.println(&line).ok();
            } else {
                println!("{}", line);
            }
        }
        outer.inc(1);
        outer.set_message(format!(
            "loss={:.3} solved={:.0}% N_poser={:.1}",
            stats.total_loss,
            100.0 * stats.fraction_solved,
            stats.mean_n_poser,
        ));
        all_stats.push(stats);
    }

    inner.finish_and_clear();
    outer.finish_with_message("done");

    let md = render_markdown(&all_stats);
    println!("\n{}", md);

    if let Some(path) = &args.out {
        let mut f = std::fs::File::create(path).expect("create out");
        f.write_all(md.as_bytes()).ok();
        println!("wrote report to {}", path.display());
    }

    if let Some(path) = &args.save_model {
        net.save(path).expect("save model");
        println!("saved model to {}", path.display());
    }
}

fn format_stats_line(s: &SelfPlayStats) -> String {
    format!(
        "iter{:5}  loss={:.3}  r_q={:.3}  r_p={:.3}  solved={:.0}%  valid={:.0}%  \
         N_poser={:.1}  S_pool={:.0}  S_sol={:.1}  fwd_mse={:.4}  \
         sigreg={:.4}  H_π={:.3}  |adv|={:.3}",
        s.iter + 1,
        s.total_loss,
        s.mean_r_searcher,
        s.mean_r_poser,
        100.0 * s.fraction_solved,
        100.0 * s.fraction_valid,
        s.mean_n_poser,
        s.mean_s_pool,
        s.mean_s_sol,
        s.forward_mse,
        s.sigreg_value,
        s.policy_entropy,
        s.mean_advantage,
    )
}

fn render_markdown(stats: &[SelfPlayStats]) -> String {
    if stats.is_empty() {
        return String::from("(no iterations recorded)\n");
    }
    let mut s = String::new();
    s.push_str("| Iter | Loss | r_q | r_p | Solved | Valid | N_poser | S_pool | fwd_mse | SIGReg | H_π |\n");
    s.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    // Show first, every 10th, and last to keep the report compact.
    let n = stats.len();
    let mut shown: Vec<usize> = (0..n).step_by((n / 30).max(1)).collect();
    if !shown.contains(&(n - 1)) { shown.push(n - 1); }
    for i in shown {
        let st = &stats[i];
        s.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.0}% | {:.0}% | {:.1} | {:.0} | {:.4} | {:.4} | {:.3} |\n",
            st.iter + 1,
            st.total_loss,
            st.mean_r_searcher,
            st.mean_r_poser,
            100.0 * st.fraction_solved,
            100.0 * st.fraction_valid,
            st.mean_n_poser,
            st.mean_s_pool,
            st.forward_mse,
            st.sigreg_value,
            st.policy_entropy,
        ));
    }
    s
}
