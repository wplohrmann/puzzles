//! `graph-seek-complexity` — train the network on dreams of increasing
//! complexity (number of `App` nodes), then evaluate by running guided
//! search to check whether the network can recover programs whose
//! outputs match the target examples.
//!
//! Writes a markdown table summarising performance per complexity
//! level; useful for verifying the network learns end-to-end.

use std::env;
use std::fs::File;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::Duration;

use candle_core::Device;
use candle_nn::AdamW;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use lang::builtin::seed_builtin_library;
use lang::ir::LitValue;

use neural::{make_optimizer, Network, NetworkCfg, Rng};

use training::{
    train_complexity_curriculum, ComplexityCfg, CurriculumEvent,
};

#[derive(Debug, Clone)]
struct Args {
    seed: u64,
    levels: Vec<usize>,
    min_iters_per_level: usize,
    log_every: usize,
    lcb_iter_window: usize,
    lcb_z: f32,
    initial_top1: f32,
    top1_shrink: f32,
    target_solve: f32,
    dreams_per_iter: usize,
    eval_dreams_per_level: usize,
    eval_budget_secs: u64,
    out: Option<PathBuf>,
    examples_per_dream: usize,
    n: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            seed: 0xC0DE_C0DE,
            levels: vec![1, 2, 3, 4, 5, 6],
            min_iters_per_level: 10,
            log_every: 25,
            lcb_iter_window: 5,
            lcb_z: 1.645,
            initial_top1: 0.95,
            top1_shrink: 0.8,
            target_solve: 0.50,
            dreams_per_iter: 6,
            eval_dreams_per_level: 20,
            eval_budget_secs: 6,
            out: None,
            examples_per_dream: 3,
            n: 32,
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
            "--levels" => {
                i += 1;
                a.levels = raw[i].split(',').map(|s| s.parse().unwrap()).collect();
            }
            "--min-iters" => { i += 1; a.min_iters_per_level = raw[i].parse().unwrap(); }
            "--log-every" => { i += 1; a.log_every = raw[i].parse().unwrap(); }
            "--lcb-iter-window" => { i += 1; a.lcb_iter_window = raw[i].parse().unwrap(); }
            "--lcb-z" => { i += 1; a.lcb_z = raw[i].parse().unwrap(); }
            "--initial-top1" => { i += 1; a.initial_top1 = raw[i].parse().unwrap(); }
            "--top1-shrink" => { i += 1; a.top1_shrink = raw[i].parse().unwrap(); }
            "--target-solve" => { i += 1; a.target_solve = raw[i].parse().unwrap(); }
            "--dreams-per-iter" => { i += 1; a.dreams_per_iter = raw[i].parse().unwrap(); }
            "--eval-dreams" => { i += 1; a.eval_dreams_per_level = raw[i].parse().unwrap(); }
            "--eval-budget" => { i += 1; a.eval_budget_secs = raw[i].parse().unwrap(); }
            "--examples" => { i += 1; a.examples_per_dream = raw[i].parse().unwrap(); }
            "--n" => { i += 1; a.n = raw[i].parse().unwrap(); }
            "--out" => { i += 1; a.out = Some(PathBuf::from(&raw[i])); }
            "--help" | "-h" => {
                println!("graph-seek-complexity\n\
                    --seed N\n\
                    --levels CSV  (default 1,2,3,4,5,6)\n\
                    --min-iters N (default 10)  minimum train iters per training phase\n\
                    --log-every N (default 25)  emit a stderr iter summary every N iters when non-TTY\n\
                    --lcb-iter-window N (default 5)  recent iters pooled into the LCB test\n\
                    --lcb-z F (default 1.645)  Wilson LCB z (1.645=95%, 2.326=99%)\n\
                    --initial-top1 F (default 0.95)  starting top1@max target\n\
                    --top1-shrink F (default 0.8)  fraction of error remaining after each bump\n\
                    --target-solve F (default 0.50)  search-eval solve-rate that advances the level\n\
                    --dreams-per-iter N (default 6)\n\
                    --eval-dreams N (default 20)\n\
                    --eval-budget SECS (default 6)\n\
                    --examples N (default 3)\n\
                    --n N  embedding width (default 32)\n\
                    --out PATH  markdown output (default stdout)\n\n\
                    Each level: train until windowed top1@max ≥ current\n\
                    top1 target, then run search-eval. If solve rate ≥\n\
                    --target-solve, advance. Otherwise bump top1 target\n\
                    via next = 1 − shrink·(1 − current) and train more.\n\
                    The loop is open-ended — a level the network cannot\n\
                    reach will block indefinitely; Ctrl-C to intervene.\n");
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown arg: {}", raw[i]);
                std::process::exit(2);
            }
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse();
    let net_cfg = NetworkCfg { n: args.n, ..NetworkCfg::default() };
    let lib = seed_builtin_library();
    let net = Network::new(&net_cfg, &lib, Device::Cpu).expect("net");
    let mut opt: AdamW = make_optimizer(&net, net_cfg.lr, net_cfg.weight_decay).expect("opt");
    let mut rng = Rng::new(args.seed);

    let cfg = ComplexityCfg {
        levels: args.levels.clone(),
        examples_per_dream: args.examples_per_dream,
        dreams_per_iter: args.dreams_per_iter,
        min_iters_per_level: args.min_iters_per_level,
        top1_lcb_iter_window: args.lcb_iter_window,
        top1_lcb_z: args.lcb_z,
        initial_top1_target: args.initial_top1,
        top1_shrink_factor: args.top1_shrink,
        target_solve_rate: args.target_solve,
        eval_dreams_per_level: args.eval_dreams_per_level,
        eval_search_budget: Duration::from_secs(args.eval_budget_secs),
        literal_seeds: vec![
            LitValue::Int(-3), LitValue::Int(-2), LitValue::Int(-1),
            LitValue::Int(0), LitValue::Int(1), LitValue::Int(2), LitValue::Int(3),
        ],
        ..ComplexityCfg::default()
    };

    println!(
        "graph-seek-complexity: levels={:?} initial_top1={:.2} target_solve={:.2} \
         shrink={:.2} min_iters={} lcb_window={} lcb_z={:.3} dreams_per_iter={} eval_dreams={}",
        cfg.levels,
        cfg.initial_top1_target, cfg.target_solve_rate, cfg.top1_shrink_factor,
        cfg.min_iters_per_level, cfg.top1_lcb_iter_window, cfg.top1_lcb_z,
        cfg.dreams_per_iter, cfg.eval_dreams_per_level,
    );

    let is_tty = std::io::stderr().is_terminal();
    let multi = MultiProgress::new();
    let outer = multi.add(ProgressBar::new(cfg.levels.len() as u64));
    outer.set_style(
        ProgressStyle::with_template(
            "Curriculum [{pos}/{len}] {bar:30.cyan/blue} {msg}",
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
    outer.set_message("starting…");

    // In non-TTY mode (output captured to a file or pipe), indicatif
    // silently suppresses the bars *and* `multi.println` — so we'd lose
    // the eval milestones. Fall back to plain stderr prints when no
    // terminal is attached. Iter-level chatter is dropped (no one wants
    // hundreds of lines per second in a log file).
    let log_milestone = |msg: String| {
        if is_tty {
            multi.println(&msg).ok();
        } else {
            eprintln!("{msg}");
        }
    };

    let report = train_complexity_curriculum(
        &net, &mut opt, &lib, &mut rng, &cfg,
        |event| match event {
            CurriculumEvent::Iter {
                level, iter, loss, top1_overall, top1_at_max, top1_target,
            } => {
                outer.set_message(format!("L{}", level));
                let at_max_disp = top1_at_max
                    .map(|t| format!("{:.2}", t))
                    .unwrap_or_else(|| " -- ".to_string());
                let line = format!(
                    "L{} train it{:5}  loss={:.3}  top1@max={}/{:.2}  overall={:.2}",
                    level, iter + 1, loss, at_max_disp, top1_target, top1_overall,
                );
                if is_tty {
                    inner.set_message(line);
                } else if args.log_every > 0 && (iter + 1) % args.log_every == 0 {
                    eprintln!("  {line}");
                }
            }
            CurriculumEvent::EvalStart { level, round } => {
                let line = format!(
                    "L{} evaluating round {} (search-eval, may take a while)…",
                    level, round,
                );
                if is_tty {
                    inner.set_message(line);
                } else {
                    eprintln!("  {line}");
                }
            }
            CurriculumEvent::Eval {
                level, round, measured_top1, eval_total, eval_solved,
                solve_rate, target_solve_rate, next_top1_target,
            } => match next_top1_target {
                None => {
                    log_milestone(format!(
                        "✓ L{} eval#{}  top1@max={:.2}  solved={}/{} ({:.0}%) ≥ {:.0}%  → advancing",
                        level, round, measured_top1, eval_solved, eval_total,
                        100.0 * solve_rate, 100.0 * target_solve_rate,
                    ));
                    outer.inc(1);
                }
                Some(next) => {
                    log_milestone(format!(
                        "  L{} eval#{}  top1@max={:.2}  solved={}/{} ({:.0}%) < {:.0}%  → target → {:.2}",
                        level, round, measured_top1, eval_solved, eval_total,
                        100.0 * solve_rate, 100.0 * target_solve_rate, next,
                    ));
                }
            },
        },
    );

    inner.finish_and_clear();
    outer.finish_with_message("done");

    let md = report.render_markdown();
    println!("\n{}", md);
    if let Some(p) = args.out {
        let mut f = File::create(&p).expect("create out");
        f.write_all(md.as_bytes()).ok();
        println!("wrote table to {}", p.display());
    }
}
