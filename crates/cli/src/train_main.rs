//! `graph-seek-train` — runs the dream-only training loop and
//! periodically evaluates against the search-benchmark.
//!
//! Usage:
//!   graph-seek-train [--iters N] [--dreams-per-iter K] [--eval-every E]
//!                    [--metrics PATH] [--checkpoint PATH] [--seed S]
//!
//! The defaults are tuned for the M4 acceptance criterion: solve every
//! task in `crates/search/benches/trivial_list.rs` (identity / sum / head
//! / length / add-one-to-each).

use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use neural::Network;

use training::{evaluate, fresh_network, train_one_iter, TrainCfg};

#[derive(Debug, Clone)]
struct Args {
    iters: usize,
    dreams_per_iter: usize,
    eval_every: usize,
    metrics_path: Option<PathBuf>,
    checkpoint_path: Option<PathBuf>,
    resume_from: Option<PathBuf>,
    seed: u64,
    eval_unguided: bool,
    final_eval_only: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            iters: 60,
            dreams_per_iter: 32,
            eval_every: 10,
            metrics_path: None,
            checkpoint_path: None,
            resume_from: None,
            seed: 0xc0ffee_face,
            eval_unguided: false,
            final_eval_only: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let raw: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--iters" => {
                i += 1; args.iters = raw[i].parse().expect("iters: integer");
            }
            "--dreams-per-iter" => {
                i += 1; args.dreams_per_iter = raw[i].parse().expect("dreams-per-iter: integer");
            }
            "--eval-every" => {
                i += 1; args.eval_every = raw[i].parse().expect("eval-every: integer");
            }
            "--metrics" => {
                i += 1; args.metrics_path = Some(PathBuf::from(&raw[i]));
            }
            "--checkpoint" => {
                i += 1; args.checkpoint_path = Some(PathBuf::from(&raw[i]));
            }
            "--resume" => {
                i += 1; args.resume_from = Some(PathBuf::from(&raw[i]));
            }
            "--seed" => {
                i += 1; args.seed = raw[i].parse().expect("seed: integer");
            }
            "--eval-unguided" => { args.eval_unguided = true; }
            "--final-eval-only" => { args.final_eval_only = true; }
            "--help" | "-h" => {
                println!(
                    "graph-seek-train\n\
                     \n\
                     Flags:\n\
                       --iters N              training iterations (default 60)\n\
                       --dreams-per-iter K    dreams sampled per iter (default 32)\n\
                       --eval-every E         eval period in iters (default 10)\n\
                       --metrics PATH         write per-iter metrics jsonl\n\
                       --checkpoint PATH      save final network weights\n\
                       --seed S               PRNG seed (default 0xc0ffeeface)\n\
                       --eval-unguided        also report the unguided baseline\n\
                       --final-eval-only      skip per-iter evals; only eval at end\n"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(2);
            }
        }
        i += 1;
    }
    args
}

fn main() {
    let args = parse_args();

    let cfg = TrainCfg {
        iterations: args.iters,
        dreams_per_iter: args.dreams_per_iter,
        seed: args.seed,
        ..TrainCfg::default()
    };
    let (mut net, lib, mut rng) = fresh_network(&cfg);
    if let Some(p) = args.resume_from.as_ref() {
        let f = File::open(p).expect("open resume checkpoint");
        let mut s = String::new();
        BufReader::new(f).read_to_string(&mut s).expect("read");
        let mut loaded: Network = serde_json::from_str(&s).expect("parse");
        loaded.rehydrate_scratch();
        net = loaded;
        println!("resumed from {}", p.display());
    }

    let mut metrics_writer = args.metrics_path.as_ref().map(|p| {
        BufWriter::new(File::create(p).expect("create metrics file"))
    });

    println!(
        "graph-seek-train: iters={} dreams_per_iter={} eval_every={} seed={}",
        args.iters, args.dreams_per_iter, args.eval_every, args.seed,
    );

    for it in 0..args.iters {
        let m = train_one_iter(it, &cfg, &mut rng, &mut net, &lib);
        println!(
            "iter {:3} max_size={} dreams={:3} samples={:5} p_loss={:.4} v_loss={:.4}",
            m.iter, m.max_size, m.dreams_used, m.policy_samples, m.policy_loss, m.value_loss,
        );
        io::stdout().flush().ok();
        if let Some(w) = metrics_writer.as_mut() {
            let entry = serde_json::json!({
                "kind": "train",
                "iter": m.iter,
                "max_size": m.max_size,
                "dreams_used": m.dreams_used,
                "policy_samples": m.policy_samples,
                "value_samples": m.value_samples,
                "policy_loss": m.policy_loss,
                "value_loss": m.value_loss,
            });
            writeln!(w, "{}", entry).ok();
        }

        if !args.final_eval_only && (it + 1) % args.eval_every == 0 {
            let outcomes = evaluate(&net, args.eval_unguided);
            print_eval(&outcomes, args.eval_unguided);
            if let Some(w) = metrics_writer.as_mut() {
                let entry = serde_json::json!({
                    "kind": "eval",
                    "iter": it,
                    "outcomes": outcomes.iter().map(|o| serde_json::json!({
                        "name": o.name,
                        "solved_guided": o.solved_guided,
                        "size_guided": o.size_guided,
                        "elapsed_guided_ms": o.elapsed_guided.as_millis() as u64,
                        "pool_guided": o.pool_guided,
                        "solved_unguided": o.solved_unguided,
                        "size_unguided": o.size_unguided,
                        "elapsed_unguided_ms": o.elapsed_unguided.map(|d| d.as_millis() as u64),
                    })).collect::<Vec<_>>(),
                });
                writeln!(w, "{}", entry).ok();
            }
        }
    }

    // Final eval always runs.
    println!("\n=== final evaluation ===");
    let outcomes = evaluate(&net, args.eval_unguided);
    print_eval(&outcomes, args.eval_unguided);

    if let Some(p) = args.checkpoint_path.as_ref() {
        let f = File::create(p).expect("create checkpoint");
        let mut w = BufWriter::new(f);
        serde_json::to_writer(&mut w, &net).expect("write checkpoint");
        w.flush().ok();
        println!("checkpoint -> {}", p.display());
    }

    let solved: usize = outcomes.iter().filter(|o| o.solved_guided).count();
    println!("\nfinal: {}/{} solved by guided search", solved, outcomes.len());

    if let Some(w) = metrics_writer.as_mut() {
        w.flush().ok();
    }
    if solved < outcomes.len() {
        std::process::exit(1);
    }
}

fn print_eval(outs: &[training::BenchOutcome], unguided: bool) {
    println!(
        "{:<18} {:<8} {:>5} {:>10} {:>9} {:<8} {:>5} {:>10}",
        "task", "G:solved", "size", "ms", "pool",
        if unguided { "U:solved" } else { "" },
        if unguided { "size" } else { "" },
        if unguided { "ms" } else { "" },
    );
    for o in outs {
        let u_solved = o.solved_unguided.map(|b| if b { "yes" } else { "no" }).unwrap_or("-");
        let u_size = o.size_unguided.map(|s| s.to_string()).unwrap_or("-".to_string());
        let u_ms = o.elapsed_unguided.map(|d| format!("{}", d.as_millis())).unwrap_or("-".to_string());
        println!(
            "{:<18} {:<8} {:>5} {:>10} {:>9} {:<8} {:>5} {:>10}",
            o.name,
            if o.solved_guided { "yes" } else { "no" },
            o.size_guided,
            o.elapsed_guided.as_millis(),
            o.pool_guided,
            u_solved, u_size, u_ms,
        );
    }
}
