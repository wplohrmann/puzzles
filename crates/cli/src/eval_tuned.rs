//! `graph-seek-eval` — load a trained checkpoint and run the bench with
//! tunable guided-search hyperparams. Useful for testing whether
//! length / add-one-to-each can be solved with a larger pool / longer
//! budget given the same network.

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Duration;

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use neural::Network;
use search::{solve_guided, GuidedConfig, SearchConfig};
use training::bench_tasks;

#[derive(Debug, Clone)]
struct Args {
    checkpoint: String,
    size_penalty: f32,
    pool_cap: usize,
    max_frontier: usize,
    max_program_size: u32,
    budget_secs: u64,
}

fn parse() -> Args {
    let raw: Vec<String> = env::args().collect();
    let mut a = Args {
        checkpoint: String::new(),
        size_penalty: 0.01,
        pool_cap: 30_000,
        max_frontier: 200_000,
        max_program_size: 16,
        budget_secs: 120,
    };
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--checkpoint" => { i += 1; a.checkpoint = raw[i].clone(); }
            "--size-penalty" => { i += 1; a.size_penalty = raw[i].parse().unwrap(); }
            "--pool-cap" => { i += 1; a.pool_cap = raw[i].parse().unwrap(); }
            "--max-frontier" => { i += 1; a.max_frontier = raw[i].parse().unwrap(); }
            "--max-size" => { i += 1; a.max_program_size = raw[i].parse().unwrap(); }
            "--budget" => { i += 1; a.budget_secs = raw[i].parse().unwrap(); }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if a.checkpoint.is_empty() {
        eprintln!("--checkpoint PATH is required");
        std::process::exit(2);
    }
    a
}

fn main() {
    let args = parse();
    let f = File::open(&args.checkpoint).expect("open checkpoint");
    let mut s = String::new();
    BufReader::new(f).read_to_string(&mut s).expect("read");
    let mut net: Network = serde_json::from_str(&s).expect("parse");
    net.rehydrate_scratch();

    let lib = seed_builtin_library();
    let gcfg = GuidedConfig {
        size_penalty: args.size_penalty,
        max_frontier: args.max_frontier,
        guided_pool_cap: args.pool_cap,
        priority_floor: f32::NEG_INFINITY,
        state_refresh_every: 8,
    };

    println!(
        "eval: checkpoint={} size_penalty={} pool_cap={} max_frontier={} budget={}s",
        args.checkpoint, args.size_penalty, args.pool_cap, args.max_frontier, args.budget_secs,
    );
    println!("{:<18} {:<8} {:>5} {:>9} {:>10}", "task", "solved", "size", "ms", "pool");
    let mut solved_count = 0;
    let total;
    {
        let tasks = bench_tasks(&lib);
        total = tasks.len();
        for (name, _, task) in tasks {
            let cfg = SearchConfig {
                time_budget: Duration::from_secs(args.budget_secs),
                max_program_size: args.max_program_size,
                max_pool_size: args.pool_cap * 4, // not binding
                ..SearchConfig::default()
            };
            let mut arena = Arena::new();
            let r = solve_guided(&mut arena, &lib, &task, &cfg, &net, &gcfg);
            println!(
                "{:<18} {:<8} {:>5} {:>9} {:>10}",
                name,
                if r.solved { "yes" } else { "no" },
                r.size,
                r.elapsed.as_millis(),
                r.final_pool_size,
            );
            if r.solved { solved_count += 1; }
        }
    }
    println!("\n{}/{} solved", solved_count, total);
}
