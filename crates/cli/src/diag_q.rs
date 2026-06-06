//! `graph-seek-diag-q` — diagnostic runner for the q-head on gold
//! tasks.
//!
//! Samples a balanced batch (N per category) of gold tasks, runs the
//! training-mode q-search on each, and reports:
//!   - per-category solve rate, mean S_pool, mean S_sol;
//!   - a few sample programs the q-search found (when it solved);
//!   - a few sample I/O for tasks the q-search FAILED on;
//!   - h_value embedding statistics (per-dim variance) on a sample of
//!     trunk inputs — to detect collapse.
//!
//! Loads weights via `--load PATH` if you have a saved checkpoint;
//! otherwise uses a fresh-init network.

use std::env;
use std::path::PathBuf;
use std::time::Duration;

use candle_core::{Device, Tensor};

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::eval::Value;
use lang::ir::LitValue;
use lang::pretty::pretty;

use neural::{EmbedCache, Network, NetworkCfg, Rng};

use search::{
    solve_guided_training, GuidedConfig, ScoringHead, SearchConfig, SearchMode, TrainingCfg,
};

use training::{sample_gold_in_category, GoldCategory, GoldTask};

#[derive(Debug, Clone)]
struct Args {
    seed: u64,
    per_category: usize,
    max_budget: u32,
    examples_per_dream: usize,
    n: usize,
    load: Option<PathBuf>,
    show_examples: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            seed: 0xC0DE_C0DE,
            per_category: 20,
            max_budget: 200,
            examples_per_dream: 3,
            n: 32,
            load: None,
            show_examples: 3,
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
            "--per-category" => { i += 1; a.per_category = raw[i].parse().unwrap(); }
            "--max-budget" => { i += 1; a.max_budget = raw[i].parse().unwrap(); }
            "--examples" => { i += 1; a.examples_per_dream = raw[i].parse().unwrap(); }
            "--n" => { i += 1; a.n = raw[i].parse().unwrap(); }
            "--load" => { i += 1; a.load = Some(PathBuf::from(&raw[i])); }
            "--show-examples" => { i += 1; a.show_examples = raw[i].parse().unwrap(); }
            "--help" | "-h" => {
                println!(
                    "graph-seek-diag-q\n\
                    --seed N                RNG seed (default 0xC0DE_C0DE)\n\
                    --per-category N        tasks per category (default 20)\n\
                    --max-budget N          q-search budget (default 200)\n\
                    --examples N            I/O per arith task (default 3)\n\
                    --n N                   embedding width (default 32)\n\
                    --load PATH             load network weights\n\
                    --show-examples N       sample programs to dump per category (default 3)\n"
                );
                std::process::exit(0);
            }
            _ => { eprintln!("unknown arg: {}", raw[i]); std::process::exit(2); }
        }
        i += 1;
    }
    a
}

fn show_value(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Float(f) => format!("{}", f),
        Value::Char(c) => format!("{:?}", c),
        Value::List(xs) => {
            let mut s = String::from("[");
            for (i, x) in xs.iter().enumerate() {
                if i > 0 { s.push_str(", "); }
                s.push_str(&show_value(x));
            }
            s.push(']'); s
        }
        Value::Pair(p) => format!("({}, {})", show_value(&p.0), show_value(&p.1)),
        Value::Closure(_) => "<closure>".to_string(),
        Value::Bottom(reason) => format!("⊥({})", reason),
    }
}

struct CategoryStats {
    total: usize,
    solved: usize,
    sum_s_pool: u64,
    sum_s_sol: u64,
    solved_samples: Vec<(GoldTask, String, u32)>,    // (task, prog_str, S_pool) — first few
    failed_samples: Vec<(GoldTask, u32)>,            // (task, S_pool)
}

impl CategoryStats {
    fn new() -> Self {
        Self {
            total: 0, solved: 0,
            sum_s_pool: 0, sum_s_sol: 0,
            solved_samples: Vec::new(), failed_samples: Vec::new(),
        }
    }
    fn solve_rate(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.solved as f32 / self.total as f32 }
    }
    fn mean_s_pool(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.sum_s_pool as f32 / self.total as f32 }
    }
    fn mean_s_sol(&self) -> f32 {
        if self.solved == 0 { 0.0 } else { self.sum_s_sol as f32 / self.solved as f32 }
    }
}

fn main() {
    let args = parse();
    let lib = seed_builtin_library();
    let net_cfg = NetworkCfg { n: args.n, ..NetworkCfg::default() };
    let mut net = Network::new(&net_cfg, &lib, Device::Cpu).expect("net build");
    if let Some(path) = &args.load {
        net.load(path).expect("load model");
        println!("# Loaded model from {}", path.display());
    } else {
        println!("# Fresh-init network (no --load)");
    }

    let mut rng = Rng::new(args.seed);

    let scfg = SearchConfig {
        time_budget: Duration::from_secs(10),
        max_program_size: args.max_budget,
        eval_fuel: 50_000,
        literal_seeds: vec![
            LitValue::Int(-3), LitValue::Int(-2), LitValue::Int(-1),
            LitValue::Int(0), LitValue::Int(1), LitValue::Int(2), LitValue::Int(3),
            LitValue::Bool(true), LitValue::Bool(false),
        ],
        ..SearchConfig::default()
    };
    let gcfg = GuidedConfig::default();
    let tcfg = TrainingCfg {
        top_k: 16, temperature: 1.0, max_steps: args.max_budget,
    };

    let categories = [
        GoldCategory::ArithUnary,
        GoldCategory::ArithCompose,
        GoldCategory::BoolTruthTable,
    ];

    let mut stats: std::collections::HashMap<GoldCategory, CategoryStats> = categories
        .iter().map(|c| (*c, CategoryStats::new())).collect();

    let mut arena = Arena::new();
    for cat in &categories {
        for _ in 0..args.per_category {
            let task = sample_gold_in_category(*cat, &mut rng, args.examples_per_dream);
            let traj = solve_guided_training(
                &mut arena, &lib, &scfg, &net, &gcfg,
                ScoringHead::Q, SearchMode::Solve,
                &task.inputs, Some(&task.outputs), &tcfg, &mut rng,
            );
            let s = stats.get_mut(cat).unwrap();
            s.total += 1;
            s.sum_s_pool += traj.s_pool as u64;
            if let Some(sol) = &traj.solution {
                s.solved += 1;
                s.sum_s_sol += sol.size as u64;
                if s.solved_samples.len() < args.show_examples {
                    let prog = pretty(&arena, &lib, sol.root);
                    s.solved_samples.push((task, prog, traj.s_pool));
                }
            } else if s.failed_samples.len() < args.show_examples {
                s.failed_samples.push((task, traj.s_pool));
            }
        }
    }

    // ------ Per-category summary table ------
    println!("\n## Per-category solve rates\n");
    println!("| Category | Solve rate | Mean S_pool | Mean S_sol |");
    println!("|---|---:|---:|---:|");
    for cat in &categories {
        let s = &stats[cat];
        println!(
            "| {} | {:.0}% ({}/{}) | {:.0} | {:.1} |",
            cat.name(),
            100.0 * s.solve_rate(),
            s.solved, s.total,
            s.mean_s_pool(),
            s.mean_s_sol(),
        );
    }

    // ------ Sample solved programs ------
    println!("\n## Sample SOLVED programs (q's discovered solutions)\n");
    for cat in &categories {
        let s = &stats[cat];
        if s.solved_samples.is_empty() {
            println!("### {} — (no successes)\n", cat.name());
            continue;
        }
        println!("### {}\n", cat.name());
        for (i, (task, prog, s_pool)) in s.solved_samples.iter().enumerate() {
            println!("- #{}  S_pool={}  program: `{}`", i + 1, s_pool, prog);
            for (j, (inp, out)) in task.inputs.iter().zip(task.outputs.iter()).enumerate() {
                if j < 2 {
                    println!("  - ex: {} → {}", show_value(inp), show_value(out));
                }
            }
        }
        println!();
    }

    // ------ Sample failed tasks ------
    println!("\n## Sample FAILED tasks (q didn't solve in budget)\n");
    for cat in &categories {
        let s = &stats[cat];
        if s.failed_samples.is_empty() {
            println!("### {} — (no failures)\n", cat.name());
            continue;
        }
        println!("### {}\n", cat.name());
        for (i, (task, s_pool)) in s.failed_samples.iter().enumerate() {
            println!("- #{}  S_pool={}", i + 1, s_pool);
            for (j, (inp, out)) in task.inputs.iter().zip(task.outputs.iter()).enumerate() {
                if j < 2 {
                    println!("  - ex: {} → {}", show_value(inp), show_value(out));
                }
            }
        }
        println!();
    }

    // ------ Trunk embedding diagnostics ------
    //
    // Measure variance of h_value across a sample of (node, input)
    // pairs. If the trunk has collapsed, per-dim variance will be
    // near zero — diagnostic for "embedding collapse".
    println!("\n## Trunk embedding statistics (h_value)\n");
    embedding_stats(&net, &lib, &mut rng, args.n);
}

fn embedding_stats(net: &Network, lib: &lang::library::Library, rng: &mut Rng, n: usize) {
    use lang::construct::{lit, param, prim_ref};
    let mut arena = Arena::new();
    let mut nodes = Vec::new();
    // Param + literal seeds + primrefs — same nodes the search uses.
    nodes.push(param(&mut arena, 0));
    for v in [-3i64, -1, 0, 1, 3] {
        nodes.push(lit(&mut arena, LitValue::Int(v)));
    }
    for b in [true, false] {
        nodes.push(lit(&mut arena, LitValue::Bool(b)));
    }
    for p in 0..(lib.len() as u32) {
        nodes.push(prim_ref(&mut arena, p));
    }

    // Some representative inputs across kinds.
    let inputs: Vec<Value> = vec![
        Value::Int(0), Value::Int(7), Value::Int(-5),
        Value::Bool(true), Value::Bool(false),
        Value::pair(Value::Bool(true), Value::Bool(false)),
        Value::pair(Value::Int(3), Value::Int(-2)),
    ];
    let _ = rng; // not used

    let mut cache = EmbedCache::default();
    let mut rows: Vec<Tensor> = Vec::new();
    for node in &nodes {
        for (i, input) in inputs.iter().enumerate() {
            match neural::h_value(
                *node, i, input, &arena, lib,
                &net.leaves, &net.app_net, net.lp, 50_000, &mut cache,
            ) {
                Ok(t) => rows.push(t),
                Err(_) => continue,
            }
        }
    }
    if rows.is_empty() {
        println!("(no h_value tensors collected)");
        return;
    }
    let stack = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).expect("stack");
    let (b, _) = stack.dims2().expect("(B, N)");
    let mean = stack.mean(0).expect("mean");                    // (N,)
    let mean_b = mean.broadcast_as((b, n)).expect("bcast");
    let dev = stack.sub(&mean_b).expect("dev");
    let var = dev.mul(&dev).expect("sq").mean(0).expect("var"); // (N,)
    let var_vec: Vec<f32> = var.to_vec1().expect("vec");

    let mean_var: f32 = var_vec.iter().sum::<f32>() / var_vec.len() as f32;
    let min_var = var_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_var = var_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("h_value batch shape: ({}, {})", b, n);
    println!("per-dim variance: mean={:.4}  min={:.4}  max={:.4}", mean_var, min_var, max_var);
    println!("⚠ if mean variance is < 0.01 the trunk has likely collapsed");
}
