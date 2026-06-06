//! `graph-seek-show-poser` — run the poser-search at a fresh-init
//! network and print each dream's program + I/O so we can actually
//! see what the poser proposes (and why most of it is invalid /
//! trivial).
//!
//! No training, no optimizer — just one network init + N poser
//! searches, each followed by evaluating the produced program and
//! reporting why it's valid or rejected.

use std::env;
use std::time::Duration;

use candle_core::Device;

use lang::arena::{Arena, NodeId};
use lang::builtin::seed_builtin_library;
use lang::eval::{eval, Value};
use lang::ir::NodeKind;
use lang::pretty::pretty;

use neural::{Network, NetworkCfg, Rng};
use search::{
    solve_guided_training, GuidedConfig, ScoringHead, SearchConfig, SearchMode, TrainingCfg,
};

use training::dream::{sample_input_kind, sample_input_of_kind};

fn parse_args() -> (usize, u64, f32, usize, usize, u32) {
    let raw: Vec<String> = env::args().collect();
    let mut count: usize = 20;
    let mut seed: u64 = 0xC0DE_C0DE;
    let mut bias: f32 = 1.0;
    let mut top_k: usize = 16;
    let mut max_poser_nodes: usize = 6;
    let mut max_steps: u32 = 500;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--count" => { i += 1; count = raw[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = raw[i].parse().unwrap(); }
            "--poser-stop-bias" => { i += 1; bias = raw[i].parse().unwrap(); }
            "--top-k" => { i += 1; top_k = raw[i].parse().unwrap(); }
            "--max-poser-nodes" => { i += 1; max_poser_nodes = raw[i].parse().unwrap(); }
            "--max-steps" => { i += 1; max_steps = raw[i].parse().unwrap(); }
            "--help" | "-h" => {
                println!(
                    "graph-seek-show-poser [--count N=20] [--seed N] \
                     [--poser-stop-bias F=1.0] [--top-k N=16] \
                     [--max-poser-nodes N=6] [--max-steps N=500]"
                );
                std::process::exit(0);
            }
            _ => { eprintln!("unknown arg: {}", raw[i]); std::process::exit(2); }
        }
        i += 1;
    }
    (count, seed, bias, top_k, max_poser_nodes, max_steps)
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
            s.push(']');
            s
        }
        Value::Pair(p) => format!("({}, {})", show_value(&p.0), show_value(&p.1)),
        Value::Closure(_) => "<closure>".to_string(),
        Value::Bottom(reason) => format!("⊥({})", reason),
    }
}

fn contains_closure(v: &Value) -> bool {
    match v {
        Value::Closure(_) => true,
        Value::List(xs) => xs.iter().any(contains_closure),
        Value::Pair(p) => contains_closure(&p.0) || contains_closure(&p.1),
        _ => false,
    }
}

fn classify(outputs: &[Value], inputs: &[Value]) -> &'static str {
    if outputs.iter().any(|v| v.is_bottom()) { return "BOTTOM"; }
    if outputs.iter().any(contains_closure) { return "CLOSURE"; }
    if outputs.iter().all(|v| v == &outputs[0]) { return "CONSTANT"; }
    if outputs.iter().zip(inputs.iter()).all(|(o, i)| o == i) { return "IDENTITY"; }
    "VALID"
}

fn main() {
    let (count, seed, bias, top_k, max_poser_nodes, max_steps) = parse_args();
    let lib = seed_builtin_library();
    let net_cfg = NetworkCfg {
        n: 32,
        poser_stop_bias: bias,
        ..NetworkCfg::default()
    };
    let net = Network::new(&net_cfg, &lib, Device::Cpu).expect("net build");

    let mut rng = Rng::new(seed);
    let scfg = SearchConfig {
        time_budget: Duration::from_secs(10),
        max_program_size: 24,
        ..SearchConfig::default()
    };
    let gcfg = GuidedConfig::default();
    let tcfg = TrainingCfg {
        top_k,
        temperature: 1.0,
        max_steps,
    };

    println!(
        "# Poser samples — fresh-init network, N=32, poser_stop_bias={:.2}, \
         top_k={}, max_poser_nodes={}, seed={:#x}\n",
        bias, top_k, max_poser_nodes, seed
    );

    let mut counts = std::collections::HashMap::new();
    let mut arena = Arena::new();
    for idx in 0..count {
        let input_kind = sample_input_kind(&mut rng);
        let inputs: Vec<Value> = (0..3).map(|_| sample_input_of_kind(&mut rng, input_kind)).collect();

        let traj = solve_guided_training(
            &mut arena, &lib, &scfg, &net, &gcfg,
            ScoringHead::Poser, SearchMode::Construct,
            &inputs, None, &tcfg, &mut rng,
        );

        let n_id: Option<NodeId> = traj.solution.as_ref().map(|s| s.root);
        let n_size = traj.solution.as_ref().map(|s| s.size).unwrap_or(0);

        let program_str = n_id.map(|n| pretty(&arena, &lib, n)).unwrap_or_else(|| "(no program)".to_string());

        let (outputs, label) = if let Some(n) = n_id {
            let mut outs = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let env = [input.clone()];
                let mut fuel = 50_000u32;
                let v = match eval(&arena, &lib, n, &env, &mut fuel) {
                    Ok(v) => v,
                    Err(_) => Value::bottom("eval err"),
                };
                outs.push(v);
            }
            let label = classify(&outs, &inputs);
            (outs, label)
        } else {
            (Vec::new(), "NO_PROGRAM")
        };

        *counts.entry(label).or_insert(0usize) += 1;

        // Pull out the head node-kind of `n` for a quick glance at
        // structure.
        let head_kind = n_id.and_then(|n| {
            if let NodeKind::App { func, .. } = arena.kind(n) {
                match arena.kind(*func) {
                    NodeKind::PrimRef(p) => Some(lib.get(*p).name.clone()),
                    NodeKind::App { .. } => Some("<App>".to_string()),
                    _ => Some("<other>".to_string()),
                }
            } else { None }
        }).unwrap_or_else(|| "—".to_string());

        println!(
            "─── #{:>2} [{}]  N_poser={}  steps={}  head={}",
            idx + 1, label, n_size, traj.steps.len(), head_kind,
        );
        println!("    program: {}", program_str);
        for (i, inp) in inputs.iter().enumerate() {
            let out_str = outputs.get(i).map(|v| show_value(v)).unwrap_or_else(|| "—".to_string());
            println!("    ex{}: {}  →  {}", i + 1, show_value(inp), out_str);
        }
        println!();
    }

    println!("\n# Summary");
    let mut pairs: Vec<_> = counts.iter().collect();
    pairs.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    for (label, c) in pairs {
        println!("  {:<10}  {:>3} / {}", label, c, count);
    }
}
