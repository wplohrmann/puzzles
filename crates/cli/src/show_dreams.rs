//! `graph-seek-show-dreams` — sample dreams from the PCFG dream sampler
//! and print each program with its (input, output) examples.
//!
//! Used to inspect what `training::dream::sample_dream` produces (the
//! same PCFG that seeded the now-removed complexity curriculum). The
//! self-play loop generates its own dreams via the poser-search, so
//! this tool is purely a debugging aid.

use std::env;

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::eval::Value;
use lang::ir::NodeKind;
use lang::pretty::pretty;

use neural::Rng;

use training::{sample_dream, DreamCfg};

fn parse_args() -> (u32, usize, u64) {
    let raw: Vec<String> = env::args().collect();
    let mut max_size: u32 = 7;
    let mut count: usize = 12;
    let mut seed: u64 = 0xC0DE_C0DE;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--max-size" => { i += 1; max_size = raw[i].parse().unwrap(); }
            "--count" => { i += 1; count = raw[i].parse().unwrap(); }
            "--seed"  => { i += 1; seed  = raw[i].parse().unwrap(); }
            "--help" | "-h" => {
                println!("graph-seek-show-dreams [--max-size N=7] [--count N=12] [--seed N]");
                std::process::exit(0);
            }
            _ => { eprintln!("unknown arg: {}", raw[i]); std::process::exit(2); }
        }
        i += 1;
    }
    (max_size, count, seed)
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

fn app_count(arena: &Arena, root: lang::arena::NodeId) -> usize {
    arena
        .reachable_topo(root)
        .iter()
        .filter(|&&id| matches!(arena.kind(id), NodeKind::App { .. }))
        .count()
}

fn main() {
    let (max_size, count, seed) = parse_args();
    let lib = seed_builtin_library();
    let mut rng = Rng::new(seed);
    let cfg = DreamCfg { max_size, ..DreamCfg::default() };

    println!(
        "# PCFG dreams: max_size={}, examples_per_dream={}, seed={:#x}",
        max_size, cfg.examples_per_dream, seed,
    );
    println!(
        "# (program = sampled function; output of program(input) is shown)\n",
    );

    let mut shown = 0usize;
    let mut attempts = 0usize;
    while shown < count && attempts < count * 100 {
        attempts += 1;
        let mut arena = Arena::new();
        let dream = match sample_dream(&mut arena, &lib, &mut rng, &cfg) {
            Some(d) => d,
            None => continue,
        };
        shown += 1;
        let apps = app_count(&arena, dream.program);
        println!(
            "─── Task {:>2}  ({} App nodes) ─────────────────────",
            shown, apps,
        );
        println!("program: {}", pretty(&arena, &lib, dream.program));
        for (i, (inp, out)) in dream.examples.iter().enumerate() {
            println!("  ex{}:  {}  →  {}", i + 1, show_value(inp), show_value(out));
        }
        println!();
    }
    if shown == 0 {
        println!("(no dreams sampled — try a different seed or larger max_size)");
    }
}
