//! `graph-seek-show-dreams` — sample dreams at a given complexity level
//! and print each program with its (input, output) examples. Uses the
//! same `ComplexityCfg::default()` as the curriculum so what is shown
//! reflects what the trainer actually trains on.

use std::env;

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::eval::Value;
use lang::pretty::pretty;

use neural::Rng;

use training::{sample_complexity_dreams, ComplexityCfg};

fn parse_args() -> (usize, usize, u64) {
    let raw: Vec<String> = env::args().collect();
    let mut level: usize = 2;
    let mut count: usize = 12;
    let mut seed: u64 = 0xC0DE_C0DE;
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--level" => { i += 1; level = raw[i].parse().unwrap(); }
            "--count" => { i += 1; count = raw[i].parse().unwrap(); }
            "--seed"  => { i += 1; seed  = raw[i].parse().unwrap(); }
            "--help" | "-h" => {
                println!("graph-seek-show-dreams [--level N=2] [--count N=12] [--seed N]");
                std::process::exit(0);
            }
            _ => { eprintln!("unknown arg: {}", raw[i]); std::process::exit(2); }
        }
        i += 1;
    }
    (level, count, seed)
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

fn main() {
    let (level, count, seed) = parse_args();
    let lib = seed_builtin_library();
    let mut rng = Rng::new(seed);
    let cfg = ComplexityCfg::default();

    println!(
        "# Dreams at complexity L{} (examples_per_dream={}, seed={:#x})",
        level, cfg.examples_per_dream, seed,
    );
    println!(
        "# (program = ground-truth function; the network sees only the I/O pairs and must rediscover it)\n",
    );

    let dreams = sample_complexity_dreams(&lib, &mut rng, level, count, &cfg);
    if dreams.is_empty() {
        println!("(no dreams sampled — try a different seed or smaller level)");
        return;
    }

    for (idx, (arena, dream)) in dreams.iter().enumerate() {
        // Re-borrow the arena as &Arena for pretty().
        let arena_ref: &Arena = arena;
        println!("─── Task {:>2} ───────────────────────────────────────", idx + 1);
        println!("program: {}", pretty(arena_ref, &lib, dream.program));
        for (i, (inp, out)) in dream.examples.iter().enumerate() {
            println!("  ex{}:  {}  →  {}", i + 1, show_value(inp), show_value(out));
        }
        println!();
    }
}
