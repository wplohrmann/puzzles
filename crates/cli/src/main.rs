//! `graph-seek` — quick eval runner.
//!
//! With `eval`, runs the bench tasks under both the unguided baseline and
//! a guided search using random network weights (or a checkpoint via
//! `--checkpoint`). Useful for sanity-checking the search pipeline
//! without launching a full training run.

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

use neural::{Network, NetworkCfg, Rng};
use training::evaluate;

fn print_eval(outs: &[training::BenchOutcome]) {
    println!(
        "{:<18} {:<8} {:>5} {:>10} {:>9}",
        "task", "G:solved", "size", "ms", "pool",
    );
    for o in outs {
        println!(
            "{:<18} {:<8} {:>5} {:>10} {:>9}",
            o.name,
            if o.solved_guided { "yes" } else { "no" },
            o.size_guided,
            o.elapsed_guided.as_millis(),
            o.pool_guided,
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut checkpoint: Option<String> = None;
    let mut seed: u64 = 0xc0ffee_face;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => { i += 1; checkpoint = Some(args[i].clone()); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }

    let net = if let Some(p) = checkpoint {
        let f = File::open(&p).expect("open checkpoint");
        let mut s = String::new();
        BufReader::new(f).read_to_string(&mut s).expect("read checkpoint");
        let mut n: Network = serde_json::from_str(&s).expect("parse checkpoint");
        n.rehydrate_scratch();
        n
    } else {
        let mut rng = Rng::new(seed);
        Network::new(&NetworkCfg::default(), &mut rng)
    };

    let outcomes = evaluate(&net, false);
    print_eval(&outcomes);
    let solved = outcomes.iter().filter(|o| o.solved_guided).count();
    println!("\n{}/{} solved", solved, outcomes.len());
}
