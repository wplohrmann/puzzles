//! `graph-seek-complexity` — train the network on dreams of increasing
//! complexity (number of `App` nodes), then evaluate by running guided
//! search to check whether the network can recover programs whose
//! outputs match the target examples.
//!
//! Writes a markdown table summarising performance per complexity
//! level; useful for verifying the network learns end-to-end.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use candle_core::Device;
use candle_nn::AdamW;

use lang::builtin::seed_builtin_library;
use lang::ir::LitValue;

use neural::{make_optimizer, Network, NetworkCfg, Rng};

use training::{
    train_complexity_curriculum, ComplexityCfg,
};

#[derive(Debug, Clone)]
struct Args {
    seed: u64,
    levels: Vec<usize>,
    iters_per_level: usize,
    dreams_per_iter: usize,
    eval_dreams_per_level: usize,
    eval_budget_secs: u64,
    out: Option<PathBuf>,
    examples_per_dream: usize,
    n: usize,
    cumulative: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            seed: 0xC0DE_C0DE,
            levels: vec![1, 2, 3, 4, 5, 6],
            iters_per_level: 30,
            dreams_per_iter: 6,
            eval_dreams_per_level: 20,
            eval_budget_secs: 6,
            out: None,
            examples_per_dream: 3,
            n: 32,
            cumulative: false,
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
            "--iters-per-level" => { i += 1; a.iters_per_level = raw[i].parse().unwrap(); }
            "--dreams-per-iter" => { i += 1; a.dreams_per_iter = raw[i].parse().unwrap(); }
            "--eval-dreams" => { i += 1; a.eval_dreams_per_level = raw[i].parse().unwrap(); }
            "--eval-budget" => { i += 1; a.eval_budget_secs = raw[i].parse().unwrap(); }
            "--examples" => { i += 1; a.examples_per_dream = raw[i].parse().unwrap(); }
            "--n" => { i += 1; a.n = raw[i].parse().unwrap(); }
            "--out" => { i += 1; a.out = Some(PathBuf::from(&raw[i])); }
            "--cumulative" => { a.cumulative = true; }
            "--help" | "-h" => {
                println!("graph-seek-complexity\n\
                    --seed N\n\
                    --levels CSV  (default 1,2,3,4,5,6)\n\
                    --iters-per-level N (default 30)\n\
                    --dreams-per-iter N (default 6)\n\
                    --eval-dreams N (default 20)\n\
                    --eval-budget SECS (default 6)\n\
                    --examples N (default 3)\n\
                    --n N  embedding width (default 32)\n\
                    --out PATH  markdown output (default stdout)\n");
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
        iters_per_level: args.iters_per_level,
        eval_dreams_per_level: args.eval_dreams_per_level,
        eval_search_budget: Duration::from_secs(args.eval_budget_secs),
        literal_seeds: vec![
            LitValue::Int(-3), LitValue::Int(-2), LitValue::Int(-1),
            LitValue::Int(0), LitValue::Int(1), LitValue::Int(2), LitValue::Int(3),
        ],
        cumulative: args.cumulative,
        ..ComplexityCfg::default()
    };

    println!(
        "graph-seek-complexity: levels={:?} iters_per_level={} dreams_per_iter={} eval_dreams={}",
        cfg.levels, cfg.iters_per_level, cfg.dreams_per_iter, cfg.eval_dreams_per_level,
    );

    let report = train_complexity_curriculum(
        &net, &mut opt, &lib, &mut rng, &cfg,
        |level, iter, loss, top1| {
            if iter == 0 || (iter + 1) % 5 == 0 {
                println!(
                    "  level {} iter {:3}/{}  loss={:.3} top1={:.2}",
                    level, iter + 1, cfg.iters_per_level, loss, top1,
                );
            }
        },
    );

    let md = report.render_markdown();
    println!("\n{}", md);
    if let Some(p) = args.out {
        let mut f = File::create(&p).expect("create out");
        f.write_all(md.as_bytes()).ok();
        println!("wrote table to {}", p.display());
    }
}
