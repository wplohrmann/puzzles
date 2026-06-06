//! Measure the per-iter wallclock and memory footprint of one training
//! step at varying `dreams_per_iter` batch sizes.
//!
//! Usage:
//!   graph-seek-bench-batch --dreams-per-iter N --iters I [--level K] [--seed S]
//!
//! Runs `iters` training iterations at level `K` (uniform-draw dream
//! complexity in [1, K]), prints per-iter wall time and the process's
//! peak RSS after the run. Pair this with a shell loop and
//! `/usr/bin/time -l` to compare batch sizes.

use std::env;
use std::time::Instant;

use candle_core::Device;
use candle_nn::AdamW;

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::ir::LitValue;

use neural::{make_optimizer, train_step, Network, NetworkCfg, Rng, TrainSample};

use training::{sample_complexity_dreams, ComplexityCfg};
use training::dream_to_samples;

#[derive(Debug, Clone)]
struct Args {
    seed: u64,
    level: usize,
    iters: usize,
    dreams_per_iter: usize,
    examples_per_dream: usize,
    max_negatives: usize,
    n: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            seed: 0xBEEF_BEEF,
            level: 6,
            iters: 10,
            dreams_per_iter: 32,
            examples_per_dream: 3,
            max_negatives: 16,
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
            "--level" => { i += 1; a.level = raw[i].parse().unwrap(); }
            "--iters" => { i += 1; a.iters = raw[i].parse().unwrap(); }
            "--dreams-per-iter" => { i += 1; a.dreams_per_iter = raw[i].parse().unwrap(); }
            "--examples" => { i += 1; a.examples_per_dream = raw[i].parse().unwrap(); }
            "--max-negatives" => { i += 1; a.max_negatives = raw[i].parse().unwrap(); }
            "--n" => { i += 1; a.n = raw[i].parse().unwrap(); }
            "--help" | "-h" => {
                println!("graph-seek-bench-batch\n\
                    --seed N           (default 0xBEEFBEEF)\n\
                    --level K          (default 6)  uniform draw over [1, K]\n\
                    --iters I          (default 10)\n\
                    --dreams-per-iter N (default 32)\n\
                    --examples N       (default 3)\n\
                    --max-negatives N  (default 16)\n\
                    --n N              (default 32) embedding width\n");
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

/// Peak resident-set size of this process, in bytes. macOS-only path;
/// on other platforms we return 0. (Bench is local-only.)
#[cfg(target_os = "macos")]
fn peak_rss_bytes() -> u64 {
    // Apple's getrusage(RUSAGE_SELF) returns ru_maxrss in BYTES on
    // macOS (unlike Linux, where it's KB).
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru as *mut _) != 0 {
            return 0;
        }
        ru.ru_maxrss as u64
    }
}

#[cfg(not(target_os = "macos"))]
fn peak_rss_bytes() -> u64 { 0 }

fn fmt_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64)
    } else {
        format!("{} B", b)
    }
}

fn main() {
    let args = parse();
    let net_cfg = NetworkCfg { n: args.n, ..NetworkCfg::default() };
    let lib = seed_builtin_library();
    let net = Network::new(&net_cfg, &lib, Device::Cpu).expect("net");
    let mut opt: AdamW = make_optimizer(&net, net_cfg.lr, net_cfg.weight_decay).expect("opt");
    let mut rng = Rng::new(args.seed);

    let cfg = ComplexityCfg {
        examples_per_dream: args.examples_per_dream,
        dreams_per_iter: args.dreams_per_iter,
        max_negatives: args.max_negatives,
        literal_seeds: vec![
            LitValue::Int(-3), LitValue::Int(-2), LitValue::Int(-1),
            LitValue::Int(0), LitValue::Int(1), LitValue::Int(2), LitValue::Int(3),
        ],
        ..ComplexityCfg::default()
    };

    eprintln!(
        "bench-batch: level={} iters={} dreams_per_iter={} examples={} K={} n={}",
        args.level, args.iters, args.dreams_per_iter,
        args.examples_per_dream, args.max_negatives, args.n,
    );

    let rss_before = peak_rss_bytes();
    let mut total_samples = 0usize;
    let started = Instant::now();
    for it in 0..args.iters {
        let mut counts = vec![0usize; args.level];
        for _ in 0..cfg.dreams_per_iter {
            let c = rng.gen_range(args.level);
            counts[c] += 1;
        }
        let mut dreams: Vec<(Arena, _)> = Vec::new();
        for (idx, &count) in counts.iter().enumerate() {
            if count == 0 { continue; }
            dreams.extend(sample_complexity_dreams(&lib, &mut rng, idx + 1, count, &cfg));
        }
        let mut arenas: Vec<Arena> = Vec::new();
        let mut samples: Vec<TrainSample> = Vec::new();
        let mut arena_idx: Vec<usize> = Vec::new();
        for (mut arena, dream) in dreams {
            let s = dream_to_samples(
                &mut arena, &lib, &dream, &cfg.literal_seeds,
                &mut rng, cfg.max_negatives,
            );
            if s.samples.is_empty() { continue; }
            let aidx = arenas.len();
            arenas.push(arena);
            for sample in s.samples {
                arena_idx.push(aidx);
                samples.push(sample);
            }
        }
        if samples.is_empty() { continue; }
        let batch: Vec<(&TrainSample, &Arena, &lang::library::Library)> = samples.iter()
            .enumerate()
            .map(|(i, s)| (s, &arenas[arena_idx[i]], &lib))
            .collect();
        let t0 = Instant::now();
        let stats = train_step(&net, &mut opt, &batch, cfg.fuel).expect("train_step");
        let dt = t0.elapsed();
        total_samples += stats.samples;
        eprintln!(
            "  it {:3}/{}  samples={:4}  loss={:.3}  top1={:.2}  step={:.0}ms  rss={}",
            it + 1, args.iters, stats.samples, stats.loss, stats.positive_top1,
            dt.as_millis(),
            fmt_bytes(peak_rss_bytes()),
        );
    }
    let elapsed = started.elapsed();
    let rss_after = peak_rss_bytes();

    println!(
        "result\tdreams_per_iter={}\titers={}\tsamples_total={}\twall_ms={}\tms_per_iter={:.0}\trss_before={}\trss_peak={}\trss_delta={}",
        args.dreams_per_iter,
        args.iters,
        total_samples,
        elapsed.as_millis(),
        elapsed.as_millis() as f64 / args.iters.max(1) as f64,
        fmt_bytes(rss_before),
        fmt_bytes(rss_after),
        fmt_bytes(rss_after.saturating_sub(rss_before)),
    );
}
