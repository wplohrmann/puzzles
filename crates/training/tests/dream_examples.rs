//! Diagnostic: print a few dreams' canonical pretty-form to sanity-check
//! that the sampler is producing programs that look like the bench
//! targets (fold-shaped programs over lists of ints).

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::pretty;

use neural::Rng;
use training::dream::{sample_dream, DreamCfg};

#[test]
#[ignore] // run on demand: `cargo test -p training -- --ignored dump_dreams`
fn dump_dreams() {
    let lib = seed_builtin_library();
    let mut rng = Rng::new(0xc0ffee);
    let cfg = DreamCfg {
        max_size: 13,
        examples_per_dream: 3,
        ..DreamCfg::default()
    };
    let mut samples = 0;
    let mut size_hist = std::collections::HashMap::<u32, usize>::new();
    let mut prim_first_seen = std::collections::HashSet::<String>::new();
    for _ in 0..200 {
        let mut arena = Arena::new();
        if let Some(d) = sample_dream(&mut arena, &lib, &mut rng, &cfg) {
            samples += 1;
            let s = lang::arena::Arena::reachable_topo(&arena, d.program).len() as u32;
            *size_hist.entry(s).or_default() += 1;
            let pretty = pretty::pretty(&arena, &lib, d.program);
            if samples <= 15 {
                println!("[{}] size={} pretty={}", samples, s, pretty);
                println!("    example 0 out: {:?}", d.examples[0].1);
            }
            // Track interesting primitive contents.
            if pretty.contains("fold") && !prim_first_seen.contains("fold") {
                prim_first_seen.insert("fold".into());
                println!("FIRST FOLD: {}", pretty);
            }
        }
    }
    println!("\ntotal dreams: {}", samples);
    let mut sizes: Vec<u32> = size_hist.keys().copied().collect();
    sizes.sort();
    for s in sizes {
        println!("size {}: {} dreams", s, size_hist[&s]);
    }
}
