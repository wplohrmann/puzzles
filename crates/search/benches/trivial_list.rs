//! Performance benchmark: 5 trivial list tasks under un-guided
//! enumeration. This is *not* a correctness test — it characterises
//! how slow the no-NN search is on size-7 to size-13 programs, which
//! the M4 neural prior is meant to pay back.
//!
//! Run with `cargo bench --bench trivial_list` (release profile).
//!
//! Reference programs (right-fold semantics):
//!   - identity        : λxs. xs                                — size 1
//!   - sum             : λxs. fold add 0 xs                     — size 7
//!   - head            : λxs. fold k 0 xs                       — size 7
//!   - length          : λxs. fold (k (add 1)) 0 xs             — size 11
//!   - add-one-to-each : λxs. fold (b cons (add 1)) nil xs      — size 13
//!
//! Custom harness: each task prints a one-line timing summary; we
//! avoid pulling Criterion in just for this.

use std::time::Duration;

use lang::arena::{Arena, NodeId};
use lang::builtin::{seed_builtin_library, BuiltinId};
use lang::construct::{app, lit, param, prim_ref};
use lang::ir::LitValue;
use lang::library::{Library, PrimId};

use search::{solve, SearchConfig};
use tasks::{programmatic_task, TaskId};
use lang::eval::Value;

fn p(lib: &Library, b: BuiltinId) -> PrimId {
    lib.lookup(b.name()).unwrap()
}

fn build_identity(arena: &mut Arena) -> NodeId {
    param(arena, 0)
}

fn build_head(arena: &mut Arena, lib: &Library) -> NodeId {
    let p_node = param(arena, 0);
    let fold = prim_ref(arena, p(lib, BuiltinId::Fold));
    let k = prim_ref(arena, p(lib, BuiltinId::K));
    let zero = lit(arena, LitValue::Int(0));
    let f1 = app(arena, fold, k);
    let f2 = app(arena, f1, zero);
    app(arena, f2, p_node)
}

fn build_length(arena: &mut Arena, lib: &Library) -> NodeId {
    let p_node = param(arena, 0);
    let fold = prim_ref(arena, p(lib, BuiltinId::Fold));
    let k = prim_ref(arena, p(lib, BuiltinId::K));
    let add = prim_ref(arena, p(lib, BuiltinId::Add));
    let one = lit(arena, LitValue::Int(1));
    let zero = lit(arena, LitValue::Int(0));
    let inc = app(arena, add, one);
    let cb = app(arena, k, inc);
    let f1 = app(arena, fold, cb);
    let f2 = app(arena, f1, zero);
    app(arena, f2, p_node)
}

fn build_add_one_to_each(arena: &mut Arena, lib: &Library) -> NodeId {
    let p_node = param(arena, 0);
    let fold = prim_ref(arena, p(lib, BuiltinId::Fold));
    let b = prim_ref(arena, p(lib, BuiltinId::B));
    let cons = prim_ref(arena, p(lib, BuiltinId::Cons));
    let add = prim_ref(arena, p(lib, BuiltinId::Add));
    let one = lit(arena, LitValue::Int(1));
    let nil = prim_ref(arena, p(lib, BuiltinId::Nil));
    let inc = app(arena, add, one);
    let bc = app(arena, b, cons);
    let cb = app(arena, bc, inc);
    let f1 = app(arena, fold, cb);
    let f2 = app(arena, f1, nil);
    app(arena, f2, p_node)
}

fn build_sum(arena: &mut Arena, lib: &Library) -> NodeId {
    let p_node = param(arena, 0);
    let fold = prim_ref(arena, p(lib, BuiltinId::Fold));
    let add = prim_ref(arena, p(lib, BuiltinId::Add));
    let zero = lit(arena, LitValue::Int(0));
    let f1 = app(arena, fold, add);
    let f2 = app(arena, f1, zero);
    app(arena, f2, p_node)
}

fn standard_inputs() -> Vec<Value> {
    vec![
        Value::list_from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        Value::list_from(vec![]),
        Value::list_from(vec![Value::Int(7)]),
        Value::list_from(vec![Value::Int(-2), Value::Int(5), Value::Int(0), Value::Int(11)]),
    ]
}

const FUEL: u32 = 200_000;

fn run(name: &str, build: impl Fn(&mut Arena, &Library) -> NodeId, time_budget_secs: u64) {
    let lib = seed_builtin_library();
    let mut arena = Arena::new();
    let gt = build(&mut arena, &lib);
    let task = programmatic_task(TaskId(0), &arena, &lib, gt, standard_inputs(), FUEL);

    let mut search_arena = Arena::new();
    let cfg = SearchConfig {
        time_budget: Duration::from_secs(time_budget_secs),
        ..SearchConfig::default()
    };
    let r = solve(&mut search_arena, &lib, &task, &cfg);
    println!(
        "{:<18} solved={} size={} elapsed={:.3?} pool={} max_size={}",
        name, r.solved, r.size, r.elapsed, r.final_pool_size, r.stats.max_size_explored,
    );
}

fn main() {
    println!("# trivial list bench (no neural prior)");
    run("identity", |a, _| build_identity(a), 10);
    run("sum", build_sum, 60);
    run("head", build_head, 60);
    run("length", build_length, 120);
    run("add-one-to-each", build_add_one_to_each, 120);
}
