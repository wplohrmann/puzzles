//! Periodic evaluation: run guided search on the canonical
//! search-benchmark tasks and report per-task wall time + success.

use std::time::Duration;

use lang::arena::Arena;
use lang::builtin::{seed_builtin_library, BuiltinId};
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::Value;
use lang::ir::LitValue;
use lang::library::{Library, PrimId};

use neural::Network;

use search::{solve, solve_guided, GuidedConfig, SearchConfig};
use tasks::{programmatic_task, ListExamplesTask, TaskId};

/// One row of the eval table.
#[derive(Clone, Debug)]
pub struct BenchOutcome {
    pub name: String,
    pub solved_guided: bool,
    pub size_guided: u32,
    pub elapsed_guided: Duration,
    pub pool_guided: usize,
    pub solved_unguided: Option<bool>,
    pub size_unguided: Option<u32>,
    pub elapsed_unguided: Option<Duration>,
}

/// Build the canonical bench tasks. Standard input set, the same as the
/// trivial-list bench harness uses.
pub fn bench_tasks(lib: &Library) -> Vec<(String, Duration, ListExamplesTask)> {
    let mut out = Vec::new();
    let mut arena = Arena::new();

    let inputs = standard_inputs();

    let identity = build_identity(&mut arena);
    out.push((
        "identity".into(), Duration::from_secs(2),
        programmatic_task(TaskId(101), &arena, lib, identity, inputs.clone(), 200_000),
    ));
    let sum = build_sum(&mut arena, lib);
    out.push((
        "sum".into(), Duration::from_secs(20),
        programmatic_task(TaskId(102), &arena, lib, sum, inputs.clone(), 200_000),
    ));
    let head = build_head(&mut arena, lib);
    out.push((
        "head".into(), Duration::from_secs(20),
        programmatic_task(TaskId(103), &arena, lib, head, inputs.clone(), 200_000),
    ));
    let length = build_length(&mut arena, lib);
    out.push((
        "length".into(), Duration::from_secs(60),
        programmatic_task(TaskId(104), &arena, lib, length, inputs.clone(), 200_000),
    ));
    let aoe = build_add_one_to_each(&mut arena, lib);
    out.push((
        "add-one-to-each".into(), Duration::from_secs(60),
        programmatic_task(TaskId(105), &arena, lib, aoe, inputs, 200_000),
    ));
    out
}

/// Run the guided search across all bench tasks and (optionally) the
/// unguided baseline at the same time budget.
pub fn evaluate(
    net: &Network,
    run_unguided_baseline: bool,
) -> Vec<BenchOutcome> {
    let lib = seed_builtin_library();
    let mut outcomes = Vec::new();
    for (name, budget, task) in bench_tasks(&lib) {
        let cfg = SearchConfig {
            time_budget: budget,
            max_program_size: 16,
            ..SearchConfig::default()
        };
        // Guided.
        let mut a = Arena::new();
        let r = solve_guided(&mut a, &lib, &task, &cfg, net, &GuidedConfig::default());

        // Unguided.
        let (solved_u, size_u, elapsed_u) = if run_unguided_baseline {
            let mut a = Arena::new();
            let r = solve(&mut a, &lib, &task, &cfg);
            (Some(r.solved), Some(r.size), Some(r.elapsed))
        } else {
            (None, None, None)
        };

        outcomes.push(BenchOutcome {
            name,
            solved_guided: r.solved, size_guided: r.size,
            elapsed_guided: r.elapsed, pool_guided: r.final_pool_size,
            solved_unguided: solved_u, size_unguided: size_u,
            elapsed_unguided: elapsed_u,
        });
    }
    outcomes
}

fn p(lib: &Library, b: BuiltinId) -> PrimId {
    lib.lookup(b.name()).unwrap()
}

fn build_identity(arena: &mut Arena) -> lang::arena::NodeId { param(arena, 0) }

fn build_head(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
    let p_node = param(arena, 0);
    let fold = prim_ref(arena, p(lib, BuiltinId::Fold));
    let k = prim_ref(arena, p(lib, BuiltinId::K));
    let zero = lit(arena, LitValue::Int(0));
    let f1 = app(arena, fold, k);
    let f2 = app(arena, f1, zero);
    app(arena, f2, p_node)
}

fn build_length(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
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

fn build_add_one_to_each(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
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

fn build_sum(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
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
