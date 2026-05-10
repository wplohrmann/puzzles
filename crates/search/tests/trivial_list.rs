//! M2 acceptance: 5 trivial list tasks must each solve in under 10s.
//!
//! The reference programs that match the seed library + (K, B) combinators:
//!   - identity        : λxs:List<Int>. xs                       — size 1
//!   - head            : λxs:List<Int>. fold k 0 xs              — size 6
//!   - last            : λxs:List<Int>. fold (k (k 0)) 0 xs      — only here
//!     (held in reserve; not part of the acceptance set)
//!   - length          : λxs:List<Int>. fold (k (add 1)) 0 xs    — size 11
//!   - add-one-to-each : λxs:List<Int>. fold (b cons (add 1)) nil xs — size 13
//!   - sum             : λxs:List<Int>. fold add 0 xs            — size 7
//!
//! The search is given the same seed library; it has to discover each
//! program by typed bottom-up enumeration.

use std::time::Duration;

use lang::arena::Arena;
use lang::builtin::{seed_builtin_library, BuiltinId};
use lang::construct::{app, lit, param, prim_ref};
use lang::eval::{eval, Value};
use lang::ir::LitValue;
use lang::library::{Library, PrimId};
use lang::ty::{Ty, TyVarGen};

use search::{solve, SearchConfig};
use tasks::{programmatic_task, ListExamplesTask, TaskId};

fn p(lib: &Library, b: BuiltinId) -> PrimId {
    lib.lookup(b.name()).unwrap()
}

/// Build `λxs. xs` (well, the body — search returns expressions in P0).
fn build_identity(arena: &mut Arena) -> lang::arena::NodeId {
    param(arena, 0, Ty::list(Ty::int()))
}

/// `fold k 0 P` — head with default 0.
fn build_head(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
    let mut g = TyVarGen::new();
    let p_node = param(arena, 0, Ty::list(Ty::int()));
    let fold = prim_ref(arena, lib, p(lib, BuiltinId::Fold));
    let k = prim_ref(arena, lib, p(lib, BuiltinId::K));
    let zero = lit(arena, LitValue::Int(0));
    let f1 = app(arena, &mut g, fold, k).unwrap();
    let f2 = app(arena, &mut g, f1, zero).unwrap();
    app(arena, &mut g, f2, p_node).unwrap()
}

/// `fold (k (add 1)) 0 P` — length.
fn build_length(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
    let mut g = TyVarGen::new();
    let p_node = param(arena, 0, Ty::list(Ty::int()));
    let fold = prim_ref(arena, lib, p(lib, BuiltinId::Fold));
    let k = prim_ref(arena, lib, p(lib, BuiltinId::K));
    let add = prim_ref(arena, lib, p(lib, BuiltinId::Add));
    let one = lit(arena, LitValue::Int(1));
    let zero = lit(arena, LitValue::Int(0));
    let inc = app(arena, &mut g, add, one).unwrap();
    let cb = app(arena, &mut g, k, inc).unwrap();
    let f1 = app(arena, &mut g, fold, cb).unwrap();
    let f2 = app(arena, &mut g, f1, zero).unwrap();
    app(arena, &mut g, f2, p_node).unwrap()
}

/// `fold (b cons (add 1)) nil P` — add-one-to-each.
fn build_add_one_to_each(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
    let mut g = TyVarGen::new();
    let p_node = param(arena, 0, Ty::list(Ty::int()));
    let fold = prim_ref(arena, lib, p(lib, BuiltinId::Fold));
    let b = prim_ref(arena, lib, p(lib, BuiltinId::B));
    let cons = prim_ref(arena, lib, p(lib, BuiltinId::Cons));
    let add = prim_ref(arena, lib, p(lib, BuiltinId::Add));
    let one = lit(arena, LitValue::Int(1));
    let nil = prim_ref(arena, lib, p(lib, BuiltinId::Nil));
    let inc = app(arena, &mut g, add, one).unwrap();
    let bc = app(arena, &mut g, b, cons).unwrap();
    let cb = app(arena, &mut g, bc, inc).unwrap();
    let f1 = app(arena, &mut g, fold, cb).unwrap();
    let f2 = app(arena, &mut g, f1, nil).unwrap();
    app(arena, &mut g, f2, p_node).unwrap()
}

/// `fold add 0 P` — sum.
fn build_sum(arena: &mut Arena, lib: &Library) -> lang::arena::NodeId {
    let mut g = TyVarGen::new();
    let p_node = param(arena, 0, Ty::list(Ty::int()));
    let fold = prim_ref(arena, lib, p(lib, BuiltinId::Fold));
    let add = prim_ref(arena, lib, p(lib, BuiltinId::Add));
    let zero = lit(arena, LitValue::Int(0));
    let f1 = app(arena, &mut g, fold, add).unwrap();
    let f2 = app(arena, &mut g, f1, zero).unwrap();
    app(arena, &mut g, f2, p_node).unwrap()
}

/// Standard input set: 4 lists of varying length (incl. empty + singleton).
fn standard_inputs() -> Vec<Value> {
    vec![
        Value::list_from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        Value::list_from(vec![]),
        Value::list_from(vec![Value::Int(7)]),
        Value::list_from(vec![Value::Int(-2), Value::Int(5), Value::Int(0), Value::Int(11)]),
    ]
}

const TIME_BUDGET_SECS: u64 = 10;
const FUEL: u32 = 200_000;

fn config() -> SearchConfig {
    SearchConfig {
        time_budget: Duration::from_secs(TIME_BUDGET_SECS),
        ..SearchConfig::default()
    }
}

/// Verify a returned program actually matches the task on every example.
fn verify_solution(arena: &Arena, lib: &Library, task: &ListExamplesTask, root: lang::arena::NodeId) {
    for (input, expected) in &task.examples {
        let env = [input.clone()];
        let mut fuel = FUEL;
        let actual = eval(arena, lib, root, &env, &mut fuel)
            .expect("solution evaluates without error");
        assert_eq!(
            &actual, expected,
            "solution disagrees on input {:?}: got {:?}, expected {:?}",
            input, actual, expected,
        );
    }
}

fn run_task(name: &str, ground_truth_builder: impl Fn(&mut Arena, &Library) -> lang::arena::NodeId,
            id: u64, arg_ty: Ty, ret_ty: Ty)
{
    let lib = seed_builtin_library();
    let mut arena = Arena::new();
    let gt = ground_truth_builder(&mut arena, &lib);
    let task = programmatic_task(
        TaskId(id), &arena, &lib, gt, standard_inputs(),
        arg_ty, ret_ty, FUEL,
    );

    let mut search_arena = Arena::new();
    let cfg = config();
    let r = solve(&mut search_arena, &lib, &task, &cfg);
    eprintln!(
        "task={:<18} solved={} size={} elapsed={:.3?} pool={} stats={:?}",
        name, r.solved, r.size, r.elapsed, r.final_pool_size, r.stats,
    );
    assert!(
        r.solved,
        "{} not solved within {}s. stats={:?}", name, TIME_BUDGET_SECS, r.stats,
    );
    verify_solution(&search_arena, &lib, &task, r.program.unwrap());
    assert!(r.elapsed.as_secs() < TIME_BUDGET_SECS,
            "{} took {:?}, exceeds budget", name, r.elapsed);
}

#[test]
fn task_identity() {
    run_task(
        "identity",
        |a, _| build_identity(a),
        1,
        Ty::list(Ty::int()),
        Ty::list(Ty::int()),
    );
}

#[test]
fn task_head() {
    run_task(
        "head",
        |a, l| build_head(a, l),
        2,
        Ty::list(Ty::int()),
        Ty::int(),
    );
}

#[test]
fn task_length() {
    run_task(
        "length",
        |a, l| build_length(a, l),
        3,
        Ty::list(Ty::int()),
        Ty::int(),
    );
}

#[test]
fn task_add_one_to_each() {
    run_task(
        "add-one-to-each",
        |a, l| build_add_one_to_each(a, l),
        4,
        Ty::list(Ty::int()),
        Ty::list(Ty::int()),
    );
}

#[test]
fn task_sum() {
    run_task(
        "sum",
        |a, l| build_sum(a, l),
        5,
        Ty::list(Ty::int()),
        Ty::int(),
    );
}
