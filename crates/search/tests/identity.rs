//! Smoke test: the search must find `identity` (size 1) at the seeding
//! step for any task whose expected output equals its input. This
//! exercises the full pipeline (Library → Arena → Pool → solve) without
//! depending on enumeration speed at all. Performance characterisation
//! lives in `benches/trivial_list.rs`.

use std::time::Duration;

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::eval::Value;

use search::{solve, SearchConfig};
use tasks::{ListExamplesTask, TaskId};

#[test]
fn identity_is_found_at_seeding() {
    let lib = seed_builtin_library();
    let mut arena = Arena::new();
    let task = ListExamplesTask {
        id: TaskId(1),
        examples: vec![
            (
                Value::list_from(vec![Value::Int(1), Value::Int(2)]),
                Value::list_from(vec![Value::Int(1), Value::Int(2)]),
            ),
            (Value::list_from(vec![]), Value::list_from(vec![])),
            (
                Value::list_from(vec![Value::Int(99)]),
                Value::list_from(vec![Value::Int(99)]),
            ),
        ],
        fuel: 100_000,
    };
    let cfg = SearchConfig {
        time_budget: Duration::from_secs(5),
        ..SearchConfig::default()
    };
    let r = solve(&mut arena, &lib, &task, &cfg);
    assert!(r.solved, "identity not solved: {:?}", r);
    assert_eq!(r.size, 1, "identity should be size 1");
}
