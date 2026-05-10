//! Serialisation round-trip tests.

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::construct::{app, lit, prim_ref};
use lang::eval::{eval_program, Value};
use lang::ir::LitValue;
use lang::serial::{deserialize, serialize, ProgramSerial};

const FUEL: u32 = 100_000;

fn build_sum_1_2_3() -> (Arena, lang::arena::NodeId) {
    let lib = seed_builtin_library();
    let mut a = Arena::new();
    let nil = prim_ref(&mut a, &lib, lib.lookup("nil").unwrap());
    let cons = prim_ref(&mut a, &lib, lib.lookup("cons").unwrap());
    let add = prim_ref(&mut a, &lib, lib.lookup("add").unwrap());
    let fold = prim_ref(&mut a, &lib, lib.lookup("fold").unwrap());

    let mut list = nil;
    for i in (1..=3).rev() {
        let n = lit(&mut a, LitValue::Int(i));
        let cons1 = app(&mut a, cons, n);
        list = app(&mut a, cons1, list);
    }
    let zero = lit(&mut a, LitValue::Int(0));
    let f1 = app(&mut a, fold, add);
    let f2 = app(&mut a, f1, zero);
    let prog = app(&mut a, f2, list);
    (a, prog)
}

#[test]
fn serialise_then_deserialise_yields_same_root_in_same_arena() {
    let (a, root) = build_sum_1_2_3();
    let repr = serialize(&a, root);
    let mut a2 = a;
    let root2 = deserialize(&repr, &mut a2);
    assert_eq!(root, root2);
}

#[test]
fn serialise_then_deserialise_into_fresh_arena_evaluates_equally() {
    let (a, root) = build_sum_1_2_3();
    let repr = serialize(&a, root);

    let lib = seed_builtin_library();
    let v1 = eval_program(&a, &lib, root, vec![], FUEL).unwrap();

    let mut a2 = Arena::new();
    let root2 = deserialize(&repr, &mut a2);
    let v2 = eval_program(&a2, &lib, root2, vec![], FUEL).unwrap();

    assert_eq!(v1, v2);
    assert_eq!(v1, Value::Int(6));
}

#[test]
fn json_round_trip() {
    let (a, root) = build_sum_1_2_3();
    let repr = serialize(&a, root);
    let s = serde_json::to_string(&repr).unwrap();
    let repr2: ProgramSerial = serde_json::from_str(&s).unwrap();
    assert_eq!(repr, repr2);

    let mut a2 = Arena::new();
    let root2 = deserialize(&repr2, &mut a2);
    let lib = seed_builtin_library();
    let v = eval_program(&a2, &lib, root2, vec![], FUEL).unwrap();
    assert_eq!(v, Value::Int(6));
}

#[test]
fn topo_order_is_dependencies_first() {
    let (a, root) = build_sum_1_2_3();
    let repr = serialize(&a, root);
    use lang::serial::KindSerial;
    for (i, n) in repr.nodes.iter().enumerate() {
        let i = i as u32;
        match &n.kind {
            KindSerial::Lambda { body } => assert!(*body < i),
            KindSerial::App { func, arg } => {
                assert!(*func < i);
                assert!(*arg < i);
            }
            _ => {}
        }
    }
}

#[test]
fn dropping_unused_nodes_round_trips_only_reachable() {
    let lib = seed_builtin_library();
    let mut a = Arena::new();
    let _unused1 = lit(&mut a, LitValue::Int(999));
    let _unused2 = lit(&mut a, LitValue::Bool(true));
    let _unused3 = prim_ref(&mut a, &lib, lib.lookup("snd").unwrap());

    let one = lit(&mut a, LitValue::Int(1));
    let two = lit(&mut a, LitValue::Int(2));
    let add = prim_ref(&mut a, &lib, lib.lookup("add").unwrap());
    let app1 = app(&mut a, add, one);
    let prog = app(&mut a, app1, two);

    let repr = serialize(&a, prog);
    assert_eq!(repr.nodes.len(), 5);
}
