//! Property-based invariants. Use small inputs by default; `cargo test
//! --release` for larger ranges.

use lang::arena::Arena;
use lang::builtin::seed_builtin_library;
use lang::construct::{app, lit, prim_ref};
use lang::eval::{eval_program, Value};
use lang::ir::LitValue;

use proptest::prelude::*;

proptest! {
    /// Inserting a literal with the same value twice yields the same NodeId.
    #[test]
    fn lit_int_intern_idempotent(i in any::<i64>()) {
        let mut a = Arena::new();
        let id1 = lit(&mut a, LitValue::Int(i));
        let id2 = lit(&mut a, LitValue::Int(i));
        prop_assert_eq!(id1, id2);
    }

    /// Two distinct ints produce different nodes.
    #[test]
    fn distinct_ints_yield_distinct_ids(i in any::<i64>(), j in any::<i64>()) {
        prop_assume!(i != j);
        let mut a = Arena::new();
        let id1 = lit(&mut a, LitValue::Int(i));
        let id2 = lit(&mut a, LitValue::Int(j));
        prop_assert_ne!(id1, id2);
    }

    /// Building `add a b` evaluates to the wrapping integer sum.
    #[test]
    fn add_well_typed(a in any::<i64>(), b in any::<i64>()) {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let na = lit(&mut arena, LitValue::Int(a));
        let nb = lit(&mut arena, LitValue::Int(b));
        let add = prim_ref(&mut arena, &lib, lib.lookup("add").unwrap());
        let app1 = app(&mut arena, add, na);
        let prog = app(&mut arena, app1, nb);
        let v = eval_program(&arena, &lib, prog, vec![], 1_000).unwrap();
        prop_assert_eq!(v, Value::Int(a.wrapping_add(b)));
    }

    /// `add` applied to a Bool now constructs successfully (no static
    /// type-check) and surfaces as Bottom at runtime.
    #[test]
    fn add_with_bool_yields_bottom(b in any::<bool>()) {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let nb = lit(&mut arena, LitValue::Bool(b));
        let one = lit(&mut arena, LitValue::Int(1));
        let add = prim_ref(&mut arena, &lib, lib.lookup("add").unwrap());
        let app1 = app(&mut arena, add, nb);
        let prog = app(&mut arena, app1, one);
        let v = eval_program(&arena, &lib, prog, vec![], 1_000).unwrap();
        prop_assert!(v.is_bottom());
    }

    /// Building the same fold-sum-list expression twice yields the same id.
    #[test]
    fn fold_sum_intern_idempotent(xs in proptest::collection::vec(-100i64..=100, 0..=10)) {
        let lib = seed_builtin_library();
        let build = |arena: &mut Arena| {
            let nil = prim_ref(arena, &lib, lib.lookup("nil").unwrap());
            let cons = prim_ref(arena, &lib, lib.lookup("cons").unwrap());
            let add = prim_ref(arena, &lib, lib.lookup("add").unwrap());
            let fold = prim_ref(arena, &lib, lib.lookup("fold").unwrap());
            let mut list = nil;
            for &i in xs.iter().rev() {
                let n = lit(arena, LitValue::Int(i));
                let c = app(arena, cons, n);
                list = app(arena, c, list);
            }
            let zero = lit(arena, LitValue::Int(0));
            let f1 = app(arena, fold, add);
            let f2 = app(arena, f1, zero);
            app(arena, f2, list)
        };
        let mut arena = Arena::new();
        let id1 = build(&mut arena);
        let id2 = build(&mut arena);
        prop_assert_eq!(id1, id2);
    }

    /// Sum of a list via fold equals the integer sum.
    #[test]
    fn fold_sum_is_correct(xs in proptest::collection::vec(-100i64..=100, 0..=20)) {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let nil = prim_ref(&mut arena, &lib, lib.lookup("nil").unwrap());
        let cons = prim_ref(&mut arena, &lib, lib.lookup("cons").unwrap());
        let add = prim_ref(&mut arena, &lib, lib.lookup("add").unwrap());
        let fold = prim_ref(&mut arena, &lib, lib.lookup("fold").unwrap());
        let mut list = nil;
        for &i in xs.iter().rev() {
            let n = lit(&mut arena, LitValue::Int(i));
            let c = app(&mut arena, cons, n);
            list = app(&mut arena, c, list);
        }
        let zero = lit(&mut arena, LitValue::Int(0));
        let f1 = app(&mut arena, fold, add);
        let f2 = app(&mut arena, f1, zero);
        let prog = app(&mut arena, f2, list);
        let v = eval_program(&arena, &lib, prog, vec![], 100_000).unwrap();
        let expected: i64 = xs.iter().fold(0i64, |a, b| a.wrapping_add(*b));
        prop_assert_eq!(v, Value::Int(expected));
    }

    /// Round-trip: serialise then deserialise into a fresh arena yields a
    /// program that evaluates to the same value.
    #[test]
    fn round_trip_eval_equal(a in -1000i64..=1000, b in -1000i64..=1000) {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let na = lit(&mut arena, LitValue::Int(a));
        let nb = lit(&mut arena, LitValue::Int(b));
        let add = prim_ref(&mut arena, &lib, lib.lookup("add").unwrap());
        let app1 = app(&mut arena, add, na);
        let prog = app(&mut arena, app1, nb);
        let v1 = eval_program(&arena, &lib, prog, vec![], 1_000).unwrap();

        let repr = lang::serial::serialize(&arena, prog);
        let mut arena2 = Arena::new();
        let prog2 = lang::serial::deserialize(&repr, &mut arena2);
        let v2 = eval_program(&arena2, &lib, prog2, vec![], 1_000).unwrap();
        prop_assert_eq!(v1, v2);
    }

    /// Reachable-topo ordering: every child appears before its parent.
    #[test]
    fn topo_order_invariant(a in -100i64..=100, b in -100i64..=100, c in -100i64..=100) {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let na = lit(&mut arena, LitValue::Int(a));
        let nb = lit(&mut arena, LitValue::Int(b));
        let nc = lit(&mut arena, LitValue::Int(c));
        let add = prim_ref(&mut arena, &lib, lib.lookup("add").unwrap());
        let mul = prim_ref(&mut arena, &lib, lib.lookup("mul").unwrap());
        let app1 = app(&mut arena, add, na);
        let sum = app(&mut arena, app1, nb);
        let app2 = app(&mut arena, mul, sum);
        let prog = app(&mut arena, app2, nc);

        let topo = arena.reachable_topo(prog);
        let mut seen = std::collections::HashSet::new();
        for id in &topo {
            for child in arena.children(*id) {
                prop_assert!(seen.contains(&child),
                    "child {:?} of {:?} appears before child in topo", child, id);
            }
            seen.insert(*id);
        }
        prop_assert_eq!(*topo.last().unwrap(), prog);
    }
}
