//! End-to-end test programs.
//!
//! Each test constructs a small program by hand using the constructors,
//! evaluates it, and checks the result. Together these exercise:
//! arithmetic, conditionals, pairs, lists, fold, unfold, lambdas,
//! currying, runtime polymorphism, lazy `if`, `Bottom` propagation, and
//! the K/B combinators.

use lang::arena::Arena;
use lang::builtin::{seed_builtin_library, BuiltinId};
use lang::construct::{app, lambda, lit, param, prim_ref};
use lang::eval::{eval_program, Value};
use lang::ir::LitValue;
use lang::library::{Library, PrimId};

const FUEL: u32 = 1_000_000;

struct Builder {
    arena: Arena,
    lib: Library,
}

impl Builder {
    fn new() -> Self {
        Builder {
            arena: Arena::new(),
            lib: seed_builtin_library(),
        }
    }

    fn p(&self, b: BuiltinId) -> PrimId {
        self.lib.lookup(b.name()).unwrap()
    }

    fn pref(&mut self, b: BuiltinId) -> lang::arena::NodeId {
        let p = self.p(b);
        prim_ref(&mut self.arena, p)
    }

    fn int(&mut self, n: i64) -> lang::arena::NodeId {
        lit(&mut self.arena, LitValue::Int(n))
    }
    fn boolean(&mut self, b: bool) -> lang::arena::NodeId {
        lit(&mut self.arena, LitValue::Bool(b))
    }

    fn ap(&mut self, f: lang::arena::NodeId, a: lang::arena::NodeId) -> lang::arena::NodeId {
        app(&mut self.arena, f, a)
    }

    fn ap2(&mut self, f: lang::arena::NodeId, a: lang::arena::NodeId, b: lang::arena::NodeId)
        -> lang::arena::NodeId
    {
        let f1 = self.ap(f, a);
        self.ap(f1, b)
    }

    fn ap3(&mut self, f: lang::arena::NodeId, a: lang::arena::NodeId,
        b: lang::arena::NodeId, c: lang::arena::NodeId) -> lang::arena::NodeId
    {
        let f1 = self.ap(f, a);
        let f2 = self.ap(f1, b);
        self.ap(f2, c)
    }

    fn list(&mut self, xs: Vec<lang::arena::NodeId>) -> lang::arena::NodeId {
        let mut acc = self.pref(BuiltinId::Nil);
        for x in xs.into_iter().rev() {
            let cons = self.pref(BuiltinId::Cons);
            acc = self.ap2(cons, x, acc);
        }
        acc
    }

    fn run(&self, root: lang::arena::NodeId) -> Value {
        eval_program(&self.arena, &self.lib, root, vec![], FUEL).expect("eval ok")
    }

    fn run_with(&self, root: lang::arena::NodeId, args: Vec<Value>) -> Value {
        eval_program(&self.arena, &self.lib, root, args, FUEL).expect("eval ok")
    }
}

// --- arithmetic ---------------------------------------------------------

#[test]
fn add_1_2_eq_3() {
    let mut b = Builder::new();
    let one = b.int(1);
    let two = b.int(2);
    let add = b.pref(BuiltinId::Add);
    let prog = b.ap2(add, one, two);
    assert_eq!(b.run(prog), Value::Int(3));
}

#[test]
fn mul_3_4_eq_12() {
    let mut b = Builder::new();
    let mul = b.pref(BuiltinId::Mul);
    let three = b.int(3);
    let four = b.int(4);
    let prog = b.ap2(mul, three, four);
    assert_eq!(b.run(prog), Value::Int(12));
}

#[test]
fn div_by_zero_yields_bottom() {
    let mut b = Builder::new();
    let div = b.pref(BuiltinId::Div);
    let one = b.int(1);
    let zero = b.int(0);
    let prog = b.ap2(div, one, zero);
    let v = b.run(prog);
    assert!(v.is_bottom(), "expected Bottom, got {:?}", v);
}

#[test]
fn div_min_by_neg_one_does_not_overflow() {
    // `i64::MIN / -1` hardware-traps with plain `/`. The evaluator must
    // wrap (like Add/Sub/Mul) rather than panic the whole search.
    let mut b = Builder::new();
    let div = b.pref(BuiltinId::Div);
    let min = b.int(i64::MIN);
    let neg1 = b.int(-1);
    let prog = b.ap2(div, min, neg1);
    let v = b.run(prog);
    assert_eq!(v, Value::Int(i64::MIN), "wrapping_div should yield MIN");
}

#[test]
fn add_with_bool_yields_bottom() {
    // `add true 1` is now constructable (no static type-check). At runtime
    // it must surface as Bottom.
    let mut b = Builder::new();
    let add = b.pref(BuiltinId::Add);
    let truth = b.boolean(true);
    let one = b.int(1);
    let prog = b.ap2(add, truth, one);
    assert!(b.run(prog).is_bottom());
}

// --- booleans -----------------------------------------------------------

#[test]
fn not_false_eq_true() {
    let mut b = Builder::new();
    let nott = b.pref(BuiltinId::Not);
    let f = b.boolean(false);
    let prog = b.ap(nott, f);
    assert_eq!(b.run(prog), Value::Bool(true));
}

#[test]
fn and_true_false_eq_false() {
    let mut b = Builder::new();
    let and = b.pref(BuiltinId::And);
    let t = b.boolean(true);
    let f = b.boolean(false);
    let prog = b.ap2(and, t, f);
    assert_eq!(b.run(prog), Value::Bool(false));
}

// --- if / lazy if -------------------------------------------------------

#[test]
fn if_true_picks_then() {
    let mut b = Builder::new();
    let iff = b.pref(BuiltinId::If);
    let t = b.boolean(true);
    let one = b.int(1);
    let two = b.int(2);
    let prog = b.ap3(iff, t, one, two);
    assert_eq!(b.run(prog), Value::Int(1));
}

#[test]
fn if_short_circuits_unused_branch() {
    let mut b = Builder::new();
    let iff = b.pref(BuiltinId::If);
    let div = b.pref(BuiltinId::Div);
    let zero = b.int(0);
    let one = b.int(1);
    let bad = b.ap2(div, one, zero);
    let t = b.boolean(true);
    let prog = b.ap3(iff, t, zero, bad);
    assert_eq!(b.run(prog), Value::Int(0));
}

#[test]
fn if_propagates_bottom_in_chosen_branch() {
    let mut b = Builder::new();
    let iff = b.pref(BuiltinId::If);
    let div = b.pref(BuiltinId::Div);
    let one = b.int(1);
    let zero = b.int(0);
    let bad = b.ap2(div, one, zero);
    let t = b.boolean(true);
    let prog = b.ap3(iff, t, bad, one);
    assert!(b.run(prog).is_bottom());
}

// --- pairs --------------------------------------------------------------

#[test]
fn fst_pair_int_int() {
    let mut b = Builder::new();
    let pair = b.pref(BuiltinId::Pair);
    let one = b.int(1);
    let two = b.int(2);
    let p = b.ap2(pair, one, two);
    let fst = b.pref(BuiltinId::Fst);
    let prog = b.ap(fst, p);
    assert_eq!(b.run(prog), Value::Int(1));
}

#[test]
fn snd_pair_int_int() {
    let mut b = Builder::new();
    let pair = b.pref(BuiltinId::Pair);
    let one = b.int(1);
    let two = b.int(2);
    let p = b.ap2(pair, one, two);
    let snd = b.pref(BuiltinId::Snd);
    let prog = b.ap(snd, p);
    assert_eq!(b.run(prog), Value::Int(2));
}

// --- lists --------------------------------------------------------------

#[test]
fn cons_1_nil_eq_singleton() {
    let mut b = Builder::new();
    let one = b.int(1);
    let prog = b.list(vec![one]);
    assert_eq!(b.run(prog), Value::list_from(vec![Value::Int(1)]));
}

#[test]
fn list_1_2_3() {
    let mut b = Builder::new();
    let xs: Vec<_> = (1..=3).map(|i| b.int(i)).collect();
    let prog = b.list(xs);
    let expected = Value::list_from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(b.run(prog), expected);
}

// --- fold ---------------------------------------------------------------

#[test]
fn fold_sum_1_2_3_eq_6() {
    let mut b = Builder::new();
    let xs: Vec<_> = (1..=3).map(|i| b.int(i)).collect();
    let lst = b.list(xs);
    let z = b.int(0);
    let add = b.pref(BuiltinId::Add);
    let fold = b.pref(BuiltinId::Fold);
    let prog = b.ap3(fold, add, z, lst);
    assert_eq!(b.run(prog), Value::Int(6));
}

#[test]
fn fold_cons_nil_is_identity() {
    let mut b = Builder::new();
    let xs: Vec<_> = (1..=3).map(|i| b.int(i)).collect();
    let lst = b.list(xs);
    let nil = b.pref(BuiltinId::Nil);
    let cons = b.pref(BuiltinId::Cons);
    let fold = b.pref(BuiltinId::Fold);
    let prog = b.ap3(fold, cons, nil, lst);
    let expected = Value::list_from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(b.run(prog), expected);
}

#[test]
fn fold_length() {
    // fold (λx. λacc. add 1 acc) 0 [1, 2, 3, 4] = 4
    let mut b = Builder::new();
    let one = b.int(1);
    let add = b.pref(BuiltinId::Add);
    let acc_p = param(&mut b.arena, 0);
    let body = b.ap2(add, one, acc_p);
    let inner_lam = lambda(&mut b.arena, body);
    let outer_lam = lambda(&mut b.arena, inner_lam);

    let xs: Vec<_> = (1..=4).map(|i| b.int(i)).collect();
    let lst = b.list(xs);
    let z = b.int(0);
    let fold = b.pref(BuiltinId::Fold);
    let prog = b.ap3(fold, outer_lam, z, lst);
    assert_eq!(b.run(prog), Value::Int(4));
}

// --- unfold -------------------------------------------------------------

#[test]
fn unfold_range_0_to_3() {
    // unfold (λn. pair (pair n (add n 1)) (lt n 3)) 0  →  [0, 1, 2]
    let mut b = Builder::new();
    let n = param(&mut b.arena, 0);
    let one = b.int(1);
    let three = b.int(3);
    let add = b.pref(BuiltinId::Add);
    let lt = b.pref(BuiltinId::Lt);
    let pair = b.pref(BuiltinId::Pair);

    let n_plus_1 = b.ap2(add, n, one);
    let inner_pair = b.ap2(pair, n, n_plus_1);
    let cont = b.ap2(lt, n, three);
    let outer_pair = b.ap2(pair, inner_pair, cont);
    let step = lambda(&mut b.arena, outer_pair);

    let zero = b.int(0);
    let unfold = b.pref(BuiltinId::Unfold);
    let prog = b.ap2(unfold, step, zero);
    assert_eq!(
        b.run(prog),
        Value::list_from(vec![Value::Int(0), Value::Int(1), Value::Int(2)]),
    );
}

// --- combinators (K, B) -------------------------------------------------

#[test]
fn k_drops_second_arg() {
    let mut b = Builder::new();
    let k = b.pref(BuiltinId::K);
    let seven = b.int(7);
    let nine = b.int(9);
    let prog = b.ap2(k, seven, nine);
    assert_eq!(b.run(prog), Value::Int(7));
}

#[test]
fn b_composes() {
    let mut b = Builder::new();
    let bcomb = b.pref(BuiltinId::B);
    let add = b.pref(BuiltinId::Add);
    let mul = b.pref(BuiltinId::Mul);
    let one = b.int(1);
    let two = b.int(2);
    let five = b.int(5);
    let inc = b.ap(add, one);
    let dbl = b.ap(mul, two);
    let composed = b.ap2(bcomb, inc, dbl);
    let prog = b.ap(composed, five);
    assert_eq!(b.run(prog), Value::Int(11));
}

#[test]
fn length_via_fold_and_k() {
    let mut b = Builder::new();
    let k = b.pref(BuiltinId::K);
    let add = b.pref(BuiltinId::Add);
    let one = b.int(1);
    let zero = b.int(0);
    let fold = b.pref(BuiltinId::Fold);
    let inc = b.ap(add, one);
    let cb = b.ap(k, inc);
    let xs: Vec<_> = (1..=5).map(|i| b.int(i)).collect();
    let lst = b.list(xs);
    let prog = b.ap3(fold, cb, zero, lst);
    assert_eq!(b.run(prog), Value::Int(5));
}

#[test]
fn head_via_fold_and_k() {
    let mut b = Builder::new();
    let k = b.pref(BuiltinId::K);
    let zero = b.int(0);
    let fold = b.pref(BuiltinId::Fold);
    let xs: Vec<_> = [11, 22, 33].iter().map(|i| b.int(*i)).collect();
    let lst = b.list(xs);
    let prog = b.ap3(fold, k, zero, lst);
    assert_eq!(b.run(prog), Value::Int(11));
}

#[test]
fn map_add_one_via_fold_b_cons() {
    let mut b = Builder::new();
    let bcomb = b.pref(BuiltinId::B);
    let cons = b.pref(BuiltinId::Cons);
    let add = b.pref(BuiltinId::Add);
    let one = b.int(1);
    let nil = b.pref(BuiltinId::Nil);
    let fold = b.pref(BuiltinId::Fold);
    let inc = b.ap(add, one);
    let cb = b.ap2(bcomb, cons, inc);
    let xs: Vec<_> = [1, 2, 3].iter().map(|i| b.int(*i)).collect();
    let lst = b.list(xs);
    let prog = b.ap3(fold, cb, nil, lst);
    let expected = Value::list_from(vec![Value::Int(2), Value::Int(3), Value::Int(4)]);
    assert_eq!(b.run(prog), expected);
}

// --- function (curried) program with input ------------------------------

#[test]
fn user_function_add_one_program() {
    // λx. add x 1 — applied to 41 returns 42.
    let mut b = Builder::new();
    let x = param(&mut b.arena, 0);
    let one = b.int(1);
    let add = b.pref(BuiltinId::Add);
    let body = b.ap2(add, x, one);
    let f = lambda(&mut b.arena, body);
    assert_eq!(b.run_with(f, vec![Value::Int(41)]), Value::Int(42));
}

// --- polymorphic eq / lt ------------------------------------------------

#[test]
fn eq_int_int() {
    let mut b = Builder::new();
    let eq = b.pref(BuiltinId::Eq);
    let one = b.int(1);
    let one_again = b.int(1);
    let prog = b.ap2(eq, one, one_again);
    assert_eq!(b.run(prog), Value::Bool(true));
}

#[test]
fn eq_bool_bool() {
    let mut b = Builder::new();
    let eq = b.pref(BuiltinId::Eq);
    let t = b.boolean(true);
    let f = b.boolean(false);
    let prog = b.ap2(eq, t, f);
    assert_eq!(b.run(prog), Value::Bool(false));
}

#[test]
fn eq_list_int_list_int() {
    let mut b = Builder::new();
    let eq = b.pref(BuiltinId::Eq);
    let xs_n: Vec<_> = (1..=3).map(|i| b.int(i)).collect();
    let ys_n: Vec<_> = (1..=3).map(|i| b.int(i)).collect();
    let xs = b.list(xs_n);
    let ys = b.list(ys_n);
    let prog = b.ap2(eq, xs, ys);
    assert_eq!(b.run(prog), Value::Bool(true));
}

#[test]
fn eq_mismatched_variants_yields_bottom() {
    let mut b = Builder::new();
    let eq = b.pref(BuiltinId::Eq);
    let one = b.int(1);
    let t = b.boolean(true);
    let prog = b.ap2(eq, one, t);
    assert!(b.run(prog).is_bottom());
}

#[test]
fn lt_int_int() {
    let mut b = Builder::new();
    let lt = b.pref(BuiltinId::Lt);
    let one = b.int(1);
    let two = b.int(2);
    let prog = b.ap2(lt, one, two);
    assert_eq!(b.run(prog), Value::Bool(true));
}

#[test]
fn lt_char_char() {
    let mut b = Builder::new();
    let lt = b.pref(BuiltinId::Lt);
    let a = lit(&mut b.arena, LitValue::Char('a'));
    let z = lit(&mut b.arena, LitValue::Char('z'));
    let prog = b.ap2(lt, a, z);
    assert_eq!(b.run(prog), Value::Bool(true));
}

#[test]
fn lt_list_lex() {
    // [1, 2] < [1, 3]
    let mut b = Builder::new();
    let lt = b.pref(BuiltinId::Lt);
    let one_a = b.int(1);
    let two_a = b.int(2);
    let one_b = b.int(1);
    let three = b.int(3);
    let xs = b.list(vec![one_a, two_a]);
    let ys = b.list(vec![one_b, three]);
    let prog = b.ap2(lt, xs, ys);
    assert_eq!(b.run(prog), Value::Bool(true));
}

#[test]
fn lt_list_prefix_smaller() {
    // [1, 2] < [1, 2, 3] (shorter prefix sorts first)
    let mut b = Builder::new();
    let lt = b.pref(BuiltinId::Lt);
    let xs_n: Vec<_> = (1..=2).map(|i| b.int(i)).collect();
    let ys_n: Vec<_> = (1..=3).map(|i| b.int(i)).collect();
    let xs = b.list(xs_n);
    let ys = b.list(ys_n);
    let prog = b.ap2(lt, xs, ys);
    assert_eq!(b.run(prog), Value::Bool(true));
}

#[test]
fn lt_mismatched_variants_yields_bottom() {
    let mut b = Builder::new();
    let lt = b.pref(BuiltinId::Lt);
    let one = b.int(1);
    let t = b.boolean(true);
    let prog = b.ap2(lt, one, t);
    assert!(b.run(prog).is_bottom());
}

// --- runtime polymorphism (no static types) ----------------------------

#[test]
fn nil_can_be_used_as_int_list_or_bool_list() {
    // Without static types, `cons 1 nil` and `cons true nil` use the same
    // `nil` and `cons` nodes structurally — runtime polymorphism falls out
    // of the strict-evaluation semantics.
    let mut b = Builder::new();
    let nil = b.pref(BuiltinId::Nil);
    let cons = b.pref(BuiltinId::Cons);
    let one = b.int(1);
    let truth = b.boolean(true);
    let int_list = b.ap2(cons, one, nil);
    let bool_list = b.ap2(cons, truth, nil);
    assert_eq!(b.run(int_list), Value::list_from(vec![Value::Int(1)]));
    assert_eq!(b.run(bool_list), Value::list_from(vec![Value::Bool(true)]));
}

// --- structural sharing across uses ------------------------------------

#[test]
fn shared_subexpression_evaluates_once_to_same_result() {
    let mut b = Builder::new();
    let add = b.pref(BuiltinId::Add);
    let two = b.int(2);
    let three = b.int(3);
    let inner_a = b.ap2(add, two, three);
    let inner_b = b.ap2(add, two, three);
    assert_eq!(inner_a, inner_b, "hash-cons reuses node");
    let pair = b.pref(BuiltinId::Pair);
    let prog = b.ap2(pair, inner_a, inner_b);
    let v = b.run(prog);
    assert_eq!(v, Value::pair(Value::Int(5), Value::Int(5)));
}
