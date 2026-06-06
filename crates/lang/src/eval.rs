//! Strict evaluator with fuel.
//!
//! Closures represent partial applications of primitives or lambdas. `if`
//! is special-cased to be lazy (cond is evaluated, only the chosen branch
//! is evaluated) so that programs can have `bottom`-producing branches in
//! unused positions.
//!
//! `Value::Bottom` propagates through evaluation and represents a runtime
//! failure (e.g. `head []`, `div 1 0`, `add (Bool) (Int)`). It is *not*
//! an `Error` — `Error` signals out-of-fuel or a structurally invalid
//! program (a `Param` index that escapes its lambda). Type mismatches in
//! primitives produce `Bottom`.

use std::rc::Rc;

use crate::arena::{Arena, NodeId};
use crate::builtin::BuiltinId;
use crate::error::{Error, Result};
use crate::ir::{LitValue, NodeKind};
use crate::library::{Library, PrimId, PrimKind};

/// Runtime values. The "type" of a node is encoded in which variant of
/// `Value` it produces — there is no separate static type.
#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Float(f64),
    Char(char),
    List(Rc<Vec<Value>>),
    Pair(Rc<(Value, Value)>),
    Closure(Closure),
    /// Runtime failure with a debug message.
    Bottom(Rc<str>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            (Int(a), Int(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
            (Char(a), Char(b)) => a == b,
            (List(a), List(b)) => a.iter().eq(b.iter()),
            (Pair(a), Pair(b)) => a.0 == b.0 && a.1 == b.1,
            (Bottom(_), Bottom(_)) => true, // any two bottoms are equal
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Closure {
    pub head: ClosureHead,
    pub args: Vec<Value>,
    pub arity: usize,
}

#[derive(Clone, Debug)]
pub enum ClosureHead {
    Prim(PrimId),
    Lambda { body: NodeId, captured_env: Vec<Value> },
}

impl Value {
    pub fn from_lit(v: &LitValue) -> Value {
        match v {
            LitValue::Int(i) => Value::Int(*i),
            LitValue::Bool(b) => Value::Bool(*b),
            LitValue::Float(f) => Value::Float(*f),
            LitValue::Char(c) => Value::Char(*c),
        }
    }

    pub fn nil() -> Value { Value::List(Rc::new(Vec::new())) }
    pub fn list_from(vs: Vec<Value>) -> Value { Value::List(Rc::new(vs)) }
    pub fn pair(a: Value, b: Value) -> Value { Value::Pair(Rc::new((a, b))) }
    pub fn bottom(reason: impl Into<String>) -> Value {
        Value::Bottom(Rc::from(reason.into()))
    }

    pub fn as_int(&self) -> Option<i64> { if let Value::Int(i) = self { Some(*i) } else { None } }
    pub fn as_bool(&self) -> Option<bool> { if let Value::Bool(b) = self { Some(*b) } else { None } }
    pub fn as_list(&self) -> Option<&[Value]> { if let Value::List(xs) = self { Some(xs) } else { None } }
    pub fn is_bottom(&self) -> bool { matches!(self, Value::Bottom(_)) }
}

/// Evaluate `root` in `env` (a stack of bindings; innermost is last).
/// `Param { index: 0 }` reads the last element of `env`; index `N` reads
/// `N` items earlier.
pub fn eval(
    arena: &Arena, lib: &Library,
    root: NodeId, env: &[Value], fuel: &mut u32,
) -> Result<Value> {
    if *fuel == 0 { return Err(Error::OutOfFuel); }
    *fuel -= 1;

    match arena.kind(root) {
        NodeKind::Literal(v) => Ok(Value::from_lit(v)),
        NodeKind::Param { index } => {
            let i = *index as usize;
            if i >= env.len() {
                return Err(Error::Invalid(format!(
                    "free param: index {} but env depth {}", i, env.len()
                )));
            }
            Ok(env[env.len() - 1 - i].clone())
        }
        NodeKind::Lambda { body } => Ok(Value::Closure(Closure {
            head: ClosureHead::Lambda { body: *body, captured_env: env.to_vec() },
            args: Vec::new(),
            arity: 1,
        })),
        NodeKind::App { func, arg } => {
            // Lazy `if`: don't pre-evaluate then/else.
            if let Some((cond_n, then_n, else_n)) = unwrap_if(arena, lib, root) {
                let cond = eval(arena, lib, cond_n, env, fuel)?;
                return match cond {
                    Value::Bool(true) => eval(arena, lib, then_n, env, fuel),
                    Value::Bool(false) => eval(arena, lib, else_n, env, fuel),
                    Value::Bottom(s) => Ok(Value::Bottom(s)),
                    _ => Ok(Value::bottom("if: cond not Bool")),
                };
            }
            let f = eval(arena, lib, *func, env, fuel)?;
            let a = eval(arena, lib, *arg, env, fuel)?;
            apply(arena, lib, f, a, fuel)
        }
        NodeKind::PrimRef(p) => {
            let arity = lib.arity(*p);
            if arity == 0 {
                exec_prim(*p, Vec::new(), arena, lib, fuel)
            } else {
                Ok(Value::Closure(Closure {
                    head: ClosureHead::Prim(*p),
                    args: Vec::new(),
                    arity,
                }))
            }
        }
    }
}

/// Apply `f` to one argument.
pub fn apply(
    arena: &Arena, lib: &Library,
    f: Value, arg: Value, fuel: &mut u32,
) -> Result<Value> {
    if let Value::Bottom(s) = f { return Ok(Value::Bottom(s)); }
    if let Value::Bottom(s) = arg { return Ok(Value::Bottom(s)); }
    let (head, mut args, arity) = match f {
        Value::Closure(Closure { head, args, arity }) => (head, args, arity),
        _ => return Ok(Value::bottom("apply: non-function value")),
    };
    args.push(arg);
    if args.len() < arity {
        Ok(Value::Closure(Closure { head, args, arity }))
    } else {
        match head {
            ClosureHead::Prim(p) => exec_prim(p, args, arena, lib, fuel),
            ClosureHead::Lambda { body, mut captured_env } => {
                captured_env.extend(args);
                eval(arena, lib, body, &captured_env, fuel)
            }
        }
    }
}

/// Evaluate a complete program with a list of arguments.
pub fn eval_program(
    arena: &Arena, lib: &Library,
    root: NodeId, args: Vec<Value>, fuel: u32,
) -> Result<Value> {
    let mut fuel = fuel;
    let mut v = eval(arena, lib, root, &[], &mut fuel)?;
    for a in args {
        v = apply(arena, lib, v, a, &mut fuel)?;
    }
    Ok(v)
}

// -- Primitive dispatch ---------------------------------------------------

fn exec_prim(
    p: PrimId, args: Vec<Value>,
    arena: &Arena, lib: &Library, fuel: &mut u32,
) -> Result<Value> {
    let prim = lib.get(p);
    match &prim.kind {
        PrimKind::Builtin(b) => exec_builtin(*b, args, arena, lib, fuel),
        PrimKind::Learned { body, .. } => {
            // The body lives in the library's own arena, not the caller's.
            let mut v = eval(&lib.arena, lib, *body, &[], fuel)?;
            for a in args {
                v = apply(arena, lib, v, a, fuel)?;
            }
            Ok(v)
        }
    }
}

fn check_bottom(args: &[Value]) -> Option<Value> {
    for a in args {
        if let Value::Bottom(s) = a {
            return Some(Value::Bottom(s.clone()));
        }
    }
    None
}

fn exec_builtin(
    b: BuiltinId, args: Vec<Value>,
    arena: &Arena, lib: &Library, fuel: &mut u32,
) -> Result<Value> {
    use BuiltinId::*;

    if !matches!(b, If) {
        if let Some(bot) = check_bottom(&args) { return Ok(bot); }
    }

    match b {
        Add | Sub | Mul => {
            let a = match args[0].as_int() { Some(v) => v, None => return Ok(Value::bottom("add/sub/mul: non-Int")) };
            let c = match args[1].as_int() { Some(v) => v, None => return Ok(Value::bottom("add/sub/mul: non-Int")) };
            let r = match b {
                Add => a.wrapping_add(c),
                Sub => a.wrapping_sub(c),
                Mul => a.wrapping_mul(c),
                _ => unreachable!(),
            };
            Ok(Value::Int(r))
        }
        Div => {
            let a = match args[0].as_int() { Some(v) => v, None => return Ok(Value::bottom("div: non-Int")) };
            let c = match args[1].as_int() { Some(v) => v, None => return Ok(Value::bottom("div: non-Int")) };
            if c == 0 { return Ok(Value::bottom("div by zero")); }
            Ok(Value::Int(a / c))
        }
        Lt => poly_lt(&args[0], &args[1]),
        Eq => poly_eq(&args[0], &args[1]),
        Not => match args[0].as_bool() {
            Some(b) => Ok(Value::Bool(!b)),
            None => Ok(Value::bottom("not: non-Bool")),
        },
        And => {
            let a = match args[0].as_bool() { Some(v) => v, None => return Ok(Value::bottom("and: non-Bool")) };
            let c = match args[1].as_bool() { Some(v) => v, None => return Ok(Value::bottom("and: non-Bool")) };
            Ok(Value::Bool(a && c))
        }
        Or => {
            let a = match args[0].as_bool() { Some(v) => v, None => return Ok(Value::bottom("or: non-Bool")) };
            let c = match args[1].as_bool() { Some(v) => v, None => return Ok(Value::bottom("or: non-Bool")) };
            Ok(Value::Bool(a || c))
        }
        If => {
            // Reached only when `if` isn't fully syntactic at the App; fall
            // back to strict semantics. Lazy-if in `eval` covers the common
            // case.
            match &args[0] {
                Value::Bool(true) => Ok(args[1].clone()),
                Value::Bool(false) => Ok(args[2].clone()),
                Value::Bottom(s) => Ok(Value::Bottom(s.clone())),
                _ => Ok(Value::bottom("if: cond not Bool")),
            }
        }
        Pair => Ok(Value::pair(args[0].clone(), args[1].clone())),
        Fst => match &args[0] {
            Value::Pair(p) => Ok(p.0.clone()),
            _ => Ok(Value::bottom("fst: non-Pair")),
        },
        Snd => match &args[0] {
            Value::Pair(p) => Ok(p.1.clone()),
            _ => Ok(Value::bottom("snd: non-Pair")),
        },
        Nil => Ok(Value::nil()),
        Cons => {
            let h = args[0].clone();
            let t = match &args[1] {
                Value::List(xs) => xs.clone(),
                _ => return Ok(Value::bottom("cons: tail not List")),
            };
            let mut out = Vec::with_capacity(t.len() + 1);
            out.push(h);
            out.extend(t.iter().cloned());
            Ok(Value::List(Rc::new(out)))
        }
        Fold => {
            // Right-fold: fold f z [a,b,c] = f a (f b (f c z)).
            let f = args[0].clone();
            let z = args[1].clone();
            let xs = match &args[2] {
                Value::List(xs) => xs.clone(),
                _ => return Ok(Value::bottom("fold: arg 3 not List")),
            };
            let mut acc = z;
            for x in xs.iter().rev() {
                if *fuel == 0 { return Err(Error::OutOfFuel); }
                *fuel -= 1;
                let f1 = apply(arena, lib, f.clone(), x.clone(), fuel)?;
                acc = apply(arena, lib, f1, acc, fuel)?;
                if acc.is_bottom() { return Ok(acc); }
            }
            Ok(acc)
        }
        Unfold => {
            let step = args[0].clone();
            let mut seed = args[1].clone();
            let mut out: Vec<Value> = Vec::new();
            loop {
                if *fuel == 0 { return Err(Error::OutOfFuel); }
                *fuel -= 1;
                let r = apply(arena, lib, step.clone(), seed.clone(), fuel)?;
                let outer = match r {
                    Value::Pair(p) => p,
                    Value::Bottom(s) => return Ok(Value::Bottom(s)),
                    _ => return Ok(Value::bottom("unfold: step did not return Pair")),
                };
                let cont = match &outer.1 {
                    Value::Bool(b) => *b,
                    Value::Bottom(s) => return Ok(Value::Bottom(s.clone())),
                    _ => return Ok(Value::bottom("unfold: continuation not Bool")),
                };
                if !cont { break; }
                let inner = match &outer.0 {
                    Value::Pair(p) => (p.0.clone(), p.1.clone()),
                    Value::Bottom(s) => return Ok(Value::Bottom(s.clone())),
                    _ => return Ok(Value::bottom("unfold: inner not Pair")),
                };
                out.push(inner.0);
                seed = inner.1;
                if out.len() > 100_000 {
                    return Ok(Value::bottom("unfold: list too long"));
                }
            }
            Ok(Value::list_from(out))
        }
        K => Ok(args[0].clone()),
        B => {
            let f = args[0].clone();
            let g = args[1].clone();
            let x = args[2].clone();
            let gx = apply(arena, lib, g, x, fuel)?;
            apply(arena, lib, f, gx, fuel)
        }
        // `stop` is a *search-time sentinel*, not a runtime function.
        // The poser-search picks `App(stop, n)` to terminate
        // construction and returns `n` as the program — `n` doesn't
        // contain `stop` anywhere. If we ever reach this branch at
        // eval time, the search has constructed a program with a
        // nested stop (i.e. stop in non-outer position), which is a
        // bug. Return Bottom with a clear marker.
        Stop => Ok(Value::bottom("stop reached at eval — search bug")),
    }
}

// -- Polymorphic comparisons ---------------------------------------------

/// `eq`: deep value equality. Closures and `Bottom` produce `Bottom` —
/// closures because the language has no decidable function equality;
/// `Bottom` because failed evaluations don't carry a defined value to
/// compare against. Mismatched variants produce `Bottom` rather than
/// `false` so that wrong-typed candidates can't sneak through a search
/// by accident.
fn poly_eq(a: &Value, b: &Value) -> Result<Value> {
    if matches!(a, Value::Closure(_)) || matches!(b, Value::Closure(_)) {
        return Ok(Value::bottom("eq: closure"));
    }
    if !same_variant(a, b) {
        return Ok(Value::bottom("eq: variant mismatch"));
    }
    Ok(Value::Bool(a == b))
}

/// `lt`: lexicographic ordering on the structurally-comparable variants
/// (`Int`, `Float`, `Char`, `Bool`, `List`, `Pair`). NaN-safe via
/// `to_bits` for Floats. Closures, `Bottom`, and mismatched variants
/// produce `Bottom`.
fn poly_lt(a: &Value, b: &Value) -> Result<Value> {
    use Value::*;
    if matches!(a, Closure(_)) || matches!(b, Closure(_)) {
        return Ok(Value::bottom("lt: closure"));
    }
    if matches!(a, Bottom(_)) || matches!(b, Bottom(_)) {
        // `check_bottom` upstream already handles this for the common
        // path, but `poly_lt` is also called recursively for List/Pair.
        return Ok(Value::bottom("lt: bottom"));
    }
    if !same_variant(a, b) {
        return Ok(Value::bottom("lt: variant mismatch"));
    }
    match (a, b) {
        (Int(x), Int(y)) => Ok(Bool(x < y)),
        (Float(x), Float(y)) => Ok(Bool(x.to_bits() < y.to_bits())),
        (Char(x), Char(y)) => Ok(Bool(x < y)),
        (Bool(x), Bool(y)) => Ok(Bool(!x & y)),
        (List(xs), List(ys)) => list_lt(xs, ys),
        (Pair(p), Pair(q)) => {
            let first = poly_lt(&p.0, &q.0)?;
            match first {
                Bool(true) => Ok(Bool(true)),
                Bool(false) => {
                    let first_eq = poly_eq(&p.0, &q.0)?;
                    match first_eq {
                        Bool(true) => poly_lt(&p.1, &q.1),
                        Bool(false) => Ok(Bool(false)),
                        bot => Ok(bot),
                    }
                }
                bot => Ok(bot),
            }
        }
        _ => unreachable!("same_variant guarded above"),
    }
}

fn list_lt(xs: &[Value], ys: &[Value]) -> Result<Value> {
    let n = xs.len().min(ys.len());
    for i in 0..n {
        let lt = poly_lt(&xs[i], &ys[i])?;
        match lt {
            Value::Bool(true) => return Ok(Value::Bool(true)),
            Value::Bool(false) => {
                let eq = poly_eq(&xs[i], &ys[i])?;
                match eq {
                    Value::Bool(true) => continue,
                    Value::Bool(false) => return Ok(Value::Bool(false)),
                    bot => return Ok(bot),
                }
            }
            bot => return Ok(bot),
        }
    }
    Ok(Value::Bool(xs.len() < ys.len()))
}

fn same_variant(a: &Value, b: &Value) -> bool {
    use Value::*;
    matches!(
        (a, b),
        (Int(_), Int(_))
            | (Bool(_), Bool(_))
            | (Float(_), Float(_))
            | (Char(_), Char(_))
            | (List(_), List(_))
            | (Pair(_), Pair(_))
            | (Closure(_), Closure(_))
            | (Bottom(_), Bottom(_))
    )
}

// -- Lazy if helper -------------------------------------------------------

fn unwrap_if(
    arena: &Arena, lib: &Library, node: NodeId,
) -> Option<(NodeId, NodeId, NodeId)> {
    let (f1, e_else) = match arena.kind(node) {
        NodeKind::App { func, arg } => (*func, *arg),
        _ => return None,
    };
    let (f2, e_then) = match arena.kind(f1) {
        NodeKind::App { func, arg } => (*func, *arg),
        _ => return None,
    };
    let (f3, c) = match arena.kind(f2) {
        NodeKind::App { func, arg } => (*func, *arg),
        _ => return None,
    };
    let p = match arena.kind(f3) {
        NodeKind::PrimRef(p) => *p,
        _ => return None,
    };
    match &lib.get(p).kind {
        PrimKind::Builtin(BuiltinId::If) => Some((c, e_then, e_else)),
        _ => None,
    }
}
