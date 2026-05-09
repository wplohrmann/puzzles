//! Strict evaluator with fuel.
//!
//! Closures represent partial applications of primitives or lambdas. `if`
//! is special-cased to be lazy (cond is evaluated, only the chosen branch
//! is evaluated) so that programs can have `bottom`-producing branches in
//! unused positions.
//!
//! `Value::Bottom` propagates through evaluation and represents a runtime
//! failure (e.g. `head []`, `div 1 0`). It is *not* an `Error`. Errors
//! signal type errors (which shouldn't happen for well-typed programs) or
//! out-of-fuel.

use std::rc::Rc;

use crate::arena::{Arena, NodeId};
use crate::builtin::BuiltinId;
use crate::error::{Error, Result};
use crate::ir::{LitValue, NodeKind};
use crate::library::{Library, PrimId, PrimKind};

/// Runtime values. Not serde-serialisable directly because closures hold
/// arena `NodeId`s; round-trip serialisation operates on programs, not
/// values.
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
        NodeKind::Lambda { body, .. } => Ok(Value::Closure(Closure {
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
                    _ => Err(Error::PrimitiveTypeMismatch("if")),
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
        _ => return Err(Error::PrimitiveTypeMismatch("apply: function value required")),
    };
    args.push(arg);
    if args.len() < arity {
        Ok(Value::Closure(Closure { head, args, arity }))
    } else {
        match head {
            ClosureHead::Prim(p) => exec_prim(p, args, arena, lib, fuel),
            ClosureHead::Lambda { body, mut captured_env } => {
                // arity is 1 for lambdas, so args has exactly one element.
                captured_env.extend(args);
                eval(arena, lib, body, &captured_env, fuel)
            }
        }
    }
}

/// Evaluate a complete program with a list of arguments. `eval_program(p, [])`
/// runs a value-typed program; `eval_program(p, [a, b])` runs a 2-arg
/// curried function.
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
            // Run the learned body and apply args.
            let mut v = eval(arena, lib, *body, &[], fuel)?;
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
            let a = args[0].as_int().ok_or(Error::PrimitiveTypeMismatch(b.name()))?;
            let c = args[1].as_int().ok_or(Error::PrimitiveTypeMismatch(b.name()))?;
            // Wrapping arithmetic so overflow is total. Programs that care
            // about overflow can guard with `lt` etc.
            let r = match b {
                Add => a.wrapping_add(c),
                Sub => a.wrapping_sub(c),
                Mul => a.wrapping_mul(c),
                _ => unreachable!(),
            };
            Ok(Value::Int(r))
        }
        Div => {
            let a = args[0].as_int().ok_or(Error::PrimitiveTypeMismatch("div"))?;
            let c = args[1].as_int().ok_or(Error::PrimitiveTypeMismatch("div"))?;
            if c == 0 { return Ok(Value::bottom("div by zero")); }
            Ok(Value::Int(a / c))
        }
        Lt => {
            let a = args[0].as_int().ok_or(Error::PrimitiveTypeMismatch("lt"))?;
            let c = args[1].as_int().ok_or(Error::PrimitiveTypeMismatch("lt"))?;
            Ok(Value::Bool(a < c))
        }
        Eq => {
            let a = args[0].as_int().ok_or(Error::PrimitiveTypeMismatch("eq"))?;
            let c = args[1].as_int().ok_or(Error::PrimitiveTypeMismatch("eq"))?;
            Ok(Value::Bool(a == c))
        }
        Not => Ok(Value::Bool(!args[0].as_bool().ok_or(Error::PrimitiveTypeMismatch("not"))?)),
        And => {
            let a = args[0].as_bool().ok_or(Error::PrimitiveTypeMismatch("and"))?;
            let c = args[1].as_bool().ok_or(Error::PrimitiveTypeMismatch("and"))?;
            Ok(Value::Bool(a && c))
        }
        Or => {
            let a = args[0].as_bool().ok_or(Error::PrimitiveTypeMismatch("or"))?;
            let c = args[1].as_bool().ok_or(Error::PrimitiveTypeMismatch("or"))?;
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
                _ => Err(Error::PrimitiveTypeMismatch("if")),
            }
        }
        Pair => Ok(Value::pair(args[0].clone(), args[1].clone())),
        Fst => match &args[0] {
            Value::Pair(p) => Ok(p.0.clone()),
            _ => Err(Error::PrimitiveTypeMismatch("fst")),
        },
        Snd => match &args[0] {
            Value::Pair(p) => Ok(p.1.clone()),
            _ => Err(Error::PrimitiveTypeMismatch("snd")),
        },
        Nil => Ok(Value::nil()),
        Cons => {
            let h = args[0].clone();
            let t = match &args[1] {
                Value::List(xs) => xs.clone(),
                _ => return Err(Error::PrimitiveTypeMismatch("cons")),
            };
            let mut out = Vec::with_capacity(t.len() + 1);
            out.push(h);
            out.extend(t.iter().cloned());
            Ok(Value::List(Rc::new(out)))
        }
        Fold => {
            // fold f z xs
            let f = args[0].clone();
            let z = args[1].clone();
            let xs = match &args[2] {
                Value::List(xs) => xs.clone(),
                _ => return Err(Error::PrimitiveTypeMismatch("fold")),
            };
            let mut acc = z;
            for x in xs.iter() {
                if *fuel == 0 { return Err(Error::OutOfFuel); }
                let f1 = apply(arena, lib, f.clone(), x.clone(), fuel)?;
                acc = apply(arena, lib, f1, acc, fuel)?;
                if acc.is_bottom() { return Ok(acc); }
            }
            Ok(acc)
        }
        Unfold => {
            // unfold step seed
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
                    _ => return Err(Error::PrimitiveTypeMismatch("unfold")),
                };
                let cont = match &outer.1 {
                    Value::Bool(b) => *b,
                    Value::Bottom(s) => return Ok(Value::Bottom(s.clone())),
                    _ => return Err(Error::PrimitiveTypeMismatch("unfold")),
                };
                if !cont { break; }
                let inner = match &outer.0 {
                    Value::Pair(p) => (p.0.clone(), p.1.clone()),
                    Value::Bottom(s) => return Ok(Value::Bottom(s.clone())),
                    _ => return Err(Error::PrimitiveTypeMismatch("unfold")),
                };
                out.push(inner.0);
                seed = inner.1;
                // Cap: bail before runaway loops produce massive lists. The
                // outer fuel will eventually trigger, but a soft cap on list
                // length is friendlier in tests.
                if out.len() > 100_000 {
                    return Ok(Value::bottom("unfold: list too long"));
                }
            }
            Ok(Value::list_from(out))
        }
    }
}

// -- Lazy if helper -------------------------------------------------------

/// If `node` looks like `if cond then else`, return the three sub-nodes.
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
