# Language: IR, Evaluation

## Goals

1. A simple, total, purely-functional language whose programs are DAGs.
2. Hash-cons-friendly: structurally equal terms share storage.
3. A small set of built-in primitives that's easy to grow via library
   extraction.

## No static type system

Nodes carry only structural information (kind + children). There is no
type field, no type scheme, no unification.

What this gets us:

- **Runtime polymorphism falls out naturally.** `nil` is a single node
  whose runtime value is `Value::List([])`; `cons 1 nil` and
  `cons true nil` use the same `nil` and `cons` nodes structurally,
  with the `Value` variant determined by the literal at evaluation
  time.
- **Type errors become `Bottom`.** A primitive applied to runtime
  values of the wrong shape (e.g. `add Bool Int`, `head []`, `div 1 0`)
  returns `Value::Bottom(reason)`. Bottom propagates through
  evaluation and is treated as not-a-solution by `Task::score`.
- **Construction is total.** Any `App(f, a)` is admissible at the IR
  level; mismatches surface only at evaluation.

The trade-off: search has to evaluate candidates to discover which
combinations Bottom-out, and library extraction (M3) constrains
anti-unification holes by the runtime `Value` variants observed at
hole positions rather than by static type.

## Combinator calculus with optional explicit lambdas

The DAG is an **untyped lambda calculus**. We default to a
**combinator-style** subset: there are no free variables in the
top-level program, higher-order functions are passed by reference
(`PrimRef` / library reference), and recursion is supplied by
primitives like `fold` and `unfold`. This is the DreamCoder approach
and avoids the Y-combinator / divergence headaches.

Explicit `Lambda` nodes are still in the IR, with **de Bruijn indices**
for parameters (innermost-first; index 0 is the parameter of the
nearest enclosing `Lambda`). The evaluator handles them, and tests
exercise them, but the search never proposes bare lambdas as actions
(see `docs/09-questions.md` #3).

## Strict evaluation

Lazy evaluation interacts badly with our search-time scoring (we want to
detect when a program loops in finite time). Strict evaluation, with a
deterministic fuel budget at the interpreter level, gives a clean
termination contract. All primitives are total when given values of
the right runtime variant; lists are finite by construction.

## No mutation, no IO, no exceptions other than `bottom`

`bottom` represents runtime failure (e.g. `head []`, `add Bool Int`).
Tasks score `bottom` as a hard failure: a program that produces `bottom`
on any example does not solve the task.

## Concrete IR

```rust
// Node IDs are interned within an Arena.
pub type NodeId = u32;
pub type PrimId = u32;

pub enum NodeKind {
    Literal(LitValue),
    Param   { index: u16 },                // de Bruijn index
    Lambda  { body: NodeId },
    App     { func: NodeId, arg: NodeId },
    PrimRef(PrimId),                       // refers to Library
}

pub enum LitValue { Int(i64), Bool(bool), Float(f64), Char(char) }

pub struct Node {
    kind: NodeKind,
    hash: u64,        // structural hash over kind + children
}

pub struct Arena {
    nodes:  Vec<Node>,
    intern: HashMap<u64, Vec<NodeId>>,
}

pub struct Program { pub root: NodeId }
```

### Library

```rust
pub struct Primitive {
    pub name:  String,
    pub arity: u8,                        // curried args before exec
    pub kind:  PrimKind,
}

pub enum PrimKind {
    /// Implemented natively in the interpreter.
    Builtin(BuiltinId),
    /// A closed program in the library's own arena.
    Learned { body: NodeId, body_size: u32 },
}

pub struct Library {
    pub primitives: Vec<Primitive>,
    /// Storage for `Learned` primitive bodies. The evaluator routes
    /// `PrimKind::Learned { body }` through `&lib.arena`, not the
    /// caller's program arena. Skipped on serde — M3 will need to
    /// wire arena round-tripping at the same time as the first
    /// Learned primitive is created.
    pub arena: Arena,
}
```

The library is the only piece of state that grows with training. The
`primitives` vector is serialisable and versioned (every abstraction
sleep emits a new `LibraryVersion`); the embedded `arena` is currently
`#[serde(skip)]` and will be wired into the wire format alongside the
first Learned primitive.

### Initial built-ins (v0)

The seed library is intentionally minimal. Anything not listed should
be *derivable* by abstraction sleep. Catching the system rediscover
`map, filter, reverse` from `fold` is a useful litmus test that
wake/sleep is working.

- Numeric: `add, sub, mul, div : Int → Int → Int` (signatures shown for
  documentation; not enforced at construction).
- Comparison: `eq` and `lt` are runtime-polymorphic. `eq` is deep
  equality on every variant except `Closure`; mismatched variants
  produce `Bottom` rather than `false`. `lt` is lex/strict ordering on
  `Int | Float | Char | Bool | List | Pair`; mismatched variants and
  closures produce `Bottom`.
- Boolean: `if`, `not, and, or`.
- Pair: `pair, fst, snd`.
- List: `nil, cons`, `fold` (right-fold), `unfold`.
- Combinators: `k`, `b` — needed because the search doesn't propose
  bare lambdas, and most fold callbacks aren't directly expressible
  from the existing primitives.

That's it. With `fold` we recover `length, sum, reverse, map, filter,
head, tail, concat, zip` — and the system *should* discover them as
library entries. With `unfold` we recover `range` and similar
generators.

Float and Char primitives come with the symbolic-regression and
string-task milestones respectively. Grid primitives come with ARC.

## Hash-consing

The structural hash mixes:

- a discriminant byte for `NodeKind`,
- the `NodeId` of every child *(not its hash — `NodeId`s are already
  canonical thanks to interning, so this terminates)*,
- the byte representation of `LitValue` / `index` / `PrimId`.

`Arena::intern` checks `intern` first; on a hit, returns the existing
id; on a miss, appends. Edits must go through `intern` exclusively —
no in-place node mutation. Every program is in canonical form by
construction.

α-equivalence: because we use de Bruijn indices, two α-equivalent terms
have identical structures. We do not need a separate α-equivalence
pass.

η-equivalence: not handled at the IR level. We perform η-reduction as a
normalisation pass that the library extractor runs before mining
patterns.

## Evaluation

A simple recursive interpreter over the DAG, with three concerns:

1. **Sharing.** Memoize evaluated values by `NodeId` *for a given
   parameter environment*. A node that does not transitively depend on
   any `Param` is computed once per program. (M2's evaluator doesn't
   yet implement this memo; the search caches values at the pool level
   instead, which serves the same goal.)
2. **Termination.** A fuel counter decremented on each evaluation step
   *and* on each iteration of the recursive primitives `fold` and
   `unfold`. Exceeding fuel yields `Err(OutOfFuel)`. Default fuel is
   task-dependent (`tasks` provides it).
3. **Higher-order primitives.** `fold`, etc. take a function value;
   we represent values as a tagged union with a `Closure` variant that
   wraps `(NodeId, Env)` for lambdas or `(PrimId, args)` for partial
   primitive application. Closures store their `arity` explicitly.

```rust
pub enum Value {
    Int(i64), Bool(bool), Float(f64), Char(char),
    List(Vec<Value>), Pair(Box<(Value, Value)>),
    Closure(Closure),
    Bottom(String),
}

pub fn eval(arena: &Arena, lib: &Library, root: NodeId,
            args: &[Value], fuel: &mut u32) -> Result<Value>;
```

`Bottom` propagates through evaluation: applying anything to or as
`Bottom` returns `Bottom`.

### Lazy `if`

`if cond then else` is detected syntactically at the apex of three
chained `App` nodes and short-circuits: only the chosen branch is
evaluated. This lets candidates carry `Bottom`-producing dead branches
without disqualifying themselves. The bottom-up search's `apply`-based
incremental value computation does **not** preserve this laziness; see
[`03-search.md`](./03-search.md).

## Construction API

`crates/lang/src/construct.rs` exposes total constructors: any
well-formed `App` succeeds.

```rust
pub fn lit(arena: &mut Arena, v: LitValue) -> NodeId;
pub fn param(arena: &mut Arena, index: u16) -> NodeId;
pub fn lambda(arena: &mut Arena, body: NodeId) -> NodeId;
pub fn app(arena: &mut Arena, func: NodeId, arg: NodeId) -> NodeId;
pub fn prim_ref(arena: &mut Arena, p: PrimId) -> NodeId;
```

## Serialisation

A program serialises to an ordered list of nodes (topologically sorted)
with each node referencing children by their *position in the list*.
Trivially round-trippable; diffs are straightforward. JSON for
development, a compact binary form (postcard / bincode) for hot paths.

## Open questions for the user

See [09-questions.md](./09-questions.md) for the full list.
