# Language: IR, Types, Evaluation

## Goals

1. A simple, total, purely-functional language whose programs are DAGs.
2. Hash-cons-friendly: structurally equal terms share storage.
3. A type system rich enough that random typed expansions are usually
   meaningful (saves the search a huge amount of work).
4. A small set of built-in primitives that's easy to grow via library
   extraction.

## Design choices, with rationale

### Combinator calculus with optional explicit lambdas

The DAG is a typed lambda calculus. To keep DAG sharing trivially sound we
default to a **combinator-style** subset: there are no free variables,
higher-order functions are passed by reference (`PrimRef`/library reference),
and recursion is supplied by primitives like `fold` and `map`. This is the
DreamCoder approach and avoids the Y-combinator / divergence headaches.

Explicit `Lambda` nodes are still in the IR, with **de Bruijn levels** for
parameters. Levels (counted from the root outward) are stable under sharing;
de Bruijn *indices* (counted from the use site) are not, since the same
subterm has different free-variable indices when shared at two different
depths. Levels make hash-consing α-equivalence-correct by construction.

We expect most programs the system finds to be lambda-free — the search will
prefer composing primitives — but we keep the option open.

### Types: HM-lite, no general inference

Every primitive (built-in or library entry) carries a hand-written
**polytype**: a closed type with universally-quantified type variables, e.g.
`map : ∀a b. (a → b) → List a → List b`. Search constructs programs by
combining nodes of known type, so the only typing operation we ever need is
**unification at `Apply` sites**: instantiate the function's polytype with
fresh type variables, unify its argument type with the candidate
argument's type, propagate the substitution. There is no type inference
for unknown terms because there are no unknown terms — every node's type
is determined when it is constructed.

This sidesteps full Hindley-Milner inference (which is for inferring types
of user-written programs) and lands at maybe 100 lines of unification
code. Each `SearchState` carries a current type-variable substitution that
threads through the search.

Why types matter even though the network could learn them:
- Action-space reduction at enumeration time: unification rejects ~10–30×
  more candidates than naive combination, and this happens before any NN
  call.
- Library extraction needs types: anti-unification creates parameter
  holes whose type *is* what defines them. Untyped anti-unification
  generates junk primitives.
- The network can still learn finer-grained constraints on top of types
  (e.g. expected list lengths, value ranges).

Concrete type constructors at the start:
- `Int`, `Bool`, `Float`, `Char`: scalars.
- `Pair<A, B>`: 2-tuples; n-tuples nest.
- `List<T>`: ordered sequences.
- `Fn<A, B>`: curried function type (built from `App` + the type
  constructor; not exposed to users).
- `Grid<T>`: deferred until the ARC milestone.

Deliberately **not** in v0:
- `Maybe<T>`: failure is `Bottom` (see below) so we don't need an option
  type for most programs.
- Sum / enum types: would require pattern matching in the IR. Deferred.
- User-defined ADTs: ditto.

Trees, dictionaries, sets and other shapes are encodable in the existing
constructors (`Tree<T> ≡ Pair<T, List<Tree<T>>>`, `Dict<K,V> ≡
List<Pair<K,V>>`); we add primitives for them only if a domain demands.

### Strict evaluation

Lazy evaluation interacts badly with our search-time scoring (we want to
detect when a program loops in finite time). Strict evaluation, with a
deterministic fuel budget at the interpreter level, gives a clean termination
contract. All primitives are total when given values of the right type; lists
are finite by construction.

### No mutation, no IO, no exceptions other than `bottom`

`bottom` represents runtime failure (e.g. `head []`). Tasks score `bottom` as
a hard failure: a program that produces `bottom` does not solve the task.
This is simpler than encoding `Maybe T` everywhere.

## Concrete IR

```rust
// Type system
pub enum Type {
    Var(TyVar),                // unification variable, used during inference
    Con(TyCon, Vec<Type>),     // type constructor application
}

pub enum TyCon { Int, Bool, Float, Char, List, Pair, Fn, Grid /* ... */ }

pub struct TypeScheme {
    quantified: Vec<TyVar>,
    body: Type,
}

// Node IDs are interned within an Arena
pub type NodeId = u32;
pub type PrimId = u32;

pub enum NodeKind {
    Literal(LitValue),                  // typed by the LitValue itself
    Param { level: u16 },               // de Bruijn level
    Lambda { param_ty: Type, body: NodeId },
    App   { func: NodeId, arg: NodeId },
    PrimRef(PrimId),                    // refers to Library
}

pub enum LitValue { Int(i64), Bool(bool), Float(f64), Char(char), Nil }

pub struct Node {
    kind: NodeKind,
    ty:   Type,
    hash: u64,            // structural hash including ty
}

pub struct Arena {
    nodes:  Vec<Node>,
    intern: HashMap<u64, NodeId>,
}

pub struct Program { pub root: NodeId }
```

### Library

```rust
pub struct Primitive {
    pub name: String,
    pub ty:   TypeScheme,
    pub kind: PrimKind,
}

pub enum PrimKind {
    /// Built-in: implemented natively in the interpreter.
    Builtin(BuiltinId),
    /// Learned: a closed program in the library's own arena.
    Learned { body: NodeId, body_size: u32 },
}

pub struct Library {
    pub primitives: Vec<Primitive>,
    pub arena:      Arena,            // shared arena for all `Learned` bodies
}
```

The library is the only piece of state that grows with training. It is fully
serializable and versioned (every abstraction sleep emits a new
`LibraryVersion`).

### Initial built-ins (proposed v0)

The seed library is intentionally minimal. Anything not listed should be
*derivable* by abstraction sleep. Catching the system rediscover `map`,
`filter`, `reverse` from `fold` is a useful litmus test that wake/sleep
is working.

- Numeric: `add, sub, mul, div : Int → Int → Int`, `lt, eq : Int → Int → Bool`.
- Boolean: `if : ∀a. Bool → a → a → a`, `not, and, or`.
- Pair: `pair : ∀a b. a → b → Pair a b`, `fst : ∀a b. Pair a b → a`,
  `snd : ∀a b. Pair a b → b`.
- List: `nil : ∀a. List a`, `cons : ∀a. a → List a → List a`,
  `fold : ∀a b. (a → b → b) → b → List a → b`,
  `unfold : ∀a b. (b → Pair (Pair a b) Bool) → b → List a`.

That's it. With `fold` we recover `length`, `sum`, `reverse`, `map`,
`filter`, `head`, `tail`, `concat`, `zip` — and the system *should*
discover them as library entries. With `unfold` we recover `range` and
similar generators.

Float and Char primitives come with the symbolic-regression and
string-task milestones respectively. Grid primitives come with ARC. The
point of starting this small is so we can *measure* the abstraction-sleep
machinery doing its job.

## Hash-consing

The structural hash mixes:

- a discriminant byte for `NodeKind`,
- the `NodeId` of every child *(not its hash — `NodeId`s are already canonical
  thanks to interning, so this terminates)*,
- the byte representation of `LitValue` / `level` / `PrimId`,
- the type of the node (so polymorphic instantiations don't collide).

`Arena::insert` checks `intern` first; on a hit, returns the existing id; on a
miss, appends. Edits must go through `insert` exclusively — no in-place node
mutation. This means *every* program is in canonical form by construction.

α-equivalence: because we use de Bruijn levels, two α-equivalent terms have
identical structures. We do not need a separate α-equivalence pass.

η-equivalence: not handled at the IR level. We perform η-reduction as a
normalization pass that the library extractor runs before mining patterns.

## Evaluation

A simple recursive interpreter over the DAG, with three concerns to address:

1. **Sharing.** Memoize evaluated values by `NodeId` *for a given parameter
   environment*. A node that does not transitively depend on any `Param` is
   computed once per program.
2. **Termination.** A fuel counter decremented on each `App`. Exceeding fuel
   yields `bottom`. Default fuel is task-dependent (`tasks` provides it).
3. **Higher-order primitives.** `map`, `fold`, etc. take a function value;
   we represent values as a tagged union with a `Closure` variant that wraps
   `(NodeId, Env)`. Calling a closure resumes evaluation with the env extended.

```rust
pub enum Value {
    Int(i64), Bool(bool), Float(f64), Char(char),
    List(Vec<Value>), Pair(Box<(Value, Value)>),
    Closure(NodeId, Env),
    Bottom,
}

pub fn eval(arena: &Arena, lib: &Library, root: NodeId, args: &[Value],
            fuel: &mut u32) -> Value;
```

## Construction API

The arena exposes a typed-constructor API so callers cannot create ill-typed
nodes. Every constructor is fallible.

```rust
impl Arena {
    pub fn lit(&mut self, v: LitValue) -> NodeId;
    pub fn param(&mut self, level: u16, ty: Type) -> NodeId;
    pub fn lambda(&mut self, param_ty: Type, body: NodeId) -> Result<NodeId>;
    pub fn app(&mut self, func: NodeId, arg: NodeId) -> Result<NodeId>;
    pub fn prim_ref(&mut self, lib: &Library, p: PrimId) -> NodeId;
}
```

`app` runs unification, instantiates polymorphic types, and refuses if the
function's argument type doesn't match.

## Serialization

A program serialises to an ordered list of nodes (topologically sorted) with
each node referencing children by their *position in the list*. This is
trivially round-trippable and makes diffing programs straightforward. JSON
for development, a compact binary form (postcard / bincode) for hot paths.

## Open questions for the user

See [09-questions.md](./09-questions.md) for the full list. Language-relevant
ones: floats and gradient-based literal optimisation; whether to permit
explicit lambdas at all; whether ARC-grids should be a built-in or a library
addition.
