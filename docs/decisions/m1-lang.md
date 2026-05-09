# Milestone 1 design decisions

A log of every non-obvious decision I made while implementing the `lang`
crate. Each is reversible — call any of these out and we'll change them.

## 1. De Bruijn indices, not levels, for `Param` nodes

`Param { index: u16 }` where `index = 0` is the innermost enclosing
lambda's parameter; index `N` reaches `N` lambdas outward.

Why: the evaluator's env stack matches indices directly (`env[len-1-i]`),
which is the simplest implementation. For closed combinator-style terms
(our common case) this works perfectly because there are no free variables
to make sharing semantically wrong.

If we later need to share open lambda subterms across different scopes,
levels would be safer. Combinator-style sidesteps the issue.

## 2. Wrapping arithmetic for `Int` ops

`add/sub/mul` use `wrapping_add/sub/mul`. Total functions; no panics.
Programs that care about overflow can guard with `lt`.

Alternative: saturating, or `Bottom` on overflow. Either is easy to switch.

## 3. Lazy `if` via syntactic detection

`eval` peels three `App` levels at every node to detect `if cond then
else`; if it matches, only the chosen branch is evaluated. So
`if true x bottom_expr` returns `x`.

Cost: a little evaluator complexity. Benefit: programs can have
bottom-producing branches in unused positions, which is a real value-add
for the search (a candidate that *might* solve the task isn't penalised
by an unrelated bottom in a dead branch).

## 4. `Bottom` is a `Value`, not an `Error`

`Error` signals type errors and out-of-fuel — bugs or hard limits.
`Value::Bottom(reason)` propagates through evaluation and represents
runtime failures like `div 1 0` or `head []`. Tasks that score programs
treat `Bottom` as not-solved, but programs that use bottom in dead
branches still work.

## 5. Canonical types via single-pass `rename`, not `Subst::apply`

The big one. `canonicalize` was originally implemented using
`Subst::apply`, which chases substitution chains. For a type like
`Var(1) -> Var(0)`, the canonicalisation subst is `{1 -> 0, 0 -> 1}` —
which creates a swap cycle that loops `apply` forever.

Fix: `rename(t, &map)` does a single-pass non-chasing replacement, used
for canonicalisation and `forall` quantification. `Subst::apply` keeps
chain semantics (which unification needs).

This bug took longer than I expected to find; I've added a panic guard
on apply depth in dev (now removed) and documented the invariant in
`ty.rs` near both functions.

## 6. Hash-cons keys include canonical type

Two `PrimRef(p)` nodes with the same `p` but different *result types*
(e.g. `cons` instantiated at `Int` vs `Bool`) hash differently and are
different nodes.

This is necessary: an `App(cons_int, ...)` has a different result type
than `App(cons_bool, ...)`, and we can't hash-collapse them or the search
loses the type-narrow versions of polymorphic primitives.

It does mean the primitive ref `cons` itself shows up multiple times in
the arena once it's been used at multiple types. That's fine — they're
all canonical hash-cons-equal at any given type.

## 7. `unfold` uses a `Pair Bool` continuation flag

`unfold : ∀a b. (b → Pair (Pair a b) Bool) → b → List a`.

I'd flagged this in the architecture docs. Confirming we shipped that
encoding because we have no sum types yet. If we add `Either`, we'll
switch `unfold`'s signature to `(b → Either (a, b) ())` or similar.

## 8. `free_vars` returns left-to-right traversal order

This determines canonicalisation: `Var(7) -> Var(3)` canonicalises with
`Var(7) → Var(0)` and `Var(3) → Var(1)`. Two alpha-equivalent types
produce the same canonical form because traversal order is fixed.

## 9. `Float` and `Char` exist in `LitValue` but no primitives use them

Forward-compat: when we add float arithmetic and string editing, the IR
doesn't change. Nothing to do here for v0.

## 10. Closures store explicit `arity`

For both `Prim` and `Lambda` heads. Partial application is just
`args.len() < arity`. Lambda arity is always 1 (curried). Primitive
arity is "number of leading arrows in the polytype body", computed once
when the closure is built.

## 11. Param stores its type explicitly

`param(arena, index, ty)` requires the type — we don't infer it from
context. This means lambda construction is `lambda(param_ty, body)` and
the body's `Param(0)` must have type `param_ty`. Verified by the type-
inference at the next `App` site.

For polymorphic lambdas (e.g. one passed to `fold` over a generic list),
the param type can be a fresh `Ty::Var(...)`. Our test suite only uses
concrete param types so far.

## 12. `Library::lookup` is linear scan

`lib.lookup(name)` is `O(n)`. Fine for 17 built-ins; will replace with a
HashMap once the library is allowed to grow.

## 13. `unfold` has a soft cap of 100,000 elements

Beyond which it returns `Bottom("unfold: list too long")`. This is a
friendliness measure — fuel would catch infinite loops eventually, but
"return bottom now" is a much nicer error in tests than "burned 100M
fuel cycles, now we're out". Fuel also fires from inside the loop.

## 14. Tests deliberately use the typed constructors directly

I considered adding a tiny parser for the test suite (so tests look like
`"add 1 2"`). I dropped that idea: the constructors are the future API
the search will use, so testing them directly is more representative.
There's a `pretty` module if we want a printer for debugging.

## 15. Arena clones types when interning

`arena.intern(kind, ty)` takes ownership of `kind` but clones `ty` for
the canonicalisation pass. Could be optimised to consume `ty` if hot.
I left it as a clone for now — measurable overhead is unlikely until
search runs.

---

## Things I deliberately *did not* do this milestone

- A parser. The architecture said maybe, but the typed constructors are
  the API of record.
- A reference Python evaluator for differential testing. The handwritten
  test suite plus property tests give similar coverage with less moving
  parts. We can add it later if differential signal is missing.
- Float / Char primitives. Deferred to the symbolic-regression and
  string-editing milestones respectively.
- Library serialisation. Programs serialise; libraries don't yet. Will
  matter when checkpointing the wake/sleep loop.

## Things to flag for review

If any of the calls above seem wrong, the most consequential ones to
revisit are:

1. **De Bruijn indices vs levels.** Current code is correct for closed
   terms; would need rework if we lean on shared open lambda fragments.
2. **Wrapping arithmetic.** Saturating or `Bottom`-on-overflow are both
   reasonable alternatives.
3. **Hash-cons including type.** Means a polymorphic primitive shows up
   multiple times if used at different types. Pro: search has access to
   type-specialised forms. Con: arena gets bigger. Worth measuring.
