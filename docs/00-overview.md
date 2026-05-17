# Architecture Overview

This project builds a neurally-guided program-synthesis system in the spirit of
DreamCoder, but with a DAG-based core language and a feed-forward graph
network that exploits hash-consing for cheap incremental inference.

The system is decomposed into five intentionally loosely-coupled subsystems:

| # | Subsystem | Role | Crate |
|---|-----------|------|-------|
| 1 | **Language** | Defines programs as DAGs, evaluates them, hash-conses them | `lang` |
| 2 | **Neural** | Embeds each DAG node and scores candidate App pairs `q(f, a)` | `neural` |
| 3 | **Search** | Explores the program space using neural guidance, returns programs | `search` |
| 4 | **Library** | Mines recurring fragments and promotes them to new primitives | `library` |
| 5 | **Tasks** | Defines what a program is supposed to do and how we score it | `tasks` |

A sixth crate, `training`, drives the wake/sleep loop and owns no domain logic
of its own. A `cli` crate wires the binaries.

Each subsystem talks to its neighbours through a small set of trait objects /
serialized structs documented in its own doc file:

- [01-language.md](./01-language.md): IR, evaluation, serialization
- [02-neural.md](./02-neural.md): embedding network, `q(f, a)` head, caching
- [03-search.md](./03-search.md): search algorithm and frontier
- [04-library.md](./04-library.md): compression and refactoring
- [05-tasks.md](./05-tasks.md): task families and scoring
- [06-training.md](./06-training.md): wake/sleep training loop
- [07-testing.md](./07-testing.md): testing strategy
- [08-roadmap.md](./08-roadmap.md): milestones and order of work
- [09-questions.md](./09-questions.md): open design questions for you

## High-level data flow

```
                 ┌─────────────┐
                 │   Tasks     │  defines: spec, scoring, training/test split
                 └──────┬──────┘
                        │ task
                        ▼
                 ┌─────────────┐  output: best program, search statistics
        ┌───────►│   Search    │
        │        └──────┬──────┘
        │               │ partial programs
        │               ▼
        │        ┌─────────────┐  output: q(f, a) for App candidates
        │        │   Neural    │
        │        └──────┬──────┘
        │               │ node embeddings (cached by hash)
        │               ▼
        │        ┌─────────────┐  output: DAGs, evaluation
        │        │  Language   │◄────────────┐
        │        └─────────────┘             │
        │                                     │ primitives, programs
        │        ┌─────────────┐             │
        └────────│   Library   │─────────────┘
                 └─────────────┘  consumes solved programs, emits new primitives
```

The training loop alternates three phases (after [Ellis et al. 2021]):

1. **Wake.** For every task in the current batch, run `Search` (using the
   current `Neural` weights and `Library`) to find a program that solves it.
   Keep the best programs found per task in a *replay buffer*.
2. **Abstraction sleep.** Take the replay buffer and run `Library`'s
   compression pass to mine new primitives. The library grows; programs are
   rewritten in terms of the new library; the replay buffer is updated.
3. **Dream sleep.** Train `Neural` on (a) the replay buffer (real solved
   tasks → real programs) and (b) "dreams": programs sampled from the prior
   under the current library, run on synthetic inputs, then used as supervised
   (task, program) pairs.

Repeat. The library ratchets monotonically toward the domain; the network
ratchets toward better proposals; the search horizon for each task therefore
shrinks even as task complexity grows.

## Core invariants the design guarantees

The following are properties we want to hold by construction so that the
subsystems remain decoupled and we can test each in isolation:

1. **No static type system.** Nodes carry only structural information;
   mismatches surface as `Value::Bottom` at evaluation time. See
   [01-language.md](./01-language.md).
2. **Hash-consing is canonical.** Two structurally identical programs share
   the same `NodeId`. `Neural` keys its caches on `NodeId`/hash. `Library`
   detects duplication structurally.
3. **Per-node embeddings split into a structural and a value component.**
   The structural component depends only on upstream nodes (and the current
   library) — invariant across tasks, cached globally. The value component
   depends on the node's concrete outputs on the task's examples — task-
   specific, cached per task. Per-edit cost in either cache is O(1) in
   graph size, O(n) in example count. See [02-neural.md](./02-neural.md).
4. **Search builds programs bottom-up out of evaluable nodes.** Every node
   in the search pool is fully formed and runnable on the task's inputs.
   We don't track partial-program-with-holes state; the whole pool is a
   set of complete sub-expressions, and a "solution" is any pool node
   whose values match the task's targets.
5. **Library entries are first-class programs.** A library primitive is
   a closed term with a fixed arity. Adding/removing entries is a pure
   data update — no code generation, no recompilation.

## Why a DAG and not a tree

A program AST is conceptually a tree, but a typical program contains many
identical subterms. Hash-consing collapses these into a single shared node,
so a program with 100 textual occurrences of `add 1` is one node. This buys
us:

- **O(1) structural equality and a free fingerprint** — two programs match
  iff their root `NodeId`s match.
- **Cache reuse across edits and across tasks** — embedding a shared node
  once benefits every program that contains it.
- **Cheap pattern-mining for library extraction** — recurring subgraphs
  *are* repeated nodes; we count occurrences trivially.

The cost is that the language must be designed so that DAG sharing preserves
semantics: this holds for purely-functional, terminating, side-effect-free
combinator calculi, which is exactly what we want. See `01-language.md` for
details.

## Glossary

- **Primitive** — a built-in function (`map`, `add`, `if`, etc.) or a library
  entry. Either is referenced from a program by a `PrimRef` node.
- **Library** — the current set of primitives. Starts as the built-ins; grows
  with abstraction sleep.
- **Frontier** — the priority queue of `App(f, a)` candidates the search
  has scored but not yet popped.
- **Replay** — a (task, program) pair where the program was actually found by
  search and verified to solve the task.
- **Dream / fantasy** — a (task, program) pair where the program was sampled
  from the prior and the task was constructed by running it.
- **MDL score** — the objective: solve the task with as few nodes as possible,
  counting both the program and any *novel* library entries it depends on.
- **PCFG / bigram prior** — the per-primitive probability under the library,
  optionally conditioned on the parent production. Used by enumeration as the
  default scoring before the network has been trained.
