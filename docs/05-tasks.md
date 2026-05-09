# Tasks

## What is a Task

A `Task` is anything that takes a candidate program and returns a score.
The system never assumes anything else about it — this is the cleanest
decoupling boundary in the project, and many domains are added as new
implementations of this single trait.

```rust
pub trait Task: Send + Sync {
    /// The type the program must have.
    fn target_type(&self) -> TypeScheme;

    /// True iff the program *exactly* solves the task.
    /// Most callers check this first.
    fn solves(&self, lang: &Language, prog: &Program) -> bool;

    /// A continuous quality score in [0, 1]. Used by search heuristics that
    /// want partial credit (e.g. matched 4 of 5 examples). For a binary
    /// task this is just `solves` cast to f32.
    fn score(&self, lang: &Language, prog: &Program) -> f32;

    /// A task-specific encoding for the neural task encoder. Returned as a
    /// trait object so each domain can pick its own representation.
    fn encoding(&self) -> Box<dyn TaskEncoding>;

    /// Deterministic id (e.g. for replay logs).
    fn id(&self) -> TaskId;
}
```

`Language` here is a small handle over the arena + library + interpreter,
not a separate world.

## Task families

Five families, listed in the order we'll implement them.

### 1. Function approximation (the simplest)

The task gives `n` (input, output) pairs. The program must be a function
`Input → Output` that produces all `n` outputs exactly.

```rust
pub struct ListExamplesTask {
    pub examples: Vec<(Value, Value)>,
    pub fuel: u32,
}

impl Task for ListExamplesTask {
    fn solves(&self, l: &Language, p: &Program) -> bool {
        self.examples.iter().all(|(x, y)|
            l.eval(p, &[x.clone()], self.fuel) == *y)
    }
    /* ... */
}
```

Domains realised this way:
- **List manipulation** — sort, reverse, dedup, sum, length, mode-of-list,
  etc. (the DreamCoder list domain).
- **String editing** — last-name first-name reformat, csv→tsv, capitalise
  initials, etc. Very rich source of compressible structure once a few
  primitives like `split_on`, `join_with`, `to_upper` exist.
- **Symbolic regression** — given (x_i, y_i) pairs of floats, find a
  closed-form `Float → Float` function. Useful for the gradient-based
  literal-optimisation experiments later.

### 2. Reconstruction (the auto-encoder framing)

Given outputs `y_1, ..., y_n` (no inputs), find:
- a shared program `f`,
- per-example latents `z_1, ..., z_n` (small inputs),
- such that `f(z_i) = y_i` for all i,
- minimising `size(f) + Σ size(z_i)`.

This is the classic MDL / minimum-information-encoding framing: the program
is a decoder, the latents are the codes. The minimiser favours pushing
shared structure into `f` and idiosyncratic structure into the `z_i`.

Implementation: search jointly. The naive way is to nest two searches —
outer over `f`, inner over `(z_1, ..., z_n)`. We instead let the search
expand a *single* program of type `(Code → Output)` plus a single
`Code` per example, with shared parameters at the outer level. Concretely
the task target type becomes `List<Code> → List<Output>` and the program
is forced to be of the form `map f codes` for some `f`. Rough but
effective.

### 3. Hallucination / dreams (training data, not user-facing)

For each abstraction sleep iteration:

1. Sample a program `ρ ~ P(· | L)` from the prior under the current library.
2. Sample inputs `x_1, ..., x_n` from a type-appropriate distribution.
3. Compute `y_i = ρ(x_i)` (skip if `bottom`).
4. Emit `(ρ, [(x_i, y_i)])` as a synthetic supervised example for the
   recognition network.

Hallucinated tasks bias the network toward whatever the *current library*
makes likely. They're not used for end-of-iteration evaluation; their job
is to keep the network calibrated.

### 4. ARC-AGI (the hard target)

Each ARC task is `n` example grid-pairs plus held-out test grids. We treat
ARC as function approximation over the `Grid` type. The grid type and a
suite of grid primitives (rotate, transpose, flood-fill, find-objects,
mask, recolor, paste, …) ship as a domain pack on top of the v0 list
language.

Realistically ARC is *aspirational*. The system should learn the simpler
families first and grow primitives that ARC needs — exactly the wake-sleep
argument. Hard tasks where no current program solves the task contribute
nothing to abstraction sleep but everything to motivation and benchmarking.
We measure on ARC every iteration but don't expect early success.

The submodule `ARC-AGI/` is already in the repo; the existing Python
solutions in `solutions/` are a goldmine for the primitive vocabulary —
inspecting them tells us what abstractions to seed manually before training.

### 5. Curriculum and held-out sets

For non-ARC families we generate large training task pools by *programmatic
sampling*. A list-task generator might:

- Pick a target program by hand (e.g. `sum`, `reverse`, `every-other`).
- Sample 5–10 random inputs.
- Run the target program; record (inputs, outputs).
- Emit the task. The target program is *not* given to the system.

We additionally hand-curate a hold-out evaluation set per family: a few
dozen tasks that we report on but never train on. The training-vs-held-out
split is the diff between "did we learn" and "did we memorise".

## Configuration

Each task family has a config:

```rust
pub struct TaskFamilyConfig {
    pub name: String,
    pub seed_primitives: Vec<BuiltinId>,    // domain-specific built-ins
    pub max_program_size: u32,               // search budget cap
    pub eval_fuel: u32,
    pub generator: Box<dyn TaskGenerator>,
    pub hold_out: Vec<TaskId>,
}
```

The training loop iterates over all enabled families round-robin. The
abstraction sleep operates *across* families — primitives mined from
list tasks are available to ARC tasks and vice versa, which is the whole
point of building a shared library.

## Why no "agent in environment" tasks

DreamCoder is sometimes extended to RL-flavoured tasks (a program that acts
in an environment). We deliberately exclude that family because (a) it
breaks the "deterministic program → score" abstraction the design relies
on, and (b) the cool bits of RL tasks are orthogonal to what we want to
learn here. We can revisit later.
