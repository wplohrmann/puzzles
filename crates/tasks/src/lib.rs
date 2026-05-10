//! Task families: function approximation, reconstruction, ARC.
//!
//! See `docs/05-tasks.md`. M2 ships `Task` (slim version) and
//! `ListExamplesTask` (the function-approximation family). The
//! `TaskEncoding` trait used by the neural recogniser is deferred to M4.

use serde::{Deserialize, Serialize};

use lang::arena::{Arena, NodeId};
use lang::eval::{eval, Value};
use lang::library::Library;
use lang::ty::{Ty, TypeScheme};

/// Stable, deterministic task identifier — useful for replay logs and
/// dedup. Caller-provided; we don't generate it from contents because the
/// same task may be re-emitted with new fuel or example-input shuffles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskId(pub u64);

/// What the search synthesises.
///
/// The candidate `root` is an *expression* (not a function): it uses
/// `Param(0)` of the task's argument type to refer to the input. Evaluation
/// binds `Param(0)` to each example's input and compares the resulting
/// value with the expected output.
///
/// M2 omits `encoding`; the neural recogniser lands in M4. We also drop
/// the `Send + Sync` bounds the design doc proposes — `Value` holds `Rc`s
/// and is `!Send`, so parallel-task evaluation requires switching to
/// `Arc` first. See `docs/decisions/m2-search-tasks.md`.
pub trait Task {
    /// The function type the search is solving for, e.g. `List<Int> → Int`.
    /// `target_arg_ty` and `target_ret_ty` derive from this.
    fn target_type(&self) -> &TypeScheme;

    /// True iff the candidate's value matches the expected output on every
    /// example. A candidate that produces `Bottom` on any example is
    /// not-a-solution.
    fn solves(&self, arena: &Arena, lib: &Library, root: NodeId) -> bool {
        self.score(arena, lib, root) >= 1.0
    }

    /// Continuous quality in [0, 1]. For binary tasks: fraction of examples
    /// solved exactly.
    fn score(&self, arena: &Arena, lib: &Library, root: NodeId) -> f32;

    /// Stable id.
    fn id(&self) -> TaskId;
}

/// The function-approximation family: a list of `(input, expected_output)`
/// pairs. The candidate program is an expression whose only free variable
/// is `Param(0)` of `target_arg_ty`.
#[derive(Clone, Debug)]
pub struct ListExamplesTask {
    pub id: TaskId,
    /// The function type. Currently expected to be `arg → ret` (single-arg).
    pub target_type: TypeScheme,
    pub examples: Vec<(Value, Value)>,
    pub fuel: u32,
}

impl ListExamplesTask {
    /// Convenience: extract `(arg_ty, ret_ty)` from a single-arg target type.
    /// Panics if `target_type` is not a function.
    pub fn arg_ret(&self) -> (Ty, Ty) {
        let body = &self.target_type.body;
        let (a, r) = body.as_func().expect("ListExamplesTask: target type must be a function");
        (a.clone(), r.clone())
    }
}

impl Task for ListExamplesTask {
    fn target_type(&self) -> &TypeScheme {
        &self.target_type
    }

    fn score(&self, arena: &Arena, lib: &Library, root: NodeId) -> f32 {
        if self.examples.is_empty() {
            return 0.0;
        }
        let mut hits = 0;
        for (input, expected) in &self.examples {
            let mut fuel = self.fuel;
            let env = [input.clone()];
            let actual = match eval(arena, lib, root, &env, &mut fuel) {
                Ok(v) => v,
                Err(_) => return 0.0, // any eval error = not a solution
            };
            if actual.is_bottom() {
                continue; // miss this example
            }
            if &actual == expected {
                hits += 1;
            }
        }
        hits as f32 / self.examples.len() as f32
    }

    fn id(&self) -> TaskId {
        self.id
    }
}

/// Build a `ListExamplesTask` by running a hand-written ground-truth
/// program against a list of inputs. The ground-truth is *not* exposed
/// to the search — it's only used to produce expected outputs.
///
/// `ground_truth` must be a value-typed expression that uses `Param(0)`
/// of `arg_ty` to read the input.
pub fn programmatic_task(
    id: TaskId,
    arena: &Arena,
    lib: &Library,
    ground_truth: NodeId,
    inputs: Vec<Value>,
    arg_ty: Ty,
    ret_ty: Ty,
    fuel: u32,
) -> ListExamplesTask {
    let mut examples = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut f = fuel;
        let env = [input.clone()];
        let output = eval(arena, lib, ground_truth, &env, &mut f)
            .expect("ground-truth program evaluates without error");
        assert!(
            !output.is_bottom(),
            "ground-truth produced Bottom on input — bad task",
        );
        examples.push((input, output));
    }
    ListExamplesTask {
        id,
        target_type: TypeScheme::mono(Ty::func(arg_ty, ret_ty)),
        examples,
        fuel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang::builtin::seed_builtin_library;
    use lang::construct::{app, lit, param, prim_ref};
    use lang::ir::LitValue;
    use lang::ty::TyVarGen;

    /// `λxs:List<Int>. fold add 0 xs` (sum). Used as ground-truth.
    fn build_sum_ground_truth(arena: &mut Arena, lib: &Library) -> NodeId {
        let mut gen = TyVarGen::new();
        let p = param(arena, 0, Ty::list(Ty::int()));
        let fold = prim_ref(arena, lib, lib.lookup("fold").unwrap());
        let add = prim_ref(arena, lib, lib.lookup("add").unwrap());
        let zero = lit(arena, LitValue::Int(0));
        let f1 = app(arena, &mut gen, fold, add).unwrap();
        let f2 = app(arena, &mut gen, f1, zero).unwrap();
        app(arena, &mut gen, f2, p).unwrap()
    }

    #[test]
    fn sum_task_scores_correct_program_one() {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let gt = build_sum_ground_truth(&mut arena, &lib);
        let inputs = vec![
            Value::list_from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            Value::list_from(vec![]),
            Value::list_from(vec![Value::Int(7)]),
        ];
        let task = programmatic_task(
            TaskId(1), &arena, &lib, gt, inputs,
            Ty::list(Ty::int()), Ty::int(), 1_000_000,
        );
        // The ground-truth itself trivially solves the task.
        assert_eq!(task.score(&arena, &lib, gt), 1.0);
        assert!(task.solves(&arena, &lib, gt));
    }

    #[test]
    fn sum_task_rejects_wrong_program() {
        // λxs. 0 — always returns 0. Solves the empty-list example only.
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let gt = build_sum_ground_truth(&mut arena, &lib);
        let inputs = vec![
            Value::list_from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            Value::list_from(vec![]),
        ];
        let task = programmatic_task(
            TaskId(2), &arena, &lib, gt, inputs,
            Ty::list(Ty::int()), Ty::int(), 1_000_000,
        );
        let zero = lit(&mut arena, LitValue::Int(0));
        assert!(!task.solves(&arena, &lib, zero));
        assert!((task.score(&arena, &lib, zero) - 0.5).abs() < 1e-6);
    }
}
