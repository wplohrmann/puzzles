//! Task families: function approximation, reconstruction, ARC.
//!
//! See `docs/05-tasks.md`. M2 ships `Task` (slim version) and
//! `ListExamplesTask`. The `TaskEncoding` trait used by the neural
//! recogniser is deferred to M4.

use serde::{Deserialize, Serialize};

use lang::arena::{Arena, NodeId};
use lang::eval::{eval, Value};
use lang::library::Library;

/// Stable, deterministic task identifier — useful for replay logs and
/// dedup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskId(pub u64);

/// What the search synthesises.
///
/// The candidate `root` is an expression that uses `Param(0)` to refer
/// to the task input. Evaluation binds `Param(0)` to each example's
/// input and compares the resulting value with the expected output.
///
/// There is no `target_type`: the language has no static type system.
/// A program "matches" purely by producing the expected `Value` on
/// every example input. `Value`'s `PartialEq` is type-strict (`Int(0)`
/// doesn't equal `List([])`), so value-equality alone is sound for
/// solution detection.
///
/// `Send + Sync` are not bound here because `Value` holds `Rc`s and is
/// not `Send`. Parallel-task evaluation requires moving to `Arc` first.
pub trait Task {
    /// True iff the candidate's value matches the expected output on
    /// every example. A candidate that produces `Bottom` on any example
    /// is not-a-solution.
    fn solves(&self, arena: &Arena, lib: &Library, root: NodeId) -> bool {
        self.score(arena, lib, root) >= 1.0
    }

    /// Continuous quality in [0, 1]. For binary tasks: fraction of
    /// examples solved exactly.
    fn score(&self, arena: &Arena, lib: &Library, root: NodeId) -> f32;

    /// Stable id.
    fn id(&self) -> TaskId;
}

/// The function-approximation family: a list of `(input, expected_output)`
/// pairs. The candidate program is an expression whose only free
/// variable is `Param(0)`.
#[derive(Clone, Debug)]
pub struct ListExamplesTask {
    pub id: TaskId,
    pub examples: Vec<(Value, Value)>,
    pub fuel: u32,
}

impl Task for ListExamplesTask {
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
                Err(_) => return 0.0,
            };
            if actual.is_bottom() {
                continue;
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
/// to the search — only used to produce expected outputs.
pub fn programmatic_task(
    id: TaskId,
    arena: &Arena,
    lib: &Library,
    ground_truth: NodeId,
    inputs: Vec<Value>,
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
    ListExamplesTask { id, examples, fuel }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang::builtin::seed_builtin_library;
    use lang::construct::{app, lit, param, prim_ref};
    use lang::ir::LitValue;

    /// `λxs. fold add 0 xs` (sum). Used as ground-truth.
    fn build_sum_ground_truth(arena: &mut Arena, lib: &Library) -> NodeId {
        let p = param(arena, 0);
        let fold = prim_ref(arena, lib, lib.lookup("fold").unwrap());
        let add = prim_ref(arena, lib, lib.lookup("add").unwrap());
        let zero = lit(arena, LitValue::Int(0));
        let f1 = app(arena, fold, add);
        let f2 = app(arena, f1, zero);
        app(arena, f2, p)
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
        let task = programmatic_task(TaskId(1), &arena, &lib, gt, inputs, 1_000_000);
        assert_eq!(task.score(&arena, &lib, gt), 1.0);
        assert!(task.solves(&arena, &lib, gt));
    }

    #[test]
    fn sum_task_rejects_wrong_program() {
        let lib = seed_builtin_library();
        let mut arena = Arena::new();
        let gt = build_sum_ground_truth(&mut arena, &lib);
        let inputs = vec![
            Value::list_from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            Value::list_from(vec![]),
        ];
        let task = programmatic_task(TaskId(2), &arena, &lib, gt, inputs, 1_000_000);
        let zero = lit(&mut arena, LitValue::Int(0));
        assert!(!task.solves(&arena, &lib, zero));
        assert!((task.score(&arena, &lib, zero) - 0.5).abs() < 1e-6);
    }
}
