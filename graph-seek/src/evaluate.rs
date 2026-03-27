use crate::{Program, Definition, Branch, Expr, PatternElement, ValueType, Literal};

#[derive(Debug, Clone)]
pub enum EvaluatedValue {
	Int(i32),
	List(Vec<EvaluatedValue>),
	Matrix(Vec<Vec<EvaluatedValue>>),
	NaN,
	// Add more as needed
}

/// Evaluates a program and returns a vector of evaluated values for each definition.
pub fn evaluate_program(program: &Program) -> Vec<(String, Vec<EvaluatedValue>)> {
	let mut results = Vec::new();
	for def in &program.definitions {
		// TODO: Evaluate each definition according to its branches and expressions
		// For now, just push a placeholder
		results.push((def.name.clone(), vec![]));
	}
	results
}
