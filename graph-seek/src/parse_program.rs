use crate::{Program, Definition, Branch, Expr, PatternElement, ValueType, Literal};

pub fn parse_program(input: &str) -> Result<Program, String> {
    let mut definitions: Vec<Definition> = vec![];
    let mut current_definition: Option<Definition> = None;
    let mut current_args: Vec<ValueType> = vec![];
    let mut current_name: Option<String> = None;
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Type signature line: name :: ...
        if let Some(idx) = line.find("::") {
            let (name_part, type_part) = line.split_at(idx);
            let name = name_part.trim().to_string();
            let args: Vec<ValueType> = type_part[2..]
                .split("->")
                .map(|s| match s.trim() {
                    "int" => Some(ValueType::Int),
                    "list" => Some(ValueType::List),
                    "matrix" => Some(ValueType::Matrix),
                    _ => None,
                })
                .filter_map(|x| x)
                .collect();
            // If there was a previous definition, push it
            if let Some(def) = current_definition.take() {
                definitions.push(def);
            }
            current_name = Some(name.clone());
            current_args = args;
            current_definition = Some(Definition {
                name,
                args: current_args.clone(),
                branches: vec![],
            });
            continue;
        }
        // Branch line: name [args] = expr
        if let Some(eq_idx) = line.find('=') {
            let (lhs, rhs) = line.split_at(eq_idx);
            let lhs = lhs.trim();
            let rhs = rhs[1..].trim(); // skip '='
            let mut lhs_words = lhs.split_whitespace();
            let branch_name = lhs_words.next().unwrap(); // safe: always at least one
            let pattern: Vec<PatternElement> = lhs_words.map(|w| {
                if let Ok(num) = w.parse::<i32>() {
                    PatternElement::Literal(Literal::Literal(num))
                } else {
                    PatternElement::Variable(w.to_string())
                }
            }).collect();
            // Parse expression (very simple: function call or literal)
            let expr = if let Some((fname, rest)) = rhs.split_once(' ') {
                // function call
                let args: Vec<Expr> = rest
                    .split_whitespace()
                    .map(|w| {
                        if let Ok(num) = w.parse::<i32>() {
                            Expr::Literal(Literal::Literal(num))
                        } else {
                            Expr::Variable(w.to_string())
                        }
                    })
                    .collect();
                Expr::FunctionCall(fname.to_string(), args)
            } else if let Ok(num) = rhs.parse::<i32>() {
                Expr::Literal(Literal::Literal(num))
            } else {
                Expr::Variable(rhs.to_string())
            };
            if let Some(def) = &mut current_definition {
                def.branches.push(Branch {
                    pattern,
                    expression: expr,
                });
            } else {
                // No current definition, so create one (for definitions without type signature)
                let mut def = Definition {
                    name: branch_name.to_string(),
                    args: pattern.iter().filter_map(|p| match p { PatternElement::Variable(v) => Some(ValueType::Int), _ => None }).collect(),
                    branches: vec![],
                };
                def.branches.push(Branch {
                    pattern,
                    expression: expr,
                });
                definitions.push(def);
            }
            continue;
        }
    }
    // Push the last definition if any
    if let Some(def) = current_definition.take() {
        definitions.push(def);
    }
    Ok(Program { definitions })
}

