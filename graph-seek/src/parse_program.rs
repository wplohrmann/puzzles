use pest::Parser;
use pest::iterators::Pair;

use crate::{Program, Definition, Branch, Expr, PatternElement, ValueType, Literal};

use super::parser::Rule;

pub fn parse_program(input: &str) -> Result<Program, String> {
    let pairs = LangParser::parse(Rule::program, input).map_err(|e| e.to_string())?;
    let mut definitions = Vec::new();
    for pair in pairs {
        match pair.as_rule() {
            Rule::program => {
                for def in pair.into_inner() {
                    if let Some(defn) = parse_definition(def)? {
                        definitions.push(defn);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Program { definitions })
}

fn parse_definition(pair: Pair<Rule>) -> Result<Option<Definition>, String> {
    match pair.as_rule() {
        Rule::definition => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let type_sig = inner.next().unwrap();
            let args = parse_type_sig(type_sig)?;
            let mut branches = Vec::new();
            for branch in inner {
                branches.push(parse_branch(branch)?);
            }
            Ok(Some(Definition { name, args, branches }))
        }
        Rule::EOI => Ok(None),
        _ => Ok(None),
    }
}

fn parse_type_sig(pair: Pair<Rule>) -> Result<Vec<ValueType>, String> {
    let mut types = Vec::new();
    for t in pair.into_inner() {
        match t.as_rule() {
            Rule::base_type => {
                match t.as_str() {
                    "int" => types.push(ValueType::Int),
                    "[int]" => types.push(ValueType::List),
                    "[[int]]" => types.push(ValueType::Matrix),
                    _ => return Err(format!("Unknown base type: {}", t.as_str())),
                }
            }
            _ => {}
        }
    }
    Ok(types)
}

fn parse_branch(pair: Pair<Rule>) -> Result<Branch, String> {
    let mut inner = pair.into_inner();
    let _name = inner.next().unwrap(); // function name, can be ignored
    let mut pattern = Vec::new();
    // Patterns until '='
    loop {
        let next = inner.peek();
        if let Some(p) = next {
            if p.as_rule() == Rule::expr { break; }
            let pat = inner.next().unwrap();
            pattern.push(parse_pattern(pat)?);
        } else {
            break;
        }
    }
    let expr = parse_expr(inner.next().unwrap())?;
    Ok(Branch { pattern, expression: expr })
}

fn parse_pattern(pair: Pair<Rule>) -> Result<PatternElement, String> {
    match pair.as_rule() {
        Rule::literal => {
            let lit = parse_literal(pair)?;
            Ok(PatternElement::Literal(lit))
        }
        Rule::identifier => Ok(PatternElement::Variable(pair.as_str().to_string())),
        _ => Err(format!("Unknown pattern: {:?}", pair)),
    }
}

fn parse_expr(pair: Pair<Rule>) -> Result<Expr, String> {
    match pair.as_rule() {
        Rule::expr | Rule::infix_expr | Rule::add_expr | Rule::mul_expr => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            let mut left = parse_expr(first)?;
            while let Some(op_pair) = inner.next() {
                let op = op_pair.as_str();
                let right = parse_expr(inner.next().unwrap())?;
                left = Expr::FunctionCall(op.to_string(), vec![left, right]);
            }
            Ok(left)
        }
        Rule::atom => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            match first.as_rule() {
                Rule::literal => Ok(Expr::Literal(parse_literal(first)?)),
                Rule::identifier => Ok(Expr::Variable(first.as_str().to_string())),
                Rule::expr => parse_expr(first),
                _ => Err(format!("Unknown atom: {:?}", first)),
            }
        }
        Rule::literal => Ok(Expr::Literal(parse_literal(pair)?)),
        Rule::identifier => Ok(Expr::Variable(pair.as_str().to_string())),
        _ => Err(format!("Unknown expr: {:?}", pair)),
    }
}

fn parse_literal(pair: Pair<Rule>) -> Result<Literal, String> {
    let inner = pair.clone().into_inner().next();
    match inner {
        Some(p) => match p.as_rule() {
            Rule::int_lit => Ok(Literal::Literal(p.as_str().parse().unwrap())),
            Rule::nan_lit => Ok(Literal::NaN),
            _ => Err(format!("Unknown literal: {:?}", p)),
        },
        None => match pair.as_rule() {
            Rule::int_lit => Ok(Literal::Literal(pair.as_str().parse().unwrap())),
            Rule::nan_lit => Ok(Literal::NaN),
            _ => Err(format!("Unknown literal: {:?}", pair)),
        },
    }
}

use super::parser::LangParser;
