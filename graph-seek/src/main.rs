mod parse_program;

#[derive(Debug)]
pub enum ValueType {
    Int,
    List,
    Matrix,
}

#[derive(Debug)]
pub enum Literal {
    Literal(i32),
    NaN,
}

#[derive(Debug)]
pub enum Expr {
    Literal(Literal),
    Variable(String),
    FunctionCall(String, Vec<Expr>),
}
#[derive(Debug)]
pub enum PatternElement {
    Literal(Literal),
    Variable(String),
}

#[derive(Debug)]
pub struct Branch {
    pattern: Vec<PatternElement>,
    expression: Expr,
}

#[derive(Debug)]
pub struct Definition {
    args: Vec<ValueType>,
    name: String,
    branches: Vec<Branch>,
}

// The result of a program is the last definition, which must be a function of zero arguments
#[derive(Debug)]
pub struct Program {
    definitions: Vec<Definition>,
}

fn main() {
    let example = r#"
        add_1 :: int -> int
        add_1 n = plus n 1
        five :: int
        five = add_1 4
    "#;
    let parsed = parse_program::parse_program(example);
    match parsed {
        Ok(prog) => println!("{:#?}", prog),
        Err(e) => {
            eprintln!("Error parsing program: {}", e);
            return;
        }
    }
}
