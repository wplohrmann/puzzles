mod parse_program;

#[derive(Debug, Clone)]
pub enum PrimitiveType {
    Int,
    Bool,
}

#[derive(Debug)]
pub enum Type {
    Primitive(PrimitiveType),
    Product(Vec<Box<Type>>),
    Sum(Vec<Box<Type>>),
    Refined((Box<Type>, Box<Expr>)),
    Func {
        args: Vec<Box<Type>>,
        ret: Box<Type>,
    }
}

#[derive(Debug)]
pub enum Literal {
    Int(i32),
    Bool(bool),
    Error,
}

#[derive(Debug)]
pub enum Value {
    Literal(Literal),
    Variable(String),
    FunctionCall(String, Vec<Expr>),
}

#[derive(Debug)]
pub struct Expr {
    e_type: Type,
    value: Value
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
