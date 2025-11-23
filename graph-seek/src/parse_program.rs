use nom::{IResult, branch::alt, bytes::complete::{tag, take_while1}, character::complete::{char, digit1, multispace0, newline}, combinator::{map, map_res, opt, recognize}, multi::{many0, many1}, sequence::{delimited, preceded, terminated, tuple}};
// Helper to wrap a parser and consume surrounding whitespace
fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
	F: FnMut(&'a str) -> IResult<&'a str, O>,
{
	delimited(multispace0, inner, multispace0)
}
use crate::{Program, Definition, Branch, Expr, PatternElement, ValueType, Literal};

pub fn parse_program(input: &str) -> Result<Program, String> {
	match program(input) {
		Ok((_, prog)) => Ok(prog),
		Err(e) => Err(format!("Parse error: {:?}", e)),
	}
}

fn program(input: &str) -> IResult<&str, Program> {
	let (input, _) = multispace0(input)?;
	let (input, definitions) = many0(terminated(definition, multispace0))(input)?;
	let (input, _) = multispace0(input)?;
	Ok((input, Program { definitions }))
}

fn definition(input: &str) -> IResult<&str, Definition> {
	let (input, _) = opt(newline)(input)?;
	let (input, name) = ws(identifier)(input)?;
	let (input, _) = ws(tag("::"))(input)?;
	let (input, args) = ws(type_sig)(input)?;
	let (input, branches) = many1(branch)(input)?;
	Ok((input, Definition { name, args, branches }))
}

fn branch(input: &str) -> IResult<&str, Branch> {
	let (input, _name) = ws(identifier)(input)?;
	let (input, pattern) = many0(ws(pattern))(input)?;
	let (input, _) = ws(tag("="))(input)?;
	let (input, expression) = ws(expr)(input)?;
	let (input, _) = opt(ws(newline))(input)?;
	Ok((input, Branch { pattern, expression }))
}

fn pattern(input: &str) -> IResult<&str, PatternElement> {
	alt((
		map(ws(literal), PatternElement::Literal),
		map(ws(identifier), PatternElement::Variable),
	))(input)
}

fn expr(input: &str) -> IResult<&str, Expr> {
	infix_expr(input)
}

fn infix_expr(input: &str) -> IResult<&str, Expr> {
	add_expr(input)
}

fn add_expr(input: &str) -> IResult<&str, Expr> {
	let (input, init) = mul_expr(input)?;
	let (input, rest) = many0(tuple((add_op, mul_expr)))(input)?;
	let expr = rest.into_iter().fold(init, |acc, (op, rhs)| {
		Expr::FunctionCall(op, vec![acc, rhs])
	});
	Ok((input, expr))
}

fn mul_expr(input: &str) -> IResult<&str, Expr> {
	let (input, init) = atom(input)?;
	let (input, rest) = many0(tuple((mul_op, atom)))(input)?;
	let expr = rest.into_iter().fold(init, |acc, (op, rhs)| {
		Expr::FunctionCall(op, vec![acc, rhs])
	});
	Ok((input, expr))
}

fn atom(input: &str) -> IResult<&str, Expr> {
	alt((
		map(ws(literal), Expr::Literal),
		map(ws(identifier), Expr::Variable),
		ws(delimited(
			char('('),
			expr,
			char(')')
		)),
	))(input)
}

fn add_op(input: &str) -> IResult<&str, String> {
	ws(alt((
		map(tag("+"), |s: &str| s.to_string()),
		map(tag("-"), |s: &str| s.to_string()),
	)))(input)
}

fn mul_op(input: &str) -> IResult<&str, String> {
	ws(alt((
		map(tag("*"), |s: &str| s.to_string()),
		map(tag("/"), |s: &str| s.to_string()),
	)))(input)
}

fn literal(input: &str) -> IResult<&str, Literal> {
	alt((
		map(ws(int_lit), Literal::Literal),
		map(ws(nan_lit), |_| Literal::NaN),
	))(input)
}

fn int_lit(input: &str) -> IResult<&str, i32> {
	map_res(
		recognize(digit1),
		|s: &str| s.parse::<i32>()
	)(input)
}

fn nan_lit(input: &str) -> IResult<&str, ()> {
	map(tag("NaN"), |_| ())(input)
}

fn identifier(input: &str) -> IResult<&str, String> {
	ws(map(
		recognize(
			take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_')
		),
		|s: &str| s.to_string()
	))(input)
}

fn type_sig(input: &str) -> IResult<&str, Vec<ValueType>> {
	let (input, first) = ws(base_type)(input)?;
	let (input, rest) = many0(preceded(ws(tag("->")), ws(base_type)))(input)?;
	let mut types = vec![first];
	types.extend(rest);
	Ok((input, types))
}

fn base_type(input: &str) -> IResult<&str, ValueType> {
	ws(alt((
		map(tag("int"), |_| ValueType::Int),
		map(tag("[int]"), |_| ValueType::List),
		map(tag("[[int]]"), |_| ValueType::Matrix),
	)))(input)
}


