use thiserror::Error;

/// Evaluator-level errors. Static type errors no longer exist (the
/// language has no static type system); runtime type mismatches return
/// `Value::Bottom` rather than `Err`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    #[error("evaluation: out of fuel")]
    OutOfFuel,

    #[error("invalid program: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;
