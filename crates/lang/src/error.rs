use thiserror::Error;

use crate::ty::UnifyError;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    #[error("type error: {0}")]
    Type(#[from] UnifyError),

    #[error("apply: function type required, got {0}")]
    NotAFunction(String),

    #[error("apply: argument type {arg} does not match parameter type {param}: {source}")]
    ApplyMismatch {
        param: String,
        arg: String,
        #[source] source: UnifyError,
    },

    #[error("evaluation: out of fuel")]
    OutOfFuel,

    #[error("evaluation: bottom (runtime failure: {0})")]
    Bottom(String),

    #[error("eval: type mismatch in primitive {0}")]
    PrimitiveTypeMismatch(&'static str),

    #[error("invalid program: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;
