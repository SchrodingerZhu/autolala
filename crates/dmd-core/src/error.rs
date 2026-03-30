use crate::lexer::{LexError, Token};
use std::fmt::Write;

#[derive(Debug, thiserror::Error)]
pub enum DmdError {
    #[error("lex error at {start}..{end}: {message}")]
    Lex {
        start: usize,
        end: usize,
        message: String,
    },
    #[error("parse error at {location}: {message}")]
    Parse { location: usize, message: String },
    #[error("semantic error: {message}")]
    Semantic { message: String },
    #[error("analysis error: {message}")]
    Analysis { message: String },
    #[error(transparent)]
    Barvinok(#[from] barvinok::Error),
}

pub type DmdResult<T> = Result<T, DmdError>;

impl DmdError {
    pub(crate) fn from_lex_error(error: LexError) -> Self {
        Self::Lex {
            start: error.start,
            end: error.end,
            message: error.message,
        }
    }

    pub(crate) fn from_parse_error(
        error: lalrpop_util::ParseError<usize, Token, LexError>,
    ) -> Self {
        match error {
            lalrpop_util::ParseError::InvalidToken { location } => Self::Parse {
                location,
                message: "invalid token".to_string(),
            },
            lalrpop_util::ParseError::UnrecognizedEof { location, expected } => Self::Parse {
                location,
                message: render_expected("unexpected end of input", &expected),
            },
            lalrpop_util::ParseError::UnrecognizedToken { token, expected } => Self::Parse {
                location: token.0,
                message: render_expected(
                    &format!("unexpected token {}", token_name(&token.1)),
                    &expected,
                ),
            },
            lalrpop_util::ParseError::ExtraToken { token } => Self::Parse {
                location: token.0,
                message: format!("extra token {}", token_name(&token.1)),
            },
            lalrpop_util::ParseError::User { error } => Self::from_lex_error(error),
        }
    }

    pub(crate) fn semantic(message: impl Into<String>) -> Self {
        Self::Semantic {
            message: message.into(),
        }
    }

    pub(crate) fn analysis(message: impl Into<String>) -> Self {
        Self::Analysis {
            message: message.into(),
        }
    }
}

fn render_expected(prefix: &str, expected: &[String]) -> String {
    if expected.is_empty() {
        return prefix.to_string();
    }

    let mut message = String::from(prefix);
    let _ = write!(message, "; expected {}", expected.join(", "));
    message
}

fn token_name(token: &Token) -> String {
    match token {
        Token::Params => "`params`".to_string(),
        Token::Array => "`array`".to_string(),
        Token::For => "`for`".to_string(),
        Token::In => "`in`".to_string(),
        Token::Step => "`step`".to_string(),
        Token::If => "`if`".to_string(),
        Token::Else => "`else`".to_string(),
        Token::Read => "`read`".to_string(),
        Token::Write => "`write`".to_string(),
        Token::Update => "`update`".to_string(),
        Token::LBrace => "`{`".to_string(),
        Token::RBrace => "`}`".to_string(),
        Token::LBracket => "`[`".to_string(),
        Token::RBracket => "`]`".to_string(),
        Token::LParen => "`(`".to_string(),
        Token::RParen => "`)`".to_string(),
        Token::Comma => "`,`".to_string(),
        Token::Semi => "`;`".to_string(),
        Token::DotDot => "`..`".to_string(),
        Token::Plus => "`+`".to_string(),
        Token::Minus => "`-`".to_string(),
        Token::Star => "`*`".to_string(),
        Token::Slash => "`/`".to_string(),
        Token::AndAnd => "`&&`".to_string(),
        Token::Less => "`<`".to_string(),
        Token::LessEq => "`<=`".to_string(),
        Token::Greater => "`>`".to_string(),
        Token::GreaterEq => "`>=`".to_string(),
        Token::EqEq => "`==`".to_string(),
        Token::Identifier(name) => format!("identifier `{name}`"),
        Token::Integer(value) => format!("integer `{value}`"),
    }
}
