use crate::error::{DmdError, DmdResult};
use logos::Logos;

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub start: usize,
    pub end: usize,
    pub message: String,
}

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"//[^\n\r]*")]
pub enum Token {
    #[token("params")]
    Params,
    #[token("array")]
    Array,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("step")]
    Step,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("read")]
    Read,
    #[token("write")]
    Write,
    #[token("update")]
    Update,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,
    #[token(";")]
    Semi,
    #[token("..")]
    DotDot,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("&&")]
    AndAnd,
    #[token("<=")]
    LessEq,
    #[token("<")]
    Less,
    #[token(">=")]
    GreaterEq,
    #[token(">")]
    Greater,
    #[token("==")]
    EqEq,
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Integer(i64),
}

pub type SpannedToken = Result<(usize, Token, usize), LexError>;

pub fn lex(source: &str) -> DmdResult<Vec<SpannedToken>> {
    let mut lexer = Token::lexer(source);
    let mut tokens = Vec::new();

    while let Some(token) = lexer.next() {
        let span = lexer.span();
        match token {
            Ok(token) => tokens.push(Ok((span.start, token, span.end))),
            Err(()) => {
                return Err(DmdError::Lex {
                    start: span.start,
                    end: span.end,
                    message: "unrecognized token".to_string(),
                });
            }
        }
    }

    Ok(tokens)
}
