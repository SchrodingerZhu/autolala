use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub params: Vec<String>,
    pub arrays: Vec<ArrayDecl>,
    pub body: Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayDecl {
    pub name: String,
    pub extents: Vec<Expr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub statements: Vec<Stmt>,
}

impl Block {
    pub fn new(statements: Vec<Stmt>) -> Self {
        Self { statements }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    For(ForLoop),
    If(IfStmt),
    Access(Access),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForLoop {
    pub var: String,
    pub lower: Expr,
    pub upper: Expr,
    pub step: i64,
    pub body: Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfStmt {
    pub conditions: Vec<Comparison>,
    pub then_branch: Block,
    pub else_branch: Option<Block>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AccessKind {
    Read,
    Write,
    Update,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Access {
    pub kind: AccessKind,
    pub array: String,
    pub indices: Vec<Expr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub lhs: Expr,
    pub op: ComparisonOp,
    pub rhs: Expr,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Int(i64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    FloorDiv(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
}

impl Expr {
    pub fn as_const_i64(&self) -> Option<i64> {
        match self {
            Expr::Int(value) => Some(*value),
            Expr::Neg(expr) => expr.as_const_i64().map(|value| -value),
            Expr::Add(lhs, rhs) => Some(lhs.as_const_i64()? + rhs.as_const_i64()?),
            Expr::Sub(lhs, rhs) => Some(lhs.as_const_i64()? - rhs.as_const_i64()?),
            Expr::Mul(lhs, rhs) => Some(lhs.as_const_i64()? * rhs.as_const_i64()?),
            Expr::FloorDiv(lhs, rhs) => {
                let lhs = lhs.as_const_i64()?;
                let rhs = rhs.as_const_i64()?;
                if rhs == 0 {
                    None
                } else {
                    Some(lhs.div_euclid(rhs))
                }
            }
            Expr::Var(_) => None,
        }
    }
}
