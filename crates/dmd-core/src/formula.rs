use barvinok::{
    DimType,
    aff::Affine,
    polynomial::{QuasiPolynomial, Term},
    set::Set,
    space::Space,
    value::Value,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulaExpr {
    Rational { numerator: i64, denominator: i64 },
    Symbol(String),
    Add(Vec<FormulaExpr>),
    Mul(Vec<FormulaExpr>),
    Div(Box<FormulaExpr>, Box<FormulaExpr>),
    Pow(Box<FormulaExpr>, u32),
    #[allow(dead_code)]
    Sqrt(Box<FormulaExpr>),
}

impl FormulaExpr {
    pub fn zero() -> Self {
        Self::Rational {
            numerator: 0,
            denominator: 1,
        }
    }

    pub fn one() -> Self {
        Self::Rational {
            numerator: 1,
            denominator: 1,
        }
    }

    pub fn rational(numerator: i64, denominator: i64) -> Self {
        debug_assert!(denominator != 0, "denominator must not be zero");
        if numerator == 0 {
            return Self::zero();
        }

        let sign = if denominator < 0 { -1 } else { 1 };
        let numerator = numerator * sign;
        let denominator = denominator.abs();
        let gcd = gcd_i64(numerator.abs(), denominator);
        Self::Rational {
            numerator: numerator / gcd,
            denominator: denominator / gcd,
        }
    }

    pub fn symbol(name: impl Into<String>) -> Self {
        Self::Symbol(name.into())
    }

    pub fn add(terms: impl IntoIterator<Item = FormulaExpr>) -> Self {
        let mut flattened = Vec::new();
        for term in terms {
            match term {
                FormulaExpr::Add(inner) => flattened.extend(inner),
                value if value.is_zero() => {}
                value => flattened.push(value),
            }
        }

        match flattened.len() {
            0 => Self::zero(),
            1 => flattened.into_iter().next().unwrap_or_else(Self::zero),
            _ => Self::Add(flattened),
        }
    }

    pub fn mul(factors: impl IntoIterator<Item = FormulaExpr>) -> Self {
        let mut flattened = Vec::new();
        for factor in factors {
            match factor {
                value if value.is_zero() => return Self::zero(),
                FormulaExpr::Mul(inner) => flattened.extend(inner),
                value if value.is_one() => {}
                value => flattened.push(value),
            }
        }

        match flattened.len() {
            0 => Self::one(),
            1 => flattened.into_iter().next().unwrap_or_else(Self::one),
            _ => Self::Mul(flattened),
        }
    }

    pub fn div(numerator: FormulaExpr, denominator: FormulaExpr) -> Self {
        if denominator.is_one() {
            return numerator;
        }
        Self::Div(Box::new(numerator), Box::new(denominator))
    }

    pub fn pow(base: FormulaExpr, exponent: u32) -> Self {
        match exponent {
            0 => Self::one(),
            1 => base,
            _ => Self::Pow(Box::new(base), exponent),
        }
    }

    #[allow(dead_code)]
    pub fn sqrt(expr: FormulaExpr) -> Self {
        Self::Sqrt(Box::new(expr))
    }

    pub fn is_zero(&self) -> bool {
        matches!(
            self,
            FormulaExpr::Rational {
                numerator: 0,
                denominator: 1
            }
        )
    }

    pub fn is_one(&self) -> bool {
        matches!(
            self,
            FormulaExpr::Rational {
                numerator: 1,
                denominator: 1
            }
        )
    }

    pub fn to_plain(&self) -> String {
        self.render(RenderStyle::Plain, 0)
    }

    pub fn to_latex(&self) -> String {
        self.render(RenderStyle::Latex, 0)
    }

    fn render(&self, style: RenderStyle, parent_precedence: u8) -> String {
        let (precedence, rendered) = match self {
            FormulaExpr::Rational {
                numerator,
                denominator,
            } => {
                if *denominator == 1 {
                    (4, numerator.to_string())
                } else {
                    let rendered = match style {
                        RenderStyle::Plain => format!("({numerator}/{denominator})"),
                        RenderStyle::Latex => {
                            format!("\\frac{{{numerator}}}{{{denominator}}}")
                        }
                    };
                    (4, rendered)
                }
            }
            FormulaExpr::Symbol(name) => (4, name.clone()),
            FormulaExpr::Add(terms) => {
                let joiner = match style {
                    RenderStyle::Plain => " + ",
                    RenderStyle::Latex => " + ",
                };
                let rendered = terms
                    .iter()
                    .map(|term| term.render(style, 1))
                    .collect::<Vec<_>>()
                    .join(joiner);
                (1, rendered)
            }
            FormulaExpr::Mul(factors) => {
                let joiner = match style {
                    RenderStyle::Plain => " * ",
                    RenderStyle::Latex => " \\cdot ",
                };
                let rendered = factors
                    .iter()
                    .map(|factor| factor.render(style, 2))
                    .collect::<Vec<_>>()
                    .join(joiner);
                (2, rendered)
            }
            FormulaExpr::Div(numerator, denominator) => {
                let rendered = match style {
                    RenderStyle::Plain => {
                        format!(
                            "{}/{}",
                            numerator.render(style, 2),
                            denominator.render(style, 2)
                        )
                    }
                    RenderStyle::Latex => format!(
                        "\\frac{{{}}}{{{}}}",
                        numerator.render(style, 0),
                        denominator.render(style, 0)
                    ),
                };
                (2, rendered)
            }
            FormulaExpr::Pow(base, exponent) => {
                let rendered = match style {
                    RenderStyle::Plain => format!("{}^{}", base.render(style, 3), exponent),
                    RenderStyle::Latex => format!("{}^{{{}}}", base.render(style, 3), exponent),
                };
                (3, rendered)
            }
            FormulaExpr::Sqrt(expr) => {
                let rendered = match style {
                    RenderStyle::Plain => format!("sqrt({})", expr.render(style, 0)),
                    RenderStyle::Latex => format!("\\sqrt{{{}}}", expr.render(style, 0)),
                };
                (3, rendered)
            }
        };

        if precedence < parent_precedence {
            format!("({rendered})")
        } else {
            rendered
        }
    }
}

#[derive(Debug, Clone)]
pub struct FormulaFormatter<'a> {
    space: Space<'a>,
}

impl<'a> FormulaFormatter<'a> {
    pub fn new(space: Space<'a>) -> Self {
        Self { space }
    }

    pub fn quasi_polynomial(&self, qpoly: QuasiPolynomial<'a>) -> Result<FormulaExpr, barvinok::Error> {
        let mut terms = Vec::new();
        qpoly.foreach_term(|term| {
            terms.push(self.term(term)?);
            Ok(())
        })?;
        Ok(FormulaExpr::add(terms))
    }

    pub fn affine(&self, aff: Affine<'a>) -> Result<FormulaExpr, barvinok::Error> {
        let denominator = self.value(aff.get_denominator_val()?);
        let mut numerator_terms = Vec::new();

        let constant = self.value(aff.get_constant_val()?);
        if !constant.is_zero() {
            numerator_terms.push(constant);
        }

        for ty in [DimType::Param, DimType::In] {
            let dims = aff.dim(ty)?;
            for index in 0..dims {
                let coefficient = self.value(aff.get_coefficient_val(ty, index as i32)?);
                if coefficient.is_zero() {
                    continue;
                }

                let symbol = FormulaExpr::symbol(
                    self.space
                        .get_dim_name(ty, index)?
                        .unwrap_or("unnamed"),
                );
                numerator_terms.push(FormulaExpr::mul([coefficient, symbol]));
            }
        }

        let numerator = FormulaExpr::add(numerator_terms);
        if denominator.is_one() {
            Ok(numerator)
        } else {
            Ok(FormulaExpr::div(numerator, denominator))
        }
    }

    pub fn value(&self, value: Value<'a>) -> FormulaExpr {
        FormulaExpr::rational(value.numerator(), value.denominator())
    }

    fn term(&self, term: Term<'a>) -> Result<FormulaExpr, barvinok::Error> {
        let mut factors = vec![self.value(term.coefficient()?)];
        for ty in [DimType::Param, DimType::In] {
            let dims = term.dim(ty)?;
            for index in 0..dims {
                let exponent = term.exponent(ty, index)?;
                if exponent == 0 {
                    continue;
                }

                let factor = FormulaExpr::symbol(
                    self.space
                        .get_dim_name(ty, index)?
                        .unwrap_or("unnamed"),
                );
                factors.push(FormulaExpr::pow(factor, exponent));
            }
        }

        let div_dims = term.dim(DimType::Div)?;
        for index in 0..div_dims {
            let exponent = term.exponent(DimType::Div, index)?;
            if exponent == 0 {
                continue;
            }

            let factor = self.affine(term.get_div(index)?)?;
            factors.push(FormulaExpr::pow(factor, exponent));
        }

        Ok(FormulaExpr::mul(factors))
    }
}

pub fn format_domain(domain: &Set<'_>) -> String {
    let raw = format!("{domain:?}");
    raw.split("{  : ")
        .nth(1)
        .unwrap_or(raw.as_str())
        .split(" }")
        .next()
        .unwrap_or(raw.as_str())
        .trim()
        .to_string()
}

#[derive(Clone, Copy)]
enum RenderStyle {
    Plain,
    Latex,
}

fn gcd_i64(lhs: i64, rhs: i64) -> i64 {
    let mut lhs = lhs.abs();
    let mut rhs = rhs.abs();
    while rhs != 0 {
        let next = lhs % rhs;
        lhs = rhs;
        rhs = next;
    }
    lhs.max(1)
}

#[cfg(test)]
mod tests {
    use super::FormulaExpr;

    #[test]
    fn renders_plain_and_latex() {
        let expr = FormulaExpr::mul([
            FormulaExpr::rational(3, 2),
            FormulaExpr::sqrt(FormulaExpr::add([
                FormulaExpr::symbol("N"),
                FormulaExpr::symbol("M"),
            ])),
        ]);
        assert_eq!(expr.to_plain(), "(3/2) * sqrt(N + M)");
        assert_eq!(expr.to_latex(), "\\frac{3}{2} \\cdot \\sqrt{N + M}");
    }
}
