use barvinok::{
    DimType,
    aff::Affine,
    polynomial::{QuasiPolynomial, Term},
    set::Set,
    space::Space,
    value::Value,
};
use std::cmp::Reverse;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FormulaExpr {
    Rational { numerator: i64, denominator: i64 },
    Symbol(String),
    Raw { plain: String, latex: String },
    Add(Vec<FormulaExpr>),
    Mul(Vec<FormulaExpr>),
    Div(Box<FormulaExpr>, Box<FormulaExpr>),
    Pow(Box<FormulaExpr>, u32),
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

    pub fn raw(plain: impl Into<String>, latex: impl Into<String>) -> Self {
        Self::Raw {
            plain: plain.into(),
            latex: latex.into(),
        }
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

        let mut constant = RationalValue::zero();
        let mut term_order = Vec::new();
        let mut grouped = HashMap::<FormulaExpr, RationalValue>::new();

        for term in flattened {
            let (coefficient, basis) = split_coefficient(term);
            if coefficient.is_zero() {
                continue;
            }

            if let Some(basis) = basis {
                if !grouped.contains_key(&basis) {
                    term_order.push(basis.clone());
                }
                let next = grouped
                    .get(&basis)
                    .copied()
                    .unwrap_or_else(RationalValue::zero)
                    .add(coefficient);
                grouped.insert(basis, next);
            } else {
                constant = constant.add(coefficient);
            }
        }

        let mut simplified = term_order
            .into_iter()
            .filter_map(|basis| {
                let coefficient = grouped.remove(&basis)?;
                if coefficient.is_zero() {
                    None
                } else {
                    Some(build_term(coefficient, basis))
                }
            })
            .collect::<Vec<_>>();

        if !constant.is_zero() {
            simplified.push(constant.into_expr());
        }

        simplified.sort_by_key(|term| Reverse(term.degree()));

        match simplified.len() {
            0 => Self::zero(),
            1 => simplified.into_iter().next().unwrap_or_else(Self::zero),
            _ => Self::Add(simplified),
        }
    }

    pub fn mul(factors: impl IntoIterator<Item = FormulaExpr>) -> Self {
        let mut coefficient = RationalValue::one();
        let mut flattened = Vec::new();
        for factor in factors {
            match factor {
                value if value.is_zero() => return Self::zero(),
                FormulaExpr::Mul(inner) => {
                    for nested in inner {
                        match nested {
                            value if value.is_zero() => return Self::zero(),
                            FormulaExpr::Rational {
                                numerator,
                                denominator,
                            } => {
                                coefficient =
                                    coefficient.mul(RationalValue::new(numerator, denominator));
                            }
                            value if value.is_one() => {}
                            value => flattened.push(value),
                        }
                    }
                }
                FormulaExpr::Rational {
                    numerator,
                    denominator,
                } => {
                    coefficient = coefficient.mul(RationalValue::new(numerator, denominator));
                }
                value if value.is_one() => {}
                value => flattened.push(value),
            }
        }

        if coefficient.is_zero() {
            return Self::zero();
        }

        if let Some(add_index) = flattened
            .iter()
            .position(|factor| matches!(factor, FormulaExpr::Add(_)))
        {
            let FormulaExpr::Add(terms) = flattened.remove(add_index) else {
                unreachable!("matched additive factor");
            };
            let mut shared_factors = flattened;
            if !coefficient.is_one() {
                shared_factors.insert(0, coefficient.into_expr());
            }
            return Self::add(terms.into_iter().map(|term| {
                let mut factors = shared_factors.clone();
                factors.push(term);
                Self::mul(factors)
            }));
        }

        flattened.sort_by_key(|factor| Reverse(factor.sort_key()));

        if !coefficient.is_one() || flattened.is_empty() {
            flattened.insert(0, coefficient.into_expr());
        }

        match flattened.len() {
            0 => coefficient.into_expr(),
            1 => flattened.into_iter().next().unwrap_or_else(Self::one),
            _ => Self::Mul(flattened),
        }
    }

    pub fn div(numerator: FormulaExpr, denominator: FormulaExpr) -> Self {
        if numerator.is_zero() {
            return Self::zero();
        }
        if denominator.is_one() {
            return numerator;
        }
        if let Some(rational) = denominator.as_rational() {
            return Self::mul([numerator, rational.reciprocal().into_expr()]);
        }
        Self::Div(Box::new(numerator), Box::new(denominator))
    }

    pub fn pow(base: FormulaExpr, exponent: u32) -> Self {
        match exponent {
            0 => Self::one(),
            1 => base,
            _ if base.is_zero() => Self::zero(),
            _ if base.is_one() => Self::one(),
            _ => Self::Pow(Box::new(base), exponent),
        }
    }

    pub fn sqrt(expr: FormulaExpr) -> Self {
        if let Some(rational) = expr.as_rational()
            && let Some(square_root) = rational.sqrt()
        {
            return square_root.into_expr();
        }
        Self::Sqrt(Box::new(expr))
    }

    pub fn sub(lhs: FormulaExpr, rhs: FormulaExpr) -> Self {
        match rhs {
            FormulaExpr::Add(terms) => Self::add(
                std::iter::once(lhs).chain(
                    terms
                        .into_iter()
                        .map(|term| Self::mul([Self::rational(-1, 1), term])),
                ),
            ),
            value => Self::add([lhs, Self::mul([Self::rational(-1, 1), value])]),
        }
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

    fn as_rational(&self) -> Option<RationalValue> {
        match self {
            FormulaExpr::Rational {
                numerator,
                denominator,
            } => Some(RationalValue::new(*numerator, *denominator)),
            _ => None,
        }
    }

    fn degree(&self) -> u32 {
        match self {
            FormulaExpr::Rational { .. } | FormulaExpr::Raw { .. } => 0,
            FormulaExpr::Symbol(_) => 1,
            FormulaExpr::Add(terms) => terms.iter().map(FormulaExpr::degree).max().unwrap_or(0),
            FormulaExpr::Mul(factors) => factors.iter().map(FormulaExpr::degree).sum(),
            FormulaExpr::Div(numerator, _) => numerator.degree(),
            FormulaExpr::Pow(base, exponent) => base.degree().saturating_mul(*exponent),
            FormulaExpr::Sqrt(expr) => expr.degree(),
        }
    }

    fn sort_key(&self) -> String {
        self.to_plain()
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
            FormulaExpr::Raw { plain, latex } => match style {
                RenderStyle::Plain => (4, plain.clone()),
                RenderStyle::Latex => (4, latex.clone()),
            },
            FormulaExpr::Add(terms) => {
                let mut rendered = String::new();
                for (index, term) in terms.iter().enumerate() {
                    let (negative, magnitude) = term.render_signed(style);
                    if index == 0 {
                        if negative {
                            rendered.push('-');
                        }
                        rendered.push_str(&magnitude);
                    } else if negative {
                        rendered.push_str(" - ");
                        rendered.push_str(&magnitude);
                    } else {
                        rendered.push_str(" + ");
                        rendered.push_str(&magnitude);
                    }
                }
                (1, rendered)
            }
            FormulaExpr::Mul(factors) => {
                let rendered = render_product(factors, style);
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

    fn render_signed(&self, style: RenderStyle) -> (bool, String) {
        match self.clone().extract_sign() {
            (true, magnitude) => (true, magnitude.render(style, 2)),
            (false, magnitude) => (false, magnitude.render(style, 1)),
        }
    }

    fn extract_sign(self) -> (bool, FormulaExpr) {
        match self {
            FormulaExpr::Rational {
                numerator,
                denominator,
            } if numerator < 0 => (true, FormulaExpr::rational(-numerator, denominator)),
            FormulaExpr::Mul(factors) => {
                if let Some(FormulaExpr::Rational {
                    numerator,
                    denominator,
                }) = factors.first()
                    && *numerator < 0
                {
                    let mut magnitude = Vec::new();
                    if *numerator != -1 || factors.len() == 1 {
                        magnitude.push(FormulaExpr::rational(-numerator, *denominator));
                    }
                    magnitude.extend(factors.into_iter().skip(1));
                    return (true, FormulaExpr::mul(magnitude));
                }
                (false, FormulaExpr::Mul(factors))
            }
            value => (false, value),
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

    pub fn quasi_polynomial(
        &self,
        qpoly: QuasiPolynomial<'a>,
    ) -> Result<FormulaExpr, barvinok::Error> {
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

                let symbol =
                    FormulaExpr::symbol(self.space.get_dim_name(ty, index)?.unwrap_or("unnamed"));
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

                let factor =
                    FormulaExpr::symbol(self.space.get_dim_name(ty, index)?.unwrap_or("unnamed"));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RationalValue {
    numerator: i64,
    denominator: i64,
}

impl RationalValue {
    fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }

    fn one() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    fn new(numerator: i64, denominator: i64) -> Self {
        debug_assert!(denominator != 0, "denominator must not be zero");
        if numerator == 0 {
            return Self::zero();
        }

        let sign = if denominator < 0 { -1 } else { 1 };
        let numerator = numerator * sign;
        let denominator = denominator.abs();
        let gcd = gcd_i64(numerator.abs(), denominator);
        Self {
            numerator: numerator / gcd,
            denominator: denominator / gcd,
        }
    }

    fn add(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )
    }

    fn reciprocal(self) -> Self {
        Self::new(self.denominator, self.numerator)
    }

    fn is_zero(self) -> bool {
        self.numerator == 0
    }

    fn is_one(self) -> bool {
        self.numerator == 1 && self.denominator == 1
    }

    fn sqrt(self) -> Option<Self> {
        if self.numerator < 0 {
            return None;
        }

        let numerator = perfect_square_root(self.numerator)?;
        let denominator = perfect_square_root(self.denominator)?;
        Some(Self::new(numerator, denominator))
    }

    fn into_expr(self) -> FormulaExpr {
        FormulaExpr::rational(self.numerator, self.denominator)
    }
}

fn split_coefficient(term: FormulaExpr) -> (RationalValue, Option<FormulaExpr>) {
    match term {
        FormulaExpr::Rational {
            numerator,
            denominator,
        } => (RationalValue::new(numerator, denominator), None),
        FormulaExpr::Mul(factors) => {
            let mut coefficient = RationalValue::one();
            let mut basis = Vec::new();
            for factor in factors {
                if let Some(rational) = factor.as_rational() {
                    coefficient = coefficient.mul(rational);
                } else {
                    basis.push(factor);
                }
            }

            let basis = match basis.len() {
                0 => None,
                1 => basis.into_iter().next(),
                _ => Some(FormulaExpr::Mul(basis)),
            };
            (coefficient, basis)
        }
        value => (RationalValue::one(), Some(value)),
    }
}

fn build_term(coefficient: RationalValue, basis: FormulaExpr) -> FormulaExpr {
    if coefficient.is_one() {
        basis
    } else {
        FormulaExpr::mul([coefficient.into_expr(), basis])
    }
}

fn render_product(factors: &[FormulaExpr], style: RenderStyle) -> String {
    let joiner = match style {
        RenderStyle::Plain => " * ",
        RenderStyle::Latex => " \\cdot ",
    };

    if let Some(FormulaExpr::Rational {
        numerator,
        denominator,
    }) = factors.first()
        && *numerator < 0
    {
        let coefficient = RationalValue::new(*numerator, *denominator);
        let magnitude = RationalValue::new(-coefficient.numerator, coefficient.denominator);
        let mut rendered_factors = Vec::new();
        if !magnitude.is_one() || factors.len() == 1 {
            rendered_factors.push(magnitude.into_expr().render(style, 2));
        }
        rendered_factors.extend(factors.iter().skip(1).map(|factor| factor.render(style, 2)));
        return format!("-{}", rendered_factors.join(joiner));
    }

    factors
        .iter()
        .map(|factor| factor.render(style, 2))
        .collect::<Vec<_>>()
        .join(joiner)
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

fn perfect_square_root(value: i64) -> Option<i64> {
    if value < 0 {
        return None;
    }
    let root = (value as f64).sqrt() as i64;
    if root * root == value {
        Some(root)
    } else if (root + 1) * (root + 1) == value {
        Some(root + 1)
    } else {
        None
    }
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

    #[test]
    fn simplifies_duplicate_terms_and_subtractions() {
        let expr = FormulaExpr::add([
            FormulaExpr::one(),
            FormulaExpr::mul([FormulaExpr::rational(-1, 1), FormulaExpr::symbol("N")]),
            FormulaExpr::mul([FormulaExpr::rational(-1, 1), FormulaExpr::symbol("M")]),
            FormulaExpr::mul([FormulaExpr::symbol("N"), FormulaExpr::symbol("M")]),
            FormulaExpr::mul([FormulaExpr::rational(-1, 1), FormulaExpr::one()]),
            FormulaExpr::symbol("N"),
            FormulaExpr::symbol("N"),
        ]);

        assert_eq!(expr.to_plain(), "N * M + N - M");
        assert_eq!(expr.to_latex(), "N \\cdot M + N - M");
    }

    #[test]
    fn simplifies_square_roots_of_perfect_squares() {
        let expr = FormulaExpr::sqrt(FormulaExpr::add([
            FormulaExpr::rational(2, 1),
            FormulaExpr::rational(2, 1),
        ]));
        assert_eq!(expr.to_plain(), "2");
    }

    #[test]
    fn distributes_scalar_over_addition() {
        let expr = FormulaExpr::mul([
            FormulaExpr::rational(2, 1),
            FormulaExpr::add([FormulaExpr::symbol("N"), FormulaExpr::rational(-1, 1)]),
        ]);
        assert_eq!(expr.to_plain(), "2 * N - 2");
        assert_eq!(expr.to_latex(), "2 \\cdot N - 2");
    }
}
