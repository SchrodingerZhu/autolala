use std::collections::HashMap;

use raffine::{
    affine::{AffineExpr, AffineMap},
    tree::{Tree, ValID},
};
use symbolica::atom::Atom;
use symbolica::atom::AtomCore;
use symbolica::domains::{
    Field, integer::IntegerRing, rational_polynomial::RationalPolynomialField,
};
use symbolica::symbol;

use crate::{
    AnalysisContext,
    utils::{Poly, convert_affine_map},
};

fn isize_to_poly<'a>(value: isize, context: &AnalysisContext<'a>) -> Poly {
    let expr = AffineExpr::new_constant(context.rcontext().mlir_context(), value as i64);
    let map = AffineMap::new(
        context.rcontext().mlir_context(),
        0, // num_dims
        0, // num_symbols
        &[expr],
    );
    let step_converted = convert_affine_map(map, &[]).unwrap();
    step_converted[0].clone()
}

// reuse interval distribution without block
pub fn get_reuse_interval_distribution<'a, 'b: 'a>(
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
) -> HashMap<Poly, Poly> {
    match tree {
        Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands,
            upper_bound_operands,
            body,
            ivar,
            step,
        } => {
            let lower_bound_converted =
                convert_affine_map(*lower_bound, lower_bound_operands).unwrap();
            let upper_bound_converted =
                convert_affine_map(*upper_bound, upper_bound_operands).unwrap();
            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);
            let tmp = upper_bound_converted[0].clone() - lower_bound_converted[0].clone();
            let trip_count = field.div(&tmp, &isize_to_poly(*step, context));
            if reuse_factors.is_empty() {
                reuse_factors.insert(usize::MAX, isize_to_poly(1, context));
            }

            for (_key, value) in reuse_factors.iter_mut() {
                *value = &*value * &trip_count.clone();
            }

            let ValID::IVar(id) = ivar else {
                unreachable!("not possible ")
            };
            reuse_factors.insert(*id, isize_to_poly(1, context));
            trip_counts.insert(*id, trip_count.clone());

            get_reuse_interval_distribution(body, reuse_factors, trip_counts, ref_count, context)
        }
        Tree::Block(trees) => {
            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();
            for subtree in trees.iter() {
                let tmp = get_reuse_interval_distribution(
                    subtree,
                    reuse_factors,
                    trip_counts,
                    trees.len(),
                    context,
                );
                for (i, j) in tmp.iter() {
                    if ri_dist.contains_key(i) {
                        *ri_dist.get_mut(i).unwrap() = ri_dist.get(i).unwrap() + j;
                    } else {
                        ri_dist.insert(i.clone(), j.clone());
                    }
                }
            }
            // check if keys add up ti 1
            let mut sum = isize_to_poly(0, context);
            for (_key, value) in ri_dist.iter() {
                if *value != isize_to_poly(3, context) {
                    sum = &sum + value;
                }
            }
            ri_dist
        }
        Tree::Access { map, operands, .. } => {
            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);
            let mut reference_vector = vec![0; reuse_factors.len()];
            let converted_map = convert_affine_map(*map, operands);
            let mut block_position = 0;
            if let Result::Ok(polys) = converted_map {
                for poly in polys {
                    block_position = 0;
                    for var in poly.numerator.variables.iter() {
                        if var.to_string().starts_with('i') {
                            let index_str = &var.to_string()[1..];
                            let index = index_str.parse::<usize>().unwrap();
                            block_position = block_position.max(index + 1);
                            reference_vector[index + 1] = 1;
                        }
                    }
                }
            }

            reference_vector[block_position] = 1;

            let mut portion_factors = vec![];
            let mut p_factor = isize_to_poly(1, context);
            for i in (0..reference_vector.len()).rev() {
                portion_factors.push(field.div(&isize_to_poly(1, context), &p_factor.clone()));
                if reference_vector[i] == 0 && i != 0 {
                    p_factor = &p_factor * trip_counts.get(&(i - 1)).unwrap()
                }
            }
            portion_factors.reverse();

            reference_vector[block_position] = 2;

            let mut shrinked_ref_vec = vec![];
            let mut coefficients = vec![];
            let mut b_found = false;
            let mut last_position_of_zero_group = reference_vector.len() - 1;

            for i in (0..reference_vector.len() - 1).rev() {
                if reference_vector[i] != reference_vector[i + 1] {
                    if i == block_position {
                        if reference_vector[i + 1] == 1 {
                            coefficients.push(0);
                            last_position_of_zero_group = block_position;
                        } else {
                            reference_vector[i] = 1;
                            b_found = true;
                            continue;
                        }
                    } else if reference_vector[i] == 1 {
                        coefficients.push(-1);
                    } else if reference_vector[i] == 0 {
                        coefficients.push(1);
                        if !b_found {
                            last_position_of_zero_group = i;
                        }
                    }
                    shrinked_ref_vec.push(i);
                }
                if i == block_position {
                    b_found = true;
                }
            }

            shrinked_ref_vec.reverse();
            coefficients.reverse();
            if reference_vector[reference_vector.len() - 1] == 0 {
                shrinked_ref_vec.push(reference_vector.len() - 1);
                coefficients.push(1);
            } else if reference_vector[reference_vector.len() - 1] == 2 {
                shrinked_ref_vec.push(reference_vector.len() - 1);
                coefficients.push(0);
                last_position_of_zero_group = reference_vector.len() - 1;
            }

            let n_ref = isize_to_poly(ref_count as isize, context);
            let mut ri_value = isize_to_poly(0, context);

            let block_atom = Atom::new_var(symbol!("b"));
            let block_poly =
                block_atom.to_rational_polynomial(&IntegerRing::new(), &IntegerRing::new(), None);
            let mut ri_values: Vec<(Poly, usize)> = vec![];

            let mut ri_block_poly = isize_to_poly(-1, context);

            for (place, i) in (shrinked_ref_vec.iter().rev()).enumerate() {
                let factor = if *i != 0 {
                    reuse_factors.get(&(*i - 1)).unwrap()
                } else {
                    reuse_factors.get(&usize::MAX).unwrap()
                };

                let coefficient = coefficients[coefficients.len() - 1 - place];

                if last_position_of_zero_group == *i {
                    ri_value = &ri_value + factor;
                    ri_values.push(((&ri_value.clone() * &n_ref), (*i)));

                    ri_block_poly = &ri_value * &n_ref;
                    let block_factor = if block_position != 0 {
                        reuse_factors.get(&(block_position - 1)).unwrap()
                    } else {
                        reuse_factors.get(&usize::MAX).unwrap()
                    };
                    ri_value = &ri_value - &(&block_poly * block_factor);
                } else if coefficient == -1 {
                    ri_value = &ri_value - factor;
                } else {
                    ri_value = &ri_value + factor;
                    ri_values.push(((&ri_value.clone() * &n_ref), (*i)));
                }
            }

            ri_values.reverse();

            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();

            let mut prev_portion = isize_to_poly(0, context);

            let mut addition = isize_to_poly(0, context);

            for (i, j) in ri_values.iter() {
                let p_factor = portion_factors[*j].clone();
                if *j < block_position {
                    let without_block = &p_factor - &prev_portion;
                    let with_block = field.div(&without_block, &block_poly);
                    addition = &addition + &(&without_block - &with_block);
                    ri_dist.insert(i.clone(), field.div(&with_block, &n_ref));
                } else {
                    let portion = &p_factor - &prev_portion;
                    ri_dist.insert(i.clone(), field.div(&portion, &n_ref));
                }
                prev_portion = p_factor;
            }

            *ri_dist.get_mut(&ri_block_poly).unwrap() =
                ri_dist.get(&ri_block_poly).unwrap() + &addition;

            ri_dist
        }

        Tree::If { .. } => HashMap::new(),
    }
}
