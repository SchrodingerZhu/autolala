use std::{collections::HashMap};

use raffine::{
    affine::{AffineExpr, AffineMap},
    tree::{Tree, ValID},
};
use symbolica::domains::{Field, rational_polynomial::RationalPolynomialField};

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
                reuse_factors.insert(0x5eabed, isize_to_poly(1, context));
            }

            for (_key, value) in reuse_factors.iter_mut() {
                *value = &*value * &trip_count.clone();
            }

            let ValID::IVar(id) = ivar else {
                unreachable!("not possible ")
            };
            reuse_factors.insert(*id, isize_to_poly(1, context));
            trip_counts.insert(*id, trip_count.clone());

            get_reuse_interval_distribution(
                body,
                reuse_factors,
                trip_counts,
                ref_count,
                context,
            )
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
                        *ri_dist.get_mut(i).unwrap() = ri_dist.get(i).unwrap() +  j;
                    } else {
                        ri_dist.insert(i.clone(), j.clone());
                    }
                }
            }
            ri_dist
        }
        Tree::Access { map, operands, .. } => {
            let mut reference_vector = vec![0; reuse_factors.len()];
            let converted_map = convert_affine_map(*map, operands);

            if let Result::Ok(polys) = converted_map {
                for poly in polys {
                    for var in poly.numerator.variables.iter() {
                        if var.to_string().starts_with('i') {
                            let index_str = &var.to_string()[1..];
                            let index = index_str.parse::<usize>().unwrap();
                            reference_vector[index + 1] = 1;
                        }
                    }
                }
            }
            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);

            let mut portion = isize_to_poly(1, context);
            let mut shrinked_ref_vec = vec![];
            let mut portions = vec![];

            for i in (0..reference_vector.len() - 1).rev() {
                if reference_vector[i] != reference_vector[i + 1] {
                    shrinked_ref_vec.push(i);
                }
                if reference_vector[i + 1] == 0 {
                    portion = field.div(&portion, trip_counts.get(&(i)).unwrap());
                }
                portions.push(portion.clone());
            }

            shrinked_ref_vec.reverse();
            if reference_vector[reference_vector.len() - 1] == 0 {
                shrinked_ref_vec.push(reference_vector.len() - 1);
            }

            portions.reverse();

            let mut ri_portion_sum = isize_to_poly(0, context);

            let mut coefficient = 1;
            let n_ref = isize_to_poly(ref_count as isize, context);
            let mut ri_value = isize_to_poly(0, context);
            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();
            for (place, i) in (shrinked_ref_vec.iter().rev()).enumerate() {
                let tmp = if *i == 0 {
                    reuse_factors.get(&(0x5eabed)).unwrap() * &isize_to_poly(coefficient, context)
                } else {
                    reuse_factors.get(&(i - 1)).unwrap() * &isize_to_poly(coefficient, context)
                };
                ri_value = &ri_value + &tmp;
                if coefficient == 1 {
                    if place != shrinked_ref_vec.len() - 1 {
                        let ri_portion = portions[i - 1].clone();
                        if ri_dist.contains_key(&(&ri_value * &n_ref)) {
                            let r1 = ri_dist.get(&(&ri_value * &n_ref)).unwrap().clone();
                            let r2 = field.div(&(&ri_portion - &ri_portion_sum), &n_ref);

                            *ri_dist.get_mut(&(&ri_value * &n_ref)).unwrap() = &r1 + &r2;
                        }
                        else {
                            ri_dist.insert(
                            &ri_value * &n_ref,
                            field.div(&(&ri_portion - &ri_portion_sum), &n_ref));
                        }
                        ri_portion_sum = ri_portion;
                    } else {
                        if ri_dist.contains_key(&(&ri_value * &n_ref)) {
                            let r1 = ri_dist.get(&(&ri_value * &n_ref)).unwrap().clone();
                            let r2 = field.div(&(&isize_to_poly(1, context) - &ri_portion_sum), &n_ref);
                            *ri_dist.get_mut(&(&ri_value * &n_ref)).unwrap() = &r1 + &r2;
                        }
                        else {
                            ri_dist.insert(
                            &ri_value * &n_ref,
                            field.div(&(&isize_to_poly(1, context) - &ri_portion_sum), &n_ref));
                        }
                       
                    }
                }
                coefficient *= -1;
            }
            ri_dist
        }
        Tree::If { .. } => {HashMap::new()}
    }
}
