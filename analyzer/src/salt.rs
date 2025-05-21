use std::{collections::HashMap};

use raffine::{
    affine::{AffineExpr, AffineMap},
    tree::{Tree, ValID},
};
use symbolica::domains::{integer::IntegerRing,rational_polynomial::RationalPolynomialField, Field};
use symbolica::symbol;
use symbolica::atom::Atom;
use symbolica::atom::AtomCore;

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
           reference_vector[block_position] = 2;


            if block_position != reference_vector.len() - 1 {
               if reference_vector[block_position + 1] == 0 {
                    reference_vector[block_position] = 0;
                }
            }


            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);

            let mut portion = isize_to_poly(1, context);
            let mut shrinked_ref_vec = vec![];
            let mut portions = vec![];

            for i in (0..reference_vector.len() - 1).rev() {
                if reference_vector[i] != reference_vector[i + 1] {
                    if reference_vector[i] == 2 {
                        shrinked_ref_vec.push(-(i as isize));
                    }
                    else {
                        shrinked_ref_vec.push(i as isize);
                    }
                  
                }
                if reference_vector[i + 1] == 0 {
                    portion = field.div(&portion, trip_counts.get(&(i)).unwrap());
                }
                portions.push(portion.clone());
            }
            shrinked_ref_vec.reverse();
            if reference_vector[reference_vector.len() - 1] == 0 {
                shrinked_ref_vec.push((reference_vector.len() - 1) as isize);
                portions.push(field.div(&portion, trip_counts.get(&(reference_vector.len() - 2)).unwrap()));
            }
            else if reference_vector[reference_vector.len() - 1] == 2 {
                shrinked_ref_vec.push(-((reference_vector.len() - 1) as isize));
                portions.push(field.div(&portion, trip_counts.get(&(reference_vector.len() - 2)).unwrap()));
            }
            else {
                portions.push(portion);
            }
            


            let mut last_portion = isize_to_poly(0, context);

            let mut coefficient = 1;
            let n_ref = isize_to_poly(ref_count as isize, context);
            let mut ri_value = isize_to_poly(0, context);
            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();
            let mut block_ri = (isize_to_poly(0, context), isize_to_poly(0, context));
            


            let block_atom = Atom::new_var(symbol!("b"));
            let block_poly = block_atom.to_rational_polynomial(
                &IntegerRing::new(),
                &IntegerRing::new(),
                None,
            );
            
            
            for (place, i) in (shrinked_ref_vec.iter().rev()).enumerate() {
                
                let current_portion = portions[(*i).abs() as usize].clone();
                if *i < 0 {
                    let factor = if -(*i) > 0 {
                        reuse_factors.get( &(((*i).abs() - 1) as usize)).unwrap()
                    }
                    else {
                        reuse_factors.get(&usize::MAX).unwrap()
                    };
                    
                    if ri_value == isize_to_poly(0, context) {
                        ri_value = isize_to_poly(1, context);
                    }
                    else {
                        ri_value = &ri_value - &(&(&block_poly - &isize_to_poly(1, context)) * &factor);
                    }
                    
                    let ri_portion = &current_portion - &last_portion;
                    block_ri = (&ri_value.clone() * &n_ref, field.div(&ri_portion, &n_ref));
                    
                    if reference_vector[((*i).abs() as usize) - 1] == 0 {
                        coefficient = 1;
                    }
                    else {

                        coefficient = -1;
                    }
                    last_portion = current_portion.clone();

                    if ri_value == isize_to_poly(1, context) {
                        ri_value =  &ri_value - &block_poly;
                    }

                    continue;
                }

                
                let factor = if *i > 0 {
                    reuse_factors.get( &(((*i) - 1) as usize)).unwrap()
                }
                else {
                    reuse_factors.get(&usize::MAX).unwrap()
                };
                if coefficient == -1 {
                    ri_value = &ri_value - &factor;
                    coefficient = 1;
                    continue;
                }
                ri_value = &ri_value + &factor;
                if block_ri.0 != isize_to_poly(0, context) {
                    let without_block = if place != shrinked_ref_vec.len() - 1 {
                        field.div(&(&current_portion - &last_portion), &n_ref)
                    }
                    else {
                        field.div(&(&isize_to_poly(1, context) - &last_portion), &n_ref)
                    };

                    let ri_portion = field.div(&without_block, &block_poly);
                    block_ri.1 = &block_ri.1 + &(&without_block - &ri_portion);
                    ri_dist.insert(&ri_value.clone() * &n_ref, ri_portion);
                }
                else {
                    let ri_portion = if place != shrinked_ref_vec.len() - 1 {
                        field.div(&(&current_portion - &last_portion), &n_ref)
                    }
                    else {
                        field.div(&(&isize_to_poly(1, context) - &last_portion), &n_ref)
                    };
                    ri_dist.insert(&ri_value.clone() * &n_ref, ri_portion);

                }
                last_portion = current_portion.clone();

                coefficient = -1;
            }
            if block_ri.0 != isize_to_poly(0, context) {
                ri_dist.insert(block_ri.0, block_ri.1);

            }
            ri_dist
        }
        
        Tree::If { .. } => {HashMap::new()}
    }
}
