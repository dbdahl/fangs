mod registration;

use munkres::{solve_assignment, WeightMatrix};
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Zip;
use roxido::*;

#[no_mangle]
extern "C" fn compute_loss(z1: SEXP, z2: SEXP) -> SEXP {
    panic_to_error!({
        let z1 = z1.to_integer().protect();
        let z2 = z2.to_integer().protect();
        let n1 = z1.nrow_usize();
        let n2 = z2.nrow_usize();
        let k1 = z1.ncol_usize();
        let k2 = z2.ncol_usize();
        let y1 = unsafe { ArrayView::from_shape_ptr((n1, k1).f(), rbindings::INTEGER(z1)) };
        let y2 = unsafe { ArrayView::from_shape_ptr((n2, k2).f(), rbindings::INTEGER(z2)) };
        let k = k1.max(k2);
        let mut vec = Vec::with_capacity(k * k);
        let zero = Array1::zeros(n1);
        let zero_view = zero.view();
        for i1 in 0..k {
            let x1 = if i1 >= k1 { zero_view } else { y1.column(i1) };
            for i2 in 0..k {
                let x2 = if i2 >= k2 { zero_view } else { y2.column(i2) };
                vec.push(
                    Zip::from(&x1)
                        .and(&x2)
                        .fold(0, |acc, a, b| acc + if *a != *b { 1 } else { 0 }),
                );
            }
        }
        let mut w = WeightMatrix::from_row_vec(k, vec.clone());
        let solution = solve_assignment(&mut w);
        let w = unsafe { Array2::from_shape_vec_unchecked((k, k), vec) };
        r::unprotect(2);
        r::integer_scalar(
            solution
                .unwrap()
                .into_iter()
                .fold(0, |acc, pos| acc + w[[pos.row, pos.column]]),
        )
    })
}
