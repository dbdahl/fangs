mod registration;

use munkres::{solve_assignment, WeightMatrix};
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Zip;
use num_traits::identities::Zero;
use roxido::*;

#[no_mangle]
extern "C" fn compute_expected_loss(z: SEXP, samples: SEXP) -> SEXP {
    panic_to_error!({
        let n_samples = samples.xlength_usize();
        let mut sum = 0.0;
        if z.is_integer() {
            let z = view_integer(z);
            for i in 0..n_samples {
                let o = samples.get_list_element(i as isize);
                sum += compute_loss_from_views(z, view_integer(o))
            }
        } else if z.is_double() {
            let z = view_real(z);
            for i in 0..n_samples {
                let o = samples.get_list_element(i as isize);
                sum += compute_loss_from_views(z, view_real(o))
            }
        } else if z.is_logical() {
            let z = view_logical(z);
            for i in 0..n_samples {
                let o = samples.get_list_element(i as isize);
                sum += compute_loss_from_views(z, view_logical(o))
            }
        } else {
            return r::error("Unsupported type for 'Z'.");
        };
        r::double_scalar(sum / (n_samples as f64))
    })
}

#[no_mangle]
extern "C" fn compute_loss(z1: SEXP, z2: SEXP) -> SEXP {
    panic_to_error!({
        let loss = if z1.is_integer() && z2.is_integer() {
            compute_loss_from_views(view_integer(z1), view_integer(z2))
        } else if z1.is_double() && z2.is_double() {
            compute_loss_from_views(view_real(z1), view_real(z2))
        } else if z1.is_logical() && z2.is_logical() {
            compute_loss_from_views(view_logical(z1), view_logical(z2))
        } else {
            return r::error("Unsupported or inconsistent types for 'Z1' and 'Z2'");
        };
        r::double_scalar(loss)
    })
}

fn view_integer(z: SEXP) -> ArrayView2<'static, i32> {
    unsafe {
        ArrayView::from_shape_ptr((z.nrow_usize(), z.ncol_usize()).f(), rbindings::INTEGER(z))
    }
}

fn view_real(z: SEXP) -> ArrayView2<'static, f64> {
    unsafe { ArrayView::from_shape_ptr((z.nrow_usize(), z.ncol_usize()).f(), rbindings::REAL(z)) }
}

fn view_logical(z: SEXP) -> ArrayView2<'static, i32> {
    unsafe {
        ArrayView::from_shape_ptr((z.nrow_usize(), z.ncol_usize()).f(), rbindings::LOGICAL(z))
    }
}

fn compute_loss_from_views<A: Clone + Zero + PartialEq>(
    y1: ArrayView2<A>,
    y2: ArrayView2<A>,
) -> f64 {
    let k1 = y1.ncols();
    let k2 = y2.ncols();
    let k = k1.max(k2);
    let mut vec = Vec::with_capacity(k * k);
    let zero = Array1::zeros(y1.nrows());
    let zero_view = zero.view();
    for i1 in 0..k {
        let x1 = if i1 >= k1 { zero_view } else { y1.column(i1) };
        for i2 in 0..k {
            let x2 = if i2 >= k2 { zero_view } else { y2.column(i2) };
            vec.push(
                Zip::from(&x1)
                    .and(&x2)
                    .fold(0.0, |acc, a, b| acc + if *a != *b { 1.0 } else { 0.0 }),
            );
        }
    }
    let mut w = WeightMatrix::from_row_vec(k, vec.clone());
    let solution = solve_assignment(&mut w);
    let w = unsafe { Array2::from_shape_vec_unchecked((k, k), vec) };
    solution
        .unwrap()
        .into_iter()
        .fold(0.0, |acc, pos| acc + w[[pos.row, pos.column]])
}
