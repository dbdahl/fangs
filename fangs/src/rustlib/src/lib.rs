mod registration;

use munkres::{solve_assignment, WeightMatrix};
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Zip;
use num_traits::identities::Zero;
use rand::SeedableRng;
use rand::{Rng, RngCore};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use roxido::*;

fn stamp(start: std::time::SystemTime) {
    println!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(start)
            .expect("Time went backwards")
            .as_micros()
    );
}

#[no_mangle]
extern "C" fn fangs(
    samples: SEXP,
    n_best: SEXP,
    _k: SEXP,
    _prob1: SEXP,
    n_iterations: SEXP,
    n_cores: SEXP,
) -> SEXP {
    let start = std::time::SystemTime::now();
    panic_to_error!({
        stamp(start);
        let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
        let n_best = n_best.as_integer() as usize;
        let n_iterations = n_iterations.as_integer() as usize;
        let n_cores = n_cores.as_integer().max(0) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_cores)
            .build()
            .unwrap();
        let n_samples = samples.length_usize();
        if n_samples < 1 {
            return r::error("Number of samples must be at least one.");
        }
        let o = samples.get_list_element(0);
        if !o.is_double() || !o.is_matrix() {
            return r::error("All elements of 'samples' must be integer matrices.");
        }
        let n_items = o.nrow_usize();
        let mut views = Vec::with_capacity(n_samples);
        // let mut mean_density = 0.0;
        let mut max_features = 0;
        stamp(start);
        for i in 0..n_samples {
            let o = samples.get_list_element(i as isize);
            if !o.is_double() || !o.is_matrix() || o.nrow_usize() != n_items {
                return r::error("All elements of 'samples' must be double matrices with a consistent number of rows.");
            }
            let view = view_double(o);
            // mean_density += density(view);
            max_features += max_features.max(view.ncols());
            views.push(view)
        }
        stamp(start);
        println!("Score everything");
        // mean_density /= n_samples as f64;
        let mut losses: Vec<_> = pool.install(|| {
            views
                .par_iter()
                .map(|z| compute_expected_loss_from_views(*z, &views))
                .enumerate()
                .collect()
        });
        losses.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        stamp(start);
        println!("nbest: {}", n_best);
        let losses2: Vec<_> = losses
            .iter()
            .take(n_best)
            .map(|x| {
                let mut seed = [0_u8; 16];
                rng.fill_bytes(&mut seed);
                let new_rng = Pcg64Mcg::from_seed(seed);
                (x.0, x.1, new_rng)
            })
            .collect();
        let mut results: Vec<_> = losses2
            .into_par_iter()
            .map(|mut candidate| {
                let mut current_z = views[candidate.0].to_owned();
                let mut current_loss = candidate.1;
                let rng = &mut candidate.2;
                let n_features = current_z.ncols();
                let total_length = n_items * n_features;
                for _ in 0..n_iterations {
                    let index = rng.gen_range(0..total_length);
                    let index = [index / n_features, index % n_features];
                    current_z[index] = if current_z[index] == 0.0 { 1.0 } else { 0.0 };
                    let new_loss = compute_expected_loss_from_views(current_z.view(), &views);
                    if new_loss < current_loss {
                        current_loss = new_loss;
                    } else {
                        current_z[index] = if current_z[index] == 0.0 { 1.0 } else { 0.0 };
                    }
                }
                (current_z, current_loss)
            })
            .collect();
        stamp(start);
        results.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let (raw, loss) = results.swap_remove(0);
        let columns_to_keep: Vec<usize> = raw
            .axis_iter(Axis(1))
            .enumerate()
            .filter_map(|(j, column)| {
                if column.iter().any(|x| *x != 0.0) {
                    Some(j)
                } else {
                    None
                }
            })
            .collect();
        stamp(start);
        let estimate = r::double_matrix(o.nrow(), columns_to_keep.len() as i32).protect();
        columns_to_keep
            .iter()
            .enumerate()
            .for_each(|(j_new, j_old)| {
                matrix_copy_into_column(estimate, j_new, raw.column(*j_old).iter())
            });
        let list = r::list_vector_with_names_and_values(&[
            ("estimate", estimate),
            ("loss", r::double_scalar(loss).protect()),
        ]);
        r::unprotect(2);
        stamp(start);
        list
    })
}

fn matrix_copy_into_column<'a>(matrix: SEXP, j: usize, iter: impl Iterator<Item = &'a f64>) {
    let slice = matrix.as_double_slice_mut();
    let nrow = matrix.nrow_usize();
    let subslice = &mut slice[(j * nrow)..((j + 1) * nrow)];
    subslice.iter_mut().zip(iter).for_each(|(x, y)| *x = *y);
}

/*
fn density(y: ArrayView2<i32>) -> f64 {
    (y.iter().filter(|x| **x != 0).count() as f64) / (y.len() as f64)
}
*/

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
            let z = view_double(z);
            for i in 0..n_samples {
                let o = samples.get_list_element(i as isize);
                sum += compute_loss_from_views(z, view_double(o))
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
            compute_loss_from_views(view_double(z1), view_double(z2))
        } else if z1.is_logical() && z2.is_logical() {
            compute_loss_from_views(view_logical(z1), view_logical(z2))
        } else {
            return r::error("Unsupported or inconsistent types for 'Z1' and 'Z2'.");
        };
        r::double_scalar(loss)
    })
}

fn compute_expected_loss_from_views(z: ArrayView2<f64>, samples: &Vec<ArrayView2<f64>>) -> f64 {
    let sum = samples
        .iter()
        .fold(0.0, |acc, zz| acc + compute_loss_from_views(z, *zz));
    sum / (samples.len() as f64)
}

fn view_integer(z: SEXP) -> ArrayView2<'static, i32> {
    unsafe {
        ArrayView::from_shape_ptr((z.nrow_usize(), z.ncol_usize()).f(), rbindings::INTEGER(z))
    }
}

fn view_double(z: SEXP) -> ArrayView2<'static, f64> {
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
    if k == 0 {
        return 0.0;
    }
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
