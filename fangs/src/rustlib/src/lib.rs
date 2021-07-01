mod registration;
mod timers;

use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Zip;
use num_traits::identities::Zero;
use rand::RngCore;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use roxido::*;
use timers::{EchoTimer, TicToc};

#[no_mangle]
extern "C" fn fangs(
    samples: SEXP,
    n_iterations: SEXP,
    max_n_features: SEXP,
    n_samples: SEXP,
    n_best: SEXP,
    k: SEXP,
    prob_flip: SEXP,
    n_cores: SEXP,
) -> SEXP {
    panic_to_error!({
        let mut timer = EchoTimer::new();
        let n_samples_in_all = samples.length_usize();
        if n_samples_in_all < 1 {
            return r::error("Number of samples must be at least one.");
        }
        let max_n_features = max_n_features.as_integer().max(0) as usize;
        let n_samples = (n_samples.as_integer().max(1) as usize).min(n_samples_in_all);
        let n_best = (n_best.as_integer().max(1) as usize).min(n_samples);
        let k = k.as_integer().max(0) as u64;
        let n_iterations = n_iterations.as_integer().max(0) as usize;
        let n_cores = n_cores.as_integer().max(0) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_cores)
            .build()
            .unwrap();
        let o = samples.get_list_element(0);
        if !o.is_double() || !o.is_matrix() {
            return r::error("All elements of 'samples' must be integer matrices.");
        }
        let n_items = o.nrow_usize();
        let mut views = Vec::with_capacity(n_samples);
        let mut mean_density = 0.0;
        let mut max_n_features_observed = 0;
        let dynamic_prob_flip = k > 0 && prob_flip.is_nil();
        let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
        if timer.echo() {
            r::print(timer.stamp("Parsed parameters.\n").unwrap().as_str())
        }
        for i in 0..n_samples_in_all {
            let o = samples.get_list_element(i as isize);
            if !o.is_double() || !o.is_matrix() || o.nrow_usize() != n_items {
                return r::error("All elements of 'samples' must be double matrices with a consistent number of rows.");
            }
            let view = view_double(o);
            mean_density += if dynamic_prob_flip {
                density(view)
            } else {
                0.0
            };
            max_n_features_observed += max_n_features_observed.max(view.ncols());
            views.push(view)
        }
        let prob_flip = if dynamic_prob_flip {
            mean_density /= n_samples_in_all as f64;
            0.5 - (mean_density - 0.5).abs()
        } else {
            prob_flip.as_double().min(1.0).max(0.0)
        };
        if timer.echo() {
            r::print(timer.stamp("Made views.\n").unwrap().as_str())
        }
        let selected_views_with_rngs: Vec<_> =
            rand::seq::index::sample(&mut rng, n_samples_in_all, n_samples)
                .into_iter()
                .map(|i| {
                    let mut seed = [0_u8; 16];
                    rng.fill_bytes(&mut seed);
                    let new_rng = Pcg64Mcg::from_seed(seed);
                    (i as usize, new_rng)
                })
                .collect();
        if timer.echo() {
            r::print(timer.stamp("Selected views.\n").unwrap().as_str())
        }
        let mut candidates: Vec<_> = pool.install(|| {
            selected_views_with_rngs
                .into_par_iter()
                .map(|(index, mut rng)| {
                    let view = views[index];
                    let n_features_in_view = view.ncols();
                    let n_features = if max_n_features == 0 {
                        n_features_in_view
                    } else {
                        max_n_features.min(n_features_in_view)
                    };
                    let selected_columns: Vec<_> =
                        rand::seq::index::sample(&mut rng, view.ncols(), n_features).into_vec();
                    let z = Array2::from_shape_fn((n_items, n_features), |(i, j)| {
                        view[[i, selected_columns[j]]]
                    });
                    let loss = compute_expected_loss_from_views(z.view(), &views);
                    (z, loss, rng)
                })
                .collect()
        });
        candidates.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        candidates.truncate(n_best);
        if timer.echo() {
            r::print(
                timer
                    .stamp("Computed expected loss for candidates.\n")
                    .unwrap()
                    .as_str(),
            )
        }
        let mut bests: Vec<_> = pool.install(|| {
            candidates.into_par_iter().enumerate()
                .map(
                    |(index_into_candidates, (mut current_z, mut current_loss, mut rng))| {
                        let mut clock1 = TicToc::new();
                        let mut clock2 = TicToc::new();
                        let n_features = current_z.ncols();
                        let binomial = Binomial::new(k, prob_flip).unwrap();
                        let total_length = n_items * n_features;
                        let mut n_accepts = 0;
                        for iteration_counter in 0..n_iterations {
                            fn index_1d_to_2d(index: usize, ncols: usize) -> [usize; 2] {
                                [index / ncols, index % ncols]
                            }
                            let n_to_flip = 1 + binomial.sample(&mut rng);
                            let mut restoration_table = Vec::with_capacity(n_to_flip as usize);
                            for index in rand::seq::index::sample(
                                &mut rng,
                                total_length,
                                (n_to_flip as usize).min(total_length),
                            ) {
                                let index = index_1d_to_2d(index, n_features);
                                let current_bit = current_z[index];
                                restoration_table.push((index, current_bit));
                                current_z[index] = if current_bit == 0.0 { 1.0 } else { 0.0 };
                            }
                            let new_loss = if timer.echo() {
                                compute_expected_loss_from_views_timed(current_z.view(), &views, &mut clock1, &mut clock2)
                            } else {
                                compute_expected_loss_from_views(current_z.view(), &views)
                            };
                            if new_loss < current_loss {
                                n_accepts += 1;
                                current_loss = new_loss;
                                if timer.echo() {
                                    // R is not thread-safe, so I cannot call r::print() here.
                                    println!(
                                        "Candidate {} improved to {} by flipping {} bit{} at iteration {}.",
                                        index_into_candidates,
                                    current_loss,
                                        n_to_flip,
                                        if n_to_flip == 1 { "" } else { "s" },
                                        iteration_counter
                                    )
                                }
                            } else {
                                for (index, value) in restoration_table {
                                    current_z[index] = value;
                                }
                            }
                        }

                        (current_z, current_loss, index_into_candidates, n_accepts, clock1, clock2)
                    },
                )
                .collect()
        });
        if timer.echo() {
            r::print(timer.stamp("Sweetened bests.\n").unwrap().as_str())
        }
        bests.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let (best_z, best_loss, index_into_candidates, n_accepts, clock1, clock2) =
            bests.swap_remove(0);
        if timer.echo() {
            r::print(
                format!(
                    "Clock 1 measured {}s, Clock 2 measured {}s.\nBest result is {} from candidate {} after {} accepted proposal{}.\n",
                    clock1.as_secs_f64(), clock2.as_secs_f64(),
                    best_loss,
                    index_into_candidates,
                    n_accepts,
                    if n_accepts == 1 { "" } else { "s" }
                )
                .as_str(),
            )
        }
        let columns_to_keep: Vec<usize> = best_z
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
        let estimate = r::double_matrix(n_items as i32, columns_to_keep.len() as i32).protect();
        columns_to_keep
            .iter()
            .enumerate()
            .for_each(|(j_new, j_old)| {
                matrix_copy_into_column(estimate, j_new, best_z.column(*j_old).iter())
            });
        let list = r::list_vector_with_names_and_values(&[
            ("estimate", estimate),
            ("loss", r::double_scalar(best_loss).protect()),
            ("seconds", r::double_scalar(timer.total_as_secs_f64())),
        ]);
        r::unprotect(2);
        if timer.echo() {
            r::print(timer.stamp("Finalized results.\n").unwrap().as_str())
        }
        list
    })
}

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

fn matrix_copy_into_column<'a>(matrix: SEXP, j: usize, iter: impl Iterator<Item = &'a f64>) {
    let slice = matrix.as_double_slice_mut();
    let nrow = matrix.nrow_usize();
    let subslice = &mut slice[(j * nrow)..((j + 1) * nrow)];
    subslice.iter_mut().zip(iter).for_each(|(x, y)| *x = *y);
}

fn density(y: ArrayView2<f64>) -> f64 {
    (y.iter().filter(|x| **x != 0.0).count() as f64) / (y.len() as f64)
}

fn compute_expected_loss_from_views(z: ArrayView2<f64>, samples: &Vec<ArrayView2<f64>>) -> f64 {
    let sum = samples
        .iter()
        .fold(0.0, |acc, zz| acc + compute_loss_from_views(z, *zz));
    sum / (samples.len() as f64)
}

fn compute_expected_loss_from_views_timed(
    z: ArrayView2<f64>,
    samples: &Vec<ArrayView2<f64>>,
    clock1: &mut TicToc,
    clock2: &mut TicToc,
) -> f64 {
    let sum = samples.iter().fold(0.0, |acc, zz| {
        acc + compute_loss_from_views_timed(z, *zz, clock1, clock2)
    });
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
    match make_weight_matrix(y1, y2) {
        Some(w) => cost(&w),
        None => 0.0,
    }
}

fn compute_loss_from_views_timed<A: Clone + Zero + PartialEq>(
    y1: ArrayView2<A>,
    y2: ArrayView2<A>,
    clock1: &mut TicToc,
    clock2: &mut TicToc,
) -> f64 {
    clock1.tic();
    match make_weight_matrix(y1, y2) {
        Some(w) => {
            clock1.toc();
            clock2.tic();
            let result = cost(&w);
            clock2.toc();
            result
        }
        None => {
            clock1.toc();
            0.0
        }
    }
}

fn make_weight_matrix<A: Clone + Zero + PartialEq>(
    y1: ArrayView2<A>,
    y2: ArrayView2<A>,
) -> Option<lapjv::Matrix<f64>> {
    let k1 = y1.ncols();
    let k2 = y2.ncols();
    let k = k1.max(k2);
    if k == 0 {
        return None;
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
    Some(unsafe { lapjv::Matrix::from_shape_vec_unchecked((k, k), vec) })
}

fn cost(weight_matrix: &lapjv::Matrix<f64>) -> f64 {
    let solution = lapjv::lapjv(weight_matrix).unwrap();
    lapjv::cost(weight_matrix, &solution.0)
}
