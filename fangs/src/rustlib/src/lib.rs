mod registration;
mod timers;

use ndarray::prelude::*;
use ndarray::{Array1, Zip};
use rand::{Rng, RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use rayon::ThreadPool;
use roxido::*;
use std::convert::TryFrom;
use std::path::Path;
use timers::{EchoTimer, PeriodicTimer};

#[roxido]
fn fangs(
    samples: SEXP,
    n_iterations: SEXP,
    max_n_features: SEXP,
    n_candidates: SEXP,
    n_bests: SEXP,
    n_cores: SEXP,
    quiet: SEXP,
) -> SEXP {
    let mut timer = EchoTimer::new();
    let n_samples = samples.length();
    if n_samples < 1 {
        panic!("Number of samples must be at least one.");
    }
    let max_n_features = max_n_features.as_usize().max(0);
    let n_candidates = n_candidates.as_usize().max(1).min(n_samples);
    let n_bests = n_bests.as_usize().max(1).min(n_candidates);
    let n_iterations = n_iterations.as_usize().max(0);
    let n_cores = n_cores.as_usize().max(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores)
        .build()
        .unwrap();
    let quiet = quiet.as_bool();
    let status_file = match std::env::var("FANGS_STATUS") {
        Ok(x) => Path::new(x.as_str()).to_owned(),
        _ => std::env::current_dir()
            .unwrap_or_default()
            .join("FANGS_STATUS"),
    };
    let o = samples.get_list_element(0);
    if !o.is_double() || !o.is_matrix() {
        panic!("All elements of 'samples' must be integer matrices.");
    }
    let n_items = o.nrow_usize();
    let mut max_n_features_observed = 0;
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let mut interrupted = false;
    if timer.echo() {
        interrupted |= r::print(
            timer
                .stamp(
                    format!(
                        "Parsed parameters and using {} threads.\n",
                        pool.current_num_threads()
                    )
                    .as_str(),
                )
                .unwrap()
                .as_str(),
        );
        r::flush_console();
    }
    let mut views = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let o = samples.get_list_element(i);
        if !o.is_double() || !o.is_matrix() || o.nrow_usize() != n_items {
            panic!("All elements of 'samples' must be double matrices with a consistent number of rows.");
        }
        let view = make_view(o);
        max_n_features_observed += max_n_features_observed.max(view.ncols());
        views.push(view)
    }
    if timer.echo() {
        interrupted |= r::print(timer.stamp("Made data structures.\n").unwrap().as_str());
        r::flush_console();
    }
    let selected_candidates_with_rngs: Vec<_> =
        rand::seq::index::sample(&mut rng, n_samples, n_candidates)
            .into_iter()
            .map(|index| {
                let mut seed = [0_u8; 16];
                rng.fill_bytes(&mut seed);
                let new_rng = Pcg64Mcg::from_seed(seed);
                (views[index], new_rng)
            })
            .collect();
    if timer.echo() {
        interrupted |= r::print(timer.stamp("Selected all candidates.\n").unwrap().as_str());
        r::flush_console();
    }
    let selected_candidates_with_rngs: Vec<_> = pool.install(|| {
        selected_candidates_with_rngs
            .into_par_iter()
            .map(|(view, mut rng)| {
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
                (z, rng)
            })
            .collect()
    });
    if timer.echo() {
        interrupted |= r::print(
            timer
                .stamp("Reduced number of features for all candidates.\n")
                .unwrap()
                .as_str(),
        );
        r::flush_console();
    }
    let mut candidates = Vec::with_capacity(selected_candidates_with_rngs.len());
    for (z, rng) in selected_candidates_with_rngs {
        if interrupted || r::check_user_interrupt() {
            panic!("Caught user interrupt before main loop, so aborting.");
        }
        let loss = expected_loss_from_samples(z.view(), &views, &pool);
        candidates.push((z, loss, rng));
    }
    if timer.echo() {
        interrupted |= r::print(
            timer
                .stamp("Computed expected loss for all candidates.\n")
                .unwrap()
                .as_str(),
        );
        r::flush_console();
    }
    candidates.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    candidates.truncate(n_bests);
    let mut bests: Vec<_> = pool.install(|| {
        candidates
            .into_par_iter()
            .enumerate()
            .map(|(id, (z, loss, rng))| {
                let weight_matrices = make_weight_matrices(z.view(), &views, &pool);
                let n_accepts = 0;
                let when = 1;
                (z, weight_matrices, loss, id, n_accepts, when, rng)
            })
            .collect()
    });
    if timer.echo() {
        interrupted |= r::print(
            timer
                .stamp("Computed weight matrices for bests.\n")
                .unwrap()
                .as_str(),
        );
        r::flush_console();
    }
    let threshold_in_secs = 1.0;
    let mut period_timer = PeriodicTimer::new(threshold_in_secs);
    let mut iteration_counter = 0;
    while iteration_counter < n_iterations {
        iteration_counter += 1;
        pool.install(|| {
            bests
                .par_iter_mut()
                .for_each(|(z, weight_matrices, loss, _, n_accepts, when, rng)| {
                    let n_features = z.ncols();
                    let total_length = n_items * n_features;
                    fn index_1d_to_2d(index: usize, ncols: usize) -> [usize; 2] {
                        [index / ncols, index % ncols]
                    }
                    let index = index_1d_to_2d(rng.gen_range(0..total_length), n_features);
                    flip_bit(z, weight_matrices, index, &views);
                    let new_loss = expected_loss_from_weight_matrices(&weight_matrices, &pool);
                    if new_loss < *loss {
                        *n_accepts += 1;
                        *when = iteration_counter;
                        *loss = new_loss;
                    } else {
                        flip_bit(z, weight_matrices, index, &views);
                    }
                });
        });
        if !quiet || status_file.exists() {
            period_timer.maybe(iteration_counter == n_iterations, || {
                if quiet && status_file.exists() {
                    interrupted |= r::print(
                        format!(
                            "*** {} exists, so forcing status display.\n",
                            status_file.display()
                        )
                        .as_str(),
                    );
                    r::flush_console();
                }
                bests.sort_unstable_by(|x, y| x.2.partial_cmp(&y.2).unwrap());
                let best = bests.first().unwrap();
                interrupted |= r::print(
                    format!(
                        "\rIter. {}: Since iter. {}, E(loss) is {:.4} from #{} with {} accepts ",
                        iteration_counter,
                        best.5,
                        best.2,
                        best.3 + 1,
                        best.4,
                    )
                    .as_str(),
                );
                r::flush_console();
            });
        }
        if interrupted || r::check_user_interrupt() {
            r::print("\nCaught user interrupt, so breaking out early.");
            r::flush_console();
            break;
        }
    }
    if !quiet {
        r::print("\n");
        r::flush_console();
    }
    if timer.echo() {
        r::print(timer.stamp("Sweetened bests.\n").unwrap().as_str());
        r::flush_console();
    }
    bests.sort_unstable_by(|x, y| x.2.partial_cmp(&y.2).unwrap());
    let (best_z, _, best_loss, candidate_number, n_accepts, _, _) = bests.swap_remove(0);
    if timer.echo() {
        r::print(
            format!(
                "Best result is {} from candidate {} after {} accepted proposal{}.\n",
                best_loss,
                candidate_number,
                n_accepts,
                if n_accepts == 1 { "" } else { "s" }
            )
            .as_str(),
        );
        r::flush_console();
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
    let estimate = r::mk_double_matrix(n_items, columns_to_keep.len()).protect();
    columns_to_keep
        .iter()
        .enumerate()
        .for_each(|(j_new, j_old)| {
            matrix_copy_into_column(estimate, j_new, best_z.column(*j_old).iter())
        });
    let list = r::mk_list_vector_with_names_and_values(&[
        ("estimate", estimate),
        ("loss", r::mk_double_scalar(best_loss).protect()),
        (
            "nIterations",
            r::mk_integer_scalar(i32::try_from(iteration_counter).unwrap()),
        ),
        ("seconds", r::mk_double_scalar(timer.total_as_secs_f64())),
    ]);
    r::unprotect(2);
    if timer.echo() {
        r::print(timer.stamp("Finalized results.\n").unwrap().as_str());
        r::flush_console();
    }
    list
}

#[roxido]
fn compute_expected_loss(z: SEXP, samples: SEXP, n_cores: SEXP) -> SEXP {
    let n_cores = n_cores.as_usize().max(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores)
        .build()
        .unwrap();
    let n_samples = samples.length();
    let mut views = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        views.push(make_view(samples.get_list_element(i)));
    }
    let loss = expected_loss_from_samples(make_view(z), &views, &pool);
    r::mk_double_scalar(loss)
}

#[roxido]
fn compute_loss(z1: SEXP, z2: SEXP) -> SEXP {
    let loss = if z1.is_double() && z2.is_double() {
        match make_weight_matrix(make_view(z1), make_view(z2)) {
            Some(weight_matrix) => loss(&weight_matrix),
            None => 0.0,
        }
    } else {
        panic!("Unsupported or inconsistent types for 'Z1' and 'Z2'.");
    };
    r::mk_double_scalar(loss)
}

fn matrix_copy_into_column<'a>(matrix: SEXP, j: usize, iter: impl Iterator<Item = &'a f64>) {
    let slice = matrix.as_double_slice_mut();
    let nrow = matrix.nrow_usize();
    let subslice = &mut slice[(j * nrow)..((j + 1) * nrow)];
    subslice.iter_mut().zip(iter).for_each(|(x, y)| *x = *y);
}

fn make_view(z: SEXP) -> ArrayView2<'static, f64> {
    unsafe { ArrayView::from_shape_ptr((z.nrow_usize(), z.ncol_usize()).f(), z.as_double_ptr()) }
}

fn make_weight_matrices(
    z: ArrayView2<f64>,
    samples: &[ArrayView2<f64>],
    pool: &ThreadPool,
) -> Vec<Array2<f64>> {
    pool.install(|| {
        samples
            .par_iter()
            .map(|zz| make_weight_matrix(z, *zz).unwrap())
            .collect()
    })
}

fn flip_bit(
    z: &mut Array2<f64>,
    matrices: &mut Vec<Array2<f64>>,
    index: [usize; 2],
    samples: &[ArrayView2<f64>],
) {
    let old_bit = z[index];
    z[index] = if old_bit == 0.0 { 1.0 } else { 0.0 };
    let result = samples.iter().zip(matrices.iter_mut()).for_each(|(zz, w)| {
        for i2 in 0..w.ncols() {
            let bit_in_sample = if i2 >= zz.ncols() {
                0.0
            } else {
                zz[[index[0], i2]]
            };
            w[[index[1], i2]] += if old_bit != bit_in_sample { -1.0 } else { 1.0 };
        }
    });
    /*
    // Sanity check, but commented out for speed.
    let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    assert_eq!(
        expected_loss_from_samples(z.view(), samples, &pool),
        expected_loss_from_weight_matrices(matrices, &pool)
    );
    */
    result
}

fn make_weight_matrix(y1: ArrayView2<f64>, y2: ArrayView2<f64>) -> Option<Array2<f64>> {
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
    Some(unsafe { Array::from_shape_vec_unchecked((k, k), vec) })
}

fn expected_loss_from_samples(
    z: ArrayView2<f64>,
    samples: &[ArrayView2<f64>],
    pool: &ThreadPool,
) -> f64 {
    pool.install(|| {
        samples
            .par_iter()
            .fold(
                || 0.0,
                |acc: f64, zz: &ArrayView2<f64>| {
                    acc + match make_weight_matrix(z, *zz) {
                        Some(weight_matrix) => loss(&weight_matrix),
                        None => 0.0,
                    }
                },
            )
            .reduce(|| 0.0, |a, b| a + b)
            / (samples.len() as f64)
    })
}

fn expected_loss_from_weight_matrices(weight_matrices: &[Array2<f64>], pool: &ThreadPool) -> f64 {
    pool.install(|| {
        weight_matrices
            .par_iter()
            .fold(|| 0.0, |acc: f64, w: &Array2<f64>| acc + loss(w))
            .reduce(|| 0.0, |a, b| a + b)
            / (weight_matrices.len() as f64)
    })
}

fn loss(weight_matrix: &Array2<f64>) -> f64 {
    let solution = lapjv::lapjv(weight_matrix).unwrap();
    lapjv::cost(weight_matrix, &solution.0)
}
