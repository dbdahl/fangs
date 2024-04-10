mod registration {
    include!(concat!(env!("OUT_DIR"), "/registration.rs"));
}

mod timers;

use ndarray::prelude::*;
use ndarray::{Array1, Zip};
use rand::{Rng, RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use rayon::ThreadPool;
use roxido::*;
use std::path::Path;
use timers::{EchoTimer, PeriodicTimer};

#[allow(unused_imports)]
use approx::assert_ulps_eq;

fn get(samples: &RList, index: usize) -> &RMatrix<f64> {
    match samples.get(index) {
        Ok(element) => element
            .as_matrix()
            .stop_str("All elements of 'samples' must be a matrix.")
            .as_f64()
            .stop_str("All elements of 'samples' mus be of storage mode 'double'"),
        Err(_) => stop!("Index into 'samples' is out of bounds."),
    }
}

#[roxido]
fn fangs(
    samples: &RList,
    n_iterations: usize,
    max_seconds: f64,
    n_baselines: usize,
    n_sweet: usize,
    a: f64,
    n_cores: usize,
    use_neighbors: bool,
    quiet: bool,
) {
    let mut timer = EchoTimer::new();
    let n_samples = samples.len();
    if n_samples < 1 {
        stop!("Number of samples must be at least one.");
    }
    let threshold = a / 2.0;
    let n_baselines = n_baselines.max(1);
    let n_sweet = (n_sweet.max(1)).min(n_baselines);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores)
        .build()
        .unwrap();
    let status_file = match std::env::var("FANGS_STATUS") {
        Ok(x) => Path::new(x.as_str()).to_owned(),
        _ => std::env::current_dir()
            .unwrap_or_default()
            .join("FANGS_STATUS"),
    };
    let o = get(samples, 0);
    let n_items = o.nrow();
    let mut max_n_features_observed = 0;
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut interrupted = false;
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer
                .stamp(
                    format!(
                        "Parsed parameters.  Using {} threads.\n",
                        pool.current_num_threads()
                    )
                    .as_str(),
                )
                .unwrap()
                .as_str()
        );
        R::flush_console();
    }
    let mut views = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let o = get(samples, i);
        if o.nrow() != n_items {
            stop!("All elements of 'samples' must have the same number of rows.");
        }
        let view = make_view(o);
        max_n_features_observed = max_n_features_observed.max(view.ncols());
        views.push(view)
    }
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer.stamp("Made data structures.\n").unwrap().as_str()
        );
        R::flush_console();
    }
    let baselines_with_rngs: Vec<_> = rand::seq::index::sample(&mut rng, n_samples, n_baselines)
        .into_iter()
        .map(|index| {
            let mut seed = [0_u8; 16];
            rng.fill_bytes(&mut seed);
            let new_rng = Pcg64Mcg::from_seed(seed);
            (views[index], new_rng)
        })
        .collect();
    let initials_with_rngs: Vec<_> = pool.install(|| {
        baselines_with_rngs
            .into_par_iter()
            .map(|(view, rng)| {
                let elementwise_sums = views
                    .par_iter()
                    .map(|zz| {
                        let weight_matrix = make_weight_matrix(view, *zz, a).unwrap();
                        let solution = lapjv::lapjv(&weight_matrix).unwrap();
                        Array2::from_shape_fn((n_items, max_n_features_observed), |(i, j)| {
                            if j >= solution.0.len() {
                                0.0
                            } else {
                                let jj = solution.0[j];
                                if jj >= zz.ncols() {
                                    0.0
                                } else {
                                    zz[[i, jj]]
                                }
                            }
                        })
                    })
                    .reduce(
                        || Array2::zeros((n_items, max_n_features_observed)),
                        |z1, z2| z1 + z2,
                    );
                let elementwise_means = elementwise_sums / (n_samples as f64);
                let initial_estimate_with_zero_columns =
                    elementwise_means.mapv(|x| if x < threshold { 0.0 } else { 1.0 });
                let mut which: Vec<usize> = Vec::new();
                for (column_counter, column) in initial_estimate_with_zero_columns
                    .columns()
                    .into_iter()
                    .enumerate()
                {
                    if column.iter().any(|&x| x > 0.0) {
                        which.push(column_counter)
                    }
                }
                let initial_estimate = if which.is_empty() {
                    Array2::zeros((n_items, 1))
                } else if which.len() == initial_estimate_with_zero_columns.ncols() {
                    initial_estimate_with_zero_columns
                } else {
                    Array2::from_shape_fn((n_items, which.len()), |(i, j)| {
                        initial_estimate_with_zero_columns[[i, which[j]]]
                    })
                };
                (initial_estimate, rng)
            })
            .collect()
    });
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer.stamp("Made initial estimates.\n").unwrap().as_str()
        );
        R::flush_console();
    }
    let mut initials = Vec::with_capacity(initials_with_rngs.len());
    for (z, rng) in initials_with_rngs {
        if interrupted || R::check_user_interrupt() {
            stop!("Caught user interrupt before main loop, so aborting.");
        }
        let weight_matrices = make_weight_matrices(z.view(), &views, a, &pool);
        let loss = expected_loss_from_weight_matrices(&weight_matrices[..], &pool);
        initials.push((z, loss, weight_matrices, rng));
    }
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer
                .stamp("Computed expected loss for all initial estimates.\n")
                .unwrap()
                .as_str()
        );
        R::flush_console();
    }
    initials.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    initials.truncate(n_sweet);
    let mut sweets: Vec<_> = pool.install(|| {
        initials
            .into_par_iter()
            .enumerate()
            .map(|(id, (z, loss, weight_matrices, rng))| {
                let n_accepts = 0;
                let when = 1;
                (z, loss, weight_matrices, id, n_accepts, when, rng)
            })
            .collect()
    });
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer
                .stamp("Computed weight matrices for sweetenings.\n")
                .unwrap()
                .as_str()
        );
        R::flush_console();
    }
    let seconds_in_initialization = timer.total_as_secs_f64();
    let mut period_timer = PeriodicTimer::new(1.0);
    let mut iteration_counter = 0;
    if use_neighbors {
        pool.install(|| {
            sweets
                .par_iter_mut()
                .for_each(|(z, loss, weight_matrices, _, _, _, _)| {
                    *loss = neighborhood_sweeten(
                        z,
                        &mut weight_matrices[..],
                        &views[..],
                        n_items,
                        a,
                        &pool,
                        max_seconds,
                        &timer,
                    );
                })
        });
    } else {
        let n_iterations = if n_iterations == 0 {
            sweets
                .iter()
                .map(|x| x.0.nrows() * x.0.ncols())
                .max()
                .unwrap_or(0)
        } else {
            n_iterations
        };
        while iteration_counter < n_iterations && timer.total_as_secs_f64() < max_seconds {
            iteration_counter += 1;
            pool.install(|| {
                sweets.par_iter_mut().for_each(
                    |(z, loss, weight_matrices, _, n_accepts, when, rng)| {
                        let n_features = z.ncols();
                        let total_length = n_items * n_features;
                        let index = index_1d_to_2d(rng.gen_range(0..total_length), n_features);
                        flip_bit(z, weight_matrices, a, index, &views);
                        let new_loss = expected_loss_from_weight_matrices(weight_matrices, &pool);
                        if new_loss < *loss {
                            *n_accepts += 1;
                            *when = iteration_counter;
                            *loss = new_loss;
                        } else {
                            flip_bit(z, weight_matrices, a, index, &views);
                        }
                    },
                );
            });
            if !quiet || status_file.exists() {
                period_timer.maybe(iteration_counter == n_iterations, || {
                    if quiet && status_file.exists() {
                        interrupted |= rprint!(
                            "{}",
                            format!(
                                "*** {} exists, so forcing status display.\n",
                                status_file.display()
                            )
                            .as_str()
                        );
                        R::flush_console();
                    }
                    sweets.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
                    let best = sweets.first().unwrap();
                    interrupted |= rprint!(
                        "{}",
                        format!(
                        "\rIter. {}: Since iter. {}, E(loss) is {:.4} from #{} with {} accept{}.",
                        iteration_counter,
                        best.5,
                        best.1,
                        best.3 + 1,
                        best.4,
                        if best.4 == 1 { "" } else { "s" }
                    )
                        .as_str()
                    );
                    R::flush_console();
                });
            }
            if interrupted || R::check_user_interrupt() {
                rprint!("\nCaught user interrupt, so breaking out early.");
                R::flush_console();
                break;
            }
        }
    }
    if !quiet {
        rprint!("\n");
        R::flush_console();
    }
    let seconds_in_sweetening = timer.total_as_secs_f64() - seconds_in_initialization;
    if timer.echo() {
        rprint!(
            "{}",
            timer
                .stamp("Sweetened best initial estimates.\n")
                .unwrap()
                .as_str()
        );
        R::flush_console();
    }
    sweets.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    let (best_z, best_loss, _, sweeten_number, n_accepts, best_iteration, _) =
        sweets.swap_remove(0);
    if timer.echo() {
        rprint!(
            "{}",
            format!(
                "Best result is {} from sweetening estimate {} at iteration {} after {} accept{}.\n",
                best_loss,
                sweeten_number + 1,
                best_iteration + 1,
                n_accepts,
                if n_accepts == 1 { "" } else { "s" }
            )
            .as_str()
        );
        R::flush_console();
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
    let estimate = RMatrix::<f64>::new(n_items, columns_to_keep.len(), pc);
    let estimate_slice = estimate.slice_mut();
    columns_to_keep
        .iter()
        .enumerate()
        .for_each(|(j_new, j_old)| {
            matrix_copy_into_column(estimate_slice, n_items, j_new, best_z.column(*j_old).iter())
        });
    let names = &[
        "estimate",
        "expectedLoss",
        "iteration",
        "nIterations",
        "secondsInitialization",
        "secondsSweetening",
        "secondsTotal",
        "whichSweet",
    ];
    let list = RList::with_names(names, pc);
    list.set(0, estimate).stop();
    list.set(1, best_loss.to_r(pc)).stop();
    list.set(2, (best_iteration as i32).to_r(pc)).stop();
    list.set(3, ((iteration_counter + 1) as i32).to_r(pc))
        .stop();
    list.set(4, seconds_in_initialization.to_r(pc)).stop();
    list.set(5, seconds_in_sweetening.to_r(pc)).stop();
    list.set(7, ((sweeten_number + 1) as i32).to_r(pc)).stop();
    list.set(6, timer.total_as_secs_f64().to_r(pc)).stop();
    if timer.echo() {
        rprint!("{}", timer.stamp("Finalized results.\n").unwrap().as_str());
        R::flush_console();
    }
    list
}

#[roxido]
fn fangs_double_greedy(samples: &RList, max_seconds: f64, a: f64, n_cores: usize) {
    let timer = EchoTimer::new();
    let n_samples = samples.len();
    if n_samples < 1 {
        stop!("Number of samples must be at least one.");
    }
    let n_items = get(samples, 0).nrow();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores)
        .build()
        .unwrap();
    let mut max_n_features_observed = 0;
    let mut views = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let o = get(samples, i);
        if o.nrow() != n_items {
            stop!("All elements of 'samples' must have the same number of rows.");
        }
        let view = make_view(o);
        max_n_features_observed = max_n_features_observed.max(view.ncols());
        views.push(view)
    }
    let mut z = Array2::<f64>::zeros((n_items, max_n_features_observed));
    let mut weight_matrices = make_weight_matrices(z.view(), &views[..], a, &pool);
    let loss = neighborhood_sweeten(
        &mut z,
        &mut weight_matrices[..],
        &views[..],
        n_items,
        a,
        &pool,
        max_seconds,
        &timer,
    );
    let estimate = RMatrix::<f64>::new(n_items, z.ncols(), pc);
    let estimate_slice = estimate.slice_mut();
    let mut index = 0;
    for j in 0..z.ncols() {
        for i in 0..n_items {
            estimate_slice[index] = z[(i, j)];
            index += 1;
        }
    }
    let list = RList::with_names(&["estimate", "expectedLoss", "secondsTotal"], pc);
    list.set(0, estimate).stop();
    list.set(1, loss.to_r(pc)).stop();
    list.set(2, timer.total_as_secs_f64().to_r(pc)).stop();
    list
}

#[allow(clippy::too_many_arguments)]
fn neighborhood_sweeten(
    z: &mut Array2<f64>,
    weight_matrices: &mut [Array2<f64>],
    views: &[ArrayView2<f64>],
    n_items: usize,
    a: f64,
    pool: &ThreadPool,
    max_seconds: f64,
    timer: &EchoTimer,
) -> f64 {
    let mut outer_loss = expected_loss_from_weight_matrices(weight_matrices, pool);
    loop {
        if timer.echo() {
            println!("Current loss: {}", outer_loss);
        }
        if timer.total_as_secs_f64() >= max_seconds {
            break;
        }
        // Optimize within a given number of columns
        let mut best_candidate_loss = f64::INFINITY;
        let mut best_index = [0, 0];
        for i in 0..n_items {
            for j in 0..z.ncols() {
                let candidate_loss = expected_loss_from_weight_matrices_if_flip_bit(
                    z,
                    weight_matrices,
                    a,
                    [i, j],
                    views,
                    pool,
                );
                if candidate_loss < best_candidate_loss {
                    best_index = [i, j];
                    best_candidate_loss = candidate_loss;
                }
            }
        }
        if best_candidate_loss < outer_loss {
            flip_bit(z, weight_matrices, a, best_index, views);
            outer_loss = best_candidate_loss;
        } else {
            break;
        }
    }
    outer_loss
}

#[roxido]
fn draws(samples: &RList, a: f64, n_cores: usize, quiet: bool) {
    let mut timer = EchoTimer::new();
    let n_samples = samples.len();
    if n_samples < 1 {
        stop!("Number of samples must be at least one.");
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores)
        .build()
        .unwrap();
    let n_items = get(samples, 0).nrow();
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut interrupted = false;
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer
                .stamp(
                    format!(
                        "Parsed parameters and using {} threads.\n",
                        pool.current_num_threads()
                    )
                    .as_str(),
                )
                .unwrap()
                .as_str()
        );
        R::flush_console();
    }
    let mut views = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let o = get(samples, i);
        if o.nrow() != n_items {
            stop!("All elements of 'samples' must have the same number of rows.");
        }
        let view = make_view(get(samples, i));
        views.push(view)
    }
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer.stamp("Made data structures.\n").unwrap().as_str()
        );
        R::flush_console();
    }
    let selected_candidates_with_rngs: Vec<_> =
        rand::seq::index::sample(&mut rng, n_samples, n_samples)
            .into_iter()
            .map(|index| {
                let mut seed = [0_u8; 16];
                rng.fill_bytes(&mut seed);
                let new_rng = Pcg64Mcg::from_seed(seed);
                (views[index], new_rng)
            })
            .collect();
    if timer.echo() {
        interrupted |= rprint!(
            "{}",
            timer.stamp("Selected all candidates.\n").unwrap().as_str()
        );
        R::flush_console();
    }
    let selected_candidates_with_rngs: Vec<_> = pool.install(|| {
        selected_candidates_with_rngs
            .into_par_iter()
            .map(|(view, mut rng)| {
                let n_features = view.ncols();
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
        interrupted |= rprint!(
            "{}",
            timer
                .stamp("Reduced number of features for all candidates.\n")
                .unwrap()
                .as_str()
        );
        R::flush_console();
    }
    let mut candidates = Vec::with_capacity(selected_candidates_with_rngs.len());
    for (z, rng) in selected_candidates_with_rngs {
        if interrupted || R::check_user_interrupt() {
            stop!("Caught user interrupt before main loop, so aborting.");
        }
        let loss = expected_loss_from_samples(z.view(), &views, a, &pool);
        candidates.push((z, loss, rng));
    }
    candidates.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    candidates.truncate(1);
    let mut bests: Vec<_> = pool.install(|| {
        candidates
            .into_par_iter()
            .enumerate()
            .map(|(id, (z, loss, rng))| {
                let weight_matrices = make_weight_matrices(z.view(), &views, a, &pool);
                let n_accepts = 0;
                let when = 1;
                (z, weight_matrices, loss, id, n_accepts, when, rng)
            })
            .collect()
    });
    if !quiet {
        rprint!("\n");
        R::flush_console();
    }
    if timer.echo() {
        rprint!("{}", timer.stamp("Sweetened bests.\n").unwrap().as_str());
        R::flush_console();
    }
    bests.sort_unstable_by(|x, y| x.2.partial_cmp(&y.2).unwrap());
    let (best_z, _, best_loss, candidate_number, n_accepts, best_iteration, _) =
        bests.swap_remove(0);
    if timer.echo() {
        rprint!(
            "{}",
            format!(
                "Best result is {} from candidate {} at iteration {} after {} accept{}.\n",
                best_loss,
                candidate_number + 1,
                best_iteration + 1,
                n_accepts,
                if n_accepts == 1 { "" } else { "s" }
            )
            .as_str()
        );
        R::flush_console();
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
    let estimate = RMatrix::<f64>::new(n_items, columns_to_keep.len(), pc);
    let estimate_slice = estimate.slice_mut();
    columns_to_keep
        .iter()
        .enumerate()
        .for_each(|(j_new, j_old)| {
            matrix_copy_into_column(estimate_slice, n_items, j_new, best_z.column(*j_old).iter())
        });
    let list = RList::with_names(&["estimate", "expectedLoss", "secondsTotal"], pc);
    list.set(0, estimate).stop();
    list.set(1, best_loss.to_r(pc)).stop();
    list.set(2, timer.total_as_secs_f64().to_r(pc)).stop();
    if timer.echo() {
        rprint!("{}", timer.stamp("Finalized results.\n").unwrap().as_str());
        R::flush_console();
    }
    list
}

#[roxido]
fn compute_expected_loss(z: &RMatrix<f64>, samples: &RList, a: f64, n_cores: usize) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores)
        .build()
        .unwrap();
    let n_samples = samples.len();
    let mut views = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        views.push(make_view(get(samples, i)));
    }
    expected_loss_from_samples(make_view(z), &views, a, &pool)
}

#[roxido]
fn compute_loss(z1: &RMatrix<f64>, z2: &RMatrix<f64>, a: f64) {
    if z1.nrow() == z2.nrow() {
        match make_weight_matrix(
            make_view(z1.as_f64().stop()),
            make_view(z2.as_f64().stop()),
            a,
        ) {
            Some(weight_matrix) => loss(&weight_matrix),
            None => 0.0,
        }
    } else {
        stop!("Inconsistent number of rows in 'Z1' and 'Z2'.");
    }
}

#[roxido]
fn compute_loss_permutations(z1: &RMatrix<f64>, z2: &RMatrix<f64>, a: f64) {
    use itertools::Itertools;
    let b = 2.0 - a;
    let loss = if z1.nrow() == z2.nrow() {
        let v1 = make_view(z1);
        let v2 = make_view(z2);
        let zero = Array1::zeros(v1.nrows());
        let zero_view = zero.view();
        let k = v1.ncols().max(v2.ncols());
        (0..k)
            .permutations(k)
            .map(|permutation| {
                let mut loss = 0.0;
                for (i, &j) in permutation.iter().enumerate().take(k) {
                    let c1 = if i >= v1.ncols() {
                        zero_view
                    } else {
                        v1.column(i)
                    };
                    let c2 = if j >= v2.ncols() {
                        zero_view
                    } else {
                        v2.column(j)
                    };
                    let aa = c1.iter().zip(c2); // std::iter::zip(c1, c2);
                    loss += aa.fold(0.0, |sum, (&x1, &x2)| {
                        sum + if x1 != x2 {
                            if x1 > x2 {
                                a
                            } else {
                                b
                            }
                        } else {
                            0.0
                        }
                    });
                }
                loss
            })
            .reduce(f64::min)
            .unwrap()
    } else {
        stop!("Inconsistent number of rows in 'Z1' and 'Z2'.");
    };
    loss
}

#[roxido]
fn compute_loss_augmented(z1: &RMatrix<f64>, z2: &RMatrix<f64>, a: f64) {
    let (loss, mut solution) = {
        match make_weight_matrix(make_view(z1), make_view(z2), a) {
            Some(weight_matrix) => {
                let solution = lapjv::lapjv(&weight_matrix).unwrap();
                (lapjv::cost(&weight_matrix, &solution.0), solution)
            }
            None => (0.0, (vec![], vec![])),
        }
    };
    for x in solution.0.iter_mut() {
        *x += 1;
    }
    for x in solution.1.iter_mut() {
        *x += 1;
    }
    let list = RList::with_names(&["loss", "permutation1", "permutation2"], pc);
    list.set(0, loss.to_r(pc)).stop();
    list.set(
        1,
        solution
            .1
            .iter()
            .map(|x| i32::try_from(*x).unwrap())
            .to_r(pc),
    )
    .stop();
    list.set(
        2,
        solution
            .0
            .iter()
            .map(|x| i32::try_from(*x).unwrap())
            .to_r(pc),
    )
    .stop();
    list
}

fn matrix_copy_into_column<'a>(
    slice: &mut [f64],
    nrow: usize,
    j: usize,
    iter: impl Iterator<Item = &'a f64>,
) {
    let subslice = &mut slice[(j * nrow)..((j + 1) * nrow)];
    subslice.iter_mut().zip(iter).for_each(|(x, y)| *x = *y);
}

fn make_view(z: &RMatrix<f64>) -> ArrayView2<'static, f64> {
    unsafe { ArrayView::from_shape_ptr((z.nrow(), z.ncol()).f(), z.slice().as_ptr()) }
}

fn make_weight_matrices(
    z: ArrayView2<f64>,
    samples: &[ArrayView2<f64>],
    a: f64,
    pool: &ThreadPool,
) -> Vec<Array2<f64>> {
    pool.install(|| {
        samples
            .par_iter()
            .map(|zz| make_weight_matrix(z, *zz, a).unwrap())
            .collect()
    })
}

fn index_1d_to_2d(index: usize, ncols: usize) -> [usize; 2] {
    [index / ncols, index % ncols]
}

#[allow(clippy::float_cmp)]
fn flip_bit(
    z: &mut Array2<f64>,
    matrices: &mut [Array2<f64>],
    a: f64,
    index: [usize; 2],
    samples: &[ArrayView2<f64>],
) {
    let old_bit = z[index];
    z[index] = if old_bit == 0.0 { 1.0 } else { 0.0 };
    let b = 2.0 - a;
    let [i0, i1] = index;
    samples.iter().zip(matrices.iter_mut()).for_each(|(zz, w)| {
        for i2 in 0..w.ncols() {
            let bit_in_sample = if i2 >= zz.ncols() { 0.0 } else { zz[[i0, i2]] };
            w[[i1, i2]] += if old_bit == 0.0 {
                if bit_in_sample == 0.0 {
                    a
                } else {
                    -b
                }
            } else if bit_in_sample == 0.0 {
                -a
            } else {
                b
            };
        }
    });
    /*
    // Sanity check, but commented out for speed.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    assert_ulps_eq!(
        expected_loss_from_samples(z.view(), samples, a, &pool),
        expected_loss_from_weight_matrices(matrices, &pool),
        max_ulps = 4
    );
    */
}

fn update_w(
    zz: &ArrayView2<f64>,
    w: &mut Array2<f64>,
    i0: usize,
    i1: usize,
    a: f64,
    b: f64,
    bit: f64,
) {
    for i2 in 0..w.ncols() {
        let bit_in_sample = if i2 >= zz.ncols() { 0.0 } else { zz[[i0, i2]] };
        w[[i1, i2]] += if bit == 0.0 {
            if bit_in_sample == 0.0 {
                a
            } else {
                -b
            }
        } else if bit_in_sample == 0.0 {
            -a
        } else {
            b
        };
    }
}

#[allow(clippy::float_cmp)]
fn expected_loss_from_weight_matrices_if_flip_bit(
    z: &Array2<f64>,
    matrices: &mut [Array2<f64>],
    a: f64,
    index: [usize; 2],
    samples: &[ArrayView2<f64>],
    pool: &ThreadPool,
) -> f64 {
    let old_bit = z[index];
    let new_bit = if old_bit == 0.0 { 1.0 } else { 0.0 };
    let b = 2.0 - a;
    let [i0, i1] = index;
    pool.install(|| {
        samples
            .par_iter()
            .zip(matrices.par_iter_mut())
            .fold(
                || 0.0,
                |acc: f64, (zz, w)| {
                    update_w(zz, w, i0, i1, a, b, old_bit);
                    let lss = loss(w);
                    update_w(zz, w, i0, i1, a, b, new_bit);
                    acc + lss
                },
            )
            .reduce(|| 0.0, |a, b| a + b)
    }) / (matrices.len() as f64)
}

#[allow(clippy::float_cmp)]
fn make_weight_matrix(y1: ArrayView2<f64>, y2: ArrayView2<f64>, a: f64) -> Option<Array2<f64>> {
    let b = 2.0 - a;
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
            vec.push(Zip::from(&x1).and(&x2).fold(0.0, |acc, &aa, &bb| {
                acc + if aa != bb {
                    if aa > bb {
                        a
                    } else {
                        b
                    }
                } else {
                    0.0
                }
            }));
        }
    }
    Some(unsafe { Array::from_shape_vec_unchecked((k, k), vec) })
}

fn expected_loss_from_samples(
    z: ArrayView2<f64>,
    samples: &[ArrayView2<f64>],
    a: f64,
    pool: &ThreadPool,
) -> f64 {
    pool.install(|| {
        samples
            .par_iter()
            .fold(
                || 0.0,
                |acc: f64, zz: &ArrayView2<f64>| {
                    acc + match make_weight_matrix(z, *zz, a) {
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
