#' Estimate the expected FARO Loss for a Feature Allocation
#'
#' A Monte Carlo estimate of the expected FARO loss is computed for a feature allocation given a set of posterior samples.
#'
#' @param samples An object of class 'list' containing posterior samples from a feature allocation distribution in binary matrix form.
#' @param Z A single feature allocation object in binary matrix form.
#' @param nCores The number of CPU cores to use, i.e., the number of
#'   simultaneous runs at any given time. A value of zero indicates to use all
#'   cores on the system.
#'
#' @return The estimated expected FARO loss as a scalar value.
#'
#' @export
#' @examples
#' data(samplesFA)
#' Z <- matrix(sample(c(0,1), 60, replace=TRUE), byrow=TRUE, nrow=20)
#' # R_CARGO \dontrun{
#' # R_CARGO # Example disabled since Cargo was not found when installing from source package.
#' # R_CARGO # You can still run the example if you install Cargo. Hint: cargo::install().
#' compute_expected_loss(samplesFA, Z)
#' # R_CARGO }
#'
compute_expected_loss <- function(samples, Z, nCores=0) {
  # mean(sapply(Zs, function(Z2) compute_loss(Z2,Z)))
  .Call(.compute_expected_loss, Z, samples, nCores)
}
