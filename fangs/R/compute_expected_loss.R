#' Estimate the expected FARO Loss for a Feature Allocation
#'
#' A Monte Carlo estimate of the expected FARO loss is computed for a feature allocation given a set of posterior samples.
#'
#' @inheritParams fangs
#' @param Z A feature allocation in binary matrix form, with items in
#'   the rows and features in the columns.
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
compute_expected_loss <- function(samples, Z, a=1.0, nCores=0) {
  # mean(sapply(Zs, function(Z2) compute_loss(Z2,Z,a)))
  .Call(.compute_expected_loss, Z, samples, a, nCores)
}
