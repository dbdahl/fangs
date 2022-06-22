#' Feature Allocation Neighborhood Greedy Search
#'
#' An implementation of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param samples An object of class \sQuote{list} containing posterior samples
#'   from a feature allocation distribution. Each list element encodes one
#'   feature allocation as a binary matrix, with items in the rows and features
#'   in the columns.
#' @param nInit The number of initial feature allocations to obtain using the
#'   alignment method.  For each initial feature, a baseline feature allocation
#'   is uniformly selected from the list provided in \code{samples}. Samples are
#'   aligned to the baseline, proportions are computed for each matrix element,
#'   and the initial feature allocation is obtained by thresholding according to
#'   \eqn{a/2}.
#' @param nSweet The number of feature allocations among \code{nInit} which are
#'   chosen (by lowest expected loss) to be optimized in the sweetening phase.
#' @param nIterations The number of iterations (i.e., proposed changes) to
#'   consider per initial estimate in the sweetening phase, although the actual
#'   number may be less due to the \code{maxSeconds} argument.
#' @param maxSeconds Stop the search and return the current best estimate once
#'   the elapsed time in the sweetening phase exceeds this value.
#' @param a A numeric scalar for the cost parameter of generalized Hamming
#'   distance used in FARO loss.  The other cost parameter, \eqn{b}, is equal
#'   to \eqn{2 - a}.
#' @param nCores The number of CPU cores to use, i.e., the number of
#'   simultaneous calculations at any given time. A value of zero indicates to
#'   use all cores on the system.
#' @param quiet If \code{TRUE}, intermediate status reporting is suppressed.
#'
#' @return A list with the following elements:
#' \itemize{
#'   \item estimate - The feature allocation point estimate in binary matrix form.
#'   \item expectedLoss - The estimated expected FARO loss of the point estimate.
#'   \item iteration - The iteration number (out of \code{nIterations}) at which the point estimate was found while sweetening.
#'   \item nIterations - The number of sweetening iterations performed.
#'   \item secondsInitialization - The elapsed time in the initialization phrase.
#'   \item secondsSweetening - The elapsed time in the sweetening phrase.
#'   \item secondsTotal - The total elapsed time.
#'   \item whichSweet - The proposal number (out of \code{nSweet}) from which the point estimate was found.
#'   \item nInit - The original supplied value of \code{nInit}.
#'   \item nSweet - The original supplied value of \code{nSweet}.
#'   \item a - The original supplied value of \code{a}.
#' }
#'
#' @export
#' @examples
#' # To reduce load on CRAN testing servers, limit the number of iterations.
#' data(samplesFA)
#' # R_CARGO \dontrun{
#' # R_CARGO # Example disabled since Cargo was not found when installing from source package.
#' # R_CARGO # You can still run the example if you install Cargo. Hint: cargo::install().
#' fangs(samplesFA, nIterations=100, nCores=2)
#' # R_CARGO }
#'
fangs <- function(samples, nInit=16, nSweet=4, nIterations=1000, maxSeconds=60, a=1.0, nCores=0, quiet=FALSE) {
  if ( a <= 0.0 || a >= 2.0 ) stop("'a' must be in (0,2).")
  samples <- lapply(samples, function(x) {storage.mode(x) <- "double"; x})
  result <- if ( Sys.getenv("FANGS_USE_DRAWS") == "TRUE" ) {
    warning("Using the 'draws' method.")
    .Call(.fangs_old, samples, nIterations,             nInit, nSweet, a, nCores, quiet)
  } else {
    .Call(.fangs,     samples, nIterations, maxSeconds, nInit, nSweet, a, nCores, quiet)
  }
  c(result, nInit=nInit, nSweet=nSweet, a=a)
}
