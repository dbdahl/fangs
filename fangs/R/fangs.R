#' Feature Allocation Neighborhood Greedy Search
#'
#' An implementation of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param samples An object of class 'list' containing posterior samples from a feature allocation distribution in binary matrix form.
#' @param nIterations The number of iterations (i.e., proposed changes) to consider per proposed estimate.
#' @param maxNFeatures The maximum number of features that can be considered by
#'   the optimization algorithm, which has important implications for the
#'   interpretability of the resulting feature allocation. If the supplied value
#'   is zero, there is no constraint.
#' @param nCandidates The number of feature allocations from the provided samples
#'   in \code{Z} for which the expected loss is estimated.
#' @param nBests The number of samples among \code{nCandidates} feature allocations which are chosen (by lowest expected loss) to be tuned in the sweetening phase.
#' @param nCores The number of CPU cores to use, i.e., the number of
#'   simultaneous runs at any given time. A value of zero indicates to use all
#'   cores on the system.
#' @param quiet If \code{TRUE}, intermediate status reporting is suppressed.
#'
#' @return A list of the following elements:
#' \itemize{
#'   \item estimate - The feature allocation estimate in binary matrix form.
#'   \item loss - The estimated expected loss of the estimate.
#'   \item iteration - The iteration number (out of \code{nIterations}) at which the estimate was found.
#'   \item seconds - The elapsed time of the entire algorithm.
#'   \item whichBest - The proposal number (out of \code{nBests}) from which the estimate was found.
#'   \item nBests - The original supplied value of \code{nBests}.
#'   \item nCandidates - The original supplied value of \code{nCandidates}.
#'   \item maxNFeatures - The original supplied value of \code{maxNFeatures}.
#' }
#'
#' @export
#' @examples
#' # To reduce load on CRAN servers, limit the number of iterations, but not necessary in practice.
#' data(samplesFA)
#' # R_CARGO \dontrun{
#' # R_CARGO # Example disabled since Cargo was not found when installing from source package.
#' # R_CARGO # You can still run the example if you install Cargo. Hint: cargo::install().
#' fangs(samplesFA, nIterations=100)
#' fangs(samplesFA, nIterations=50, nCandidates=length(samplesFA)/2, nBests=3, quiet=TRUE)
#' # R_CARGO }
#'
fangs <- function(samples, nIterations=1000, maxNFeatures=0, nCandidates=length(samples), nBests=1, nCores=0, quiet=FALSE) {
  samples <- lapply(samples, function(x) {storage.mode(x) <- "double"; x})
  result <- .Call(.fangs, samples, nIterations, maxNFeatures, nCandidates, nBests, nCores, quiet)
  c(result, nBests=nBests, nCandidates=nCandidates, maxNFeatures=maxNFeatures)
}
