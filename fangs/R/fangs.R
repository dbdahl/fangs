#' Feature Allocation Neighborhood Greedy Search
#'
#' An implementation of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param samples An object of class \sQuote{list} containing posterior samples
#'   from a feature allocation distribution. Each list element encodes one
#'   feature allocation as a binary matrix, with items in the rows and features
#'   in the columns.
#' @param nIterations The number of iterations (i.e., proposed changes) to
#'   consider per draw in the sweetening phase.
#' @param maxNFeatures The maximum number of features that can be considered by
#'   the optimization algorithm, which has important implications for the
#'   interpretability of the resulting feature allocation. If the supplied value
#'   is zero, there is no constraint.
#' @param nCandidates The number of feature allocations (from the list provided
#'   in \code{samples}) to randomly select and score.
#' @param nBests The number of feature allocations among \code{nCandidates}
#'   which are chosen (by lowest expected loss) to be optimized in the
#'   sweetening phase.
#' @param a Cost parameter, a numeric scalar.
#' @param nCores The number of CPU cores to use, i.e., the number of
#'   simultaneous calculations at any given time. A value of zero indicates to
#'   use all cores on the system.
#' @param quiet If \code{TRUE}, intermediate status reporting is suppressed.
#'
#' @return A list with the following elements:
#' \itemize{
#'   \item estimate - The feature allocation point estimate in binary matrix form.
#'   \item loss - The estimated expected FARO loss of the point estimate.
#'   \item iteration - The iteration number (out of \code{nIterations}) at which the point estimate was found.
#'   \item secondsInitialization - The elapsed time in the initialization phrase.
#'   \item secondsSweetening - The elapsed time in the sweetening phrase.
#'   \item whichBest - The proposal number (out of \code{nBests}) from which the point estimate was found.
#'   \item nBests - The original supplied value of \code{nBests}.
#'   \item nCandidates - The original supplied value of \code{nCandidates}.
#'   \item maxNFeatures - The original supplied value of \code{maxNFeatures}.
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
#' fangs(samplesFA, nIterations=50, nCandidates=length(samplesFA)/2, nBests=3, nCores=2, quiet=TRUE)
#' # R_CARGO }
#'
fangs <- function(samples, nIterations=1000, maxNFeatures=0, nCandidates=length(samples), nBests=4, a=1.0, nCores=0, quiet=FALSE) {
  samples <- lapply(samples, function(x) {storage.mode(x) <- "double"; x})
  result <- .Call(.fangs, samples, nIterations, maxNFeatures, nCandidates, nBests, a, nCores, quiet)
  c(result, nBests=nBests, nCandidates=nCandidates, maxNFeatures=maxNFeatures, a=a)
}
