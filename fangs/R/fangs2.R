#' Feature Allocation Neighborhood Greedy Search
#'
#' An implementation of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param Zs something
#' @param nIterations The number of iterations (i.e., proposed changes) to consider.
#' @param maxNFeatures The maximum number of features that can be considered by
#'   the optimization algorithm, which has important implications for the
#'   interpretability of the resulting feature allocation. If the supplied value
#'   is zero, there is no constraint.
#' @param nSamples The number of feature allocations from the provided samples
#'   in \code{Zs} to use to seed the algorithm.
#' @param nBest Among \code{nSamples} feature allocations, how many should be
#'   sweetened when searching for the best estimate.
#' @param nCores The number of CPU cores to use, i.e., the number of
#'   simultaneous runs at any given time. A value of zero indicates to use all
#'   cores on the system.
#'
#' @return A list of the following elements.  *This needs to be updated*:
#' \itemize{
#'   \item Z - The letters of the alphabet.
#'   \item bestDraw - A vector of numbers.
#'   \item loss - A vector of numbers.
#'   \item best - A vector of numbers.
#'   \item k - A vector of numbers.
#'   \item probFlip - A vector of numbers.
#'   \item iter - A vector of numbers.
#' }
#'
#' @export
#' @examples
#' 1 + 3
#'
fangs2 <- function(Zs, nIterations=100, maxNFeatures=0, nSamples=length(Zs), nBest=1, nCores=0) {
  Zs <- lapply(Zs, function(x) {storage.mode(x) <- "double"; x})
  result <- .Kall(.fangs, Zs, nIterations, maxNFeatures, nSamples, nBest, nCores)
  c(result, nIterations=nIterations, maxNFeatures=maxNFeatures, nSamples=nSamples, nBest=nBest)
}
