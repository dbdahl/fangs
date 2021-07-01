#' Feature Allocation Neighborhood Greedy Search
#'
#' An implementation of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param Zs something
#' @param nProposals The number of proposals to consider.
#' @param maxNFeatures The maximum number of features that can be considered by
#'   the optimization algorithm, which has important implications for the
#'   interpretability of the resulting feature allocation. If the supplied value
#'   is zero, there is no constraint.
#' @param nSamples The number of feature allocations from the provided samples
#'   in \code{Zs} to use to seed the algorithm.
#' @param nBest Among \code{nSamples} feature allocations, how many should be
#'   sweetened when searching for the best estimate.
#' @param k At least one bit it flipped in each proposal.  This argument
#'   provides the size of the binomial distribution when deciding how many more
#'   bits should be proposed to be flipped.
#' @param probFlip At least one bit it flipped in each proposal.  This argument
#'   provides the probability of the binomial distribution when deciding how
#'   many more bits should be proposed to be flipped. WAHT IF NULL?
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
fangs2 <- function(Zs, nProposals=100, maxNFeatures=0, nSamples=length(Zs), nBest=1, k=0, probFlip=NULL, nCores=0) {
  Zs <- lapply(Zs, function(x) {storage.mode(x) <- "double"; x})
  result <- .Kall(.fangs, Zs, nProposals, maxNFeatures, nSamples, nBest, k, probFlip, nCores)
  c(result, nProposals=nProposals, maxNFeatures=maxNFeatures, nSamples=nSamples, nBest=nBest, k=k, probFlip=probFlip)
}
