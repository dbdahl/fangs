#' Feature Allocation Neighborhood Greedy Search
#'
#' An implementation of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param Zs something
#' @param nBest something
#' @param k something
#' @param probFlip something
#' @param maxIter something
#' @param nCores The number of CPU cores to use, i.e., the number of
#'   simultaneous runs at any given time. A value of zero indicates to use all
#'   cores on the system.
#'
#' @return A list of the following elements:
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
fangs2 <- function(Zs, nSamples=length(Zs), nBest=1, k=0, probFlip=NULL, maxIter=100, nCores=0) {
  Zs <- lapply(Zs, function(x) {storage.mode(x) <- "double"; x})
  result <- .Kall(.fangs, Zs, nSamples, nBest, k, probFlip, maxIter, nCores)
  c(result, nBest=nBest, k=k, probFlip=probFlip, totalIter=maxIter)
}

