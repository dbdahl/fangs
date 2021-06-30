#' Feature Allocation Neighborhood Greedy Search
#'
#' An implementation of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param Zs something
#' @param nBest something
#' @param k something
#' @param prob1 something
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
#'   \item prob1 - A vector of numbers.
#'   \item iter - A vector of numbers.
#' }
#'
#' @export
#' @examples
#' 1 + 3
#'
fangs2 <- function(Zs, nBest=1, k=1, prob1=-1, maxIter, nCores=0) {
  Zs <- lapply(Zs, function(x) {storage.mode(x) <- "double"; x})
  result <- .Kall(.fangs, Zs, nBest, k, prob1, maxIter, nCores)
  c(result, nBest=nBest, k=k, prob1=prob1, totalIter=maxIter)
}

