#' Sweeten Feature Allocations
#'
#' An implementation of the sweetening phrase of the feature allocation greedy search algorithm is
#' provided.
#'
#' @param candidates An object of class \sQuote{list} containing features allocations to sweeten.  Each list element encodes one
#'   feature allocation as a binary matrix, with items in the rows and features
#'   in the columns.
#' @inheritParams fangs
#'
#' @return A list with the following elements:
#' \itemize{
#'   \item estimate - The feature allocation point estimate in binary matrix form.
#'   \item loss - The estimated expected FARO loss of the point estimate.
#'   \item iteration - The iteration number (out of \code{nIterations}) at which the point estimate was found.
#'   \item secondsSweetening - The elapsed time in the sweetening phrase.
#'   \item whichBest - The proposal number (out of \code{nBests}) from which the point estimate was found.
#' }
#'
#' @export
#' @examples
#' # To reduce load on CRAN testing servers, limit the number of iterations.
#' data(samplesFA)
#' # R_CARGO \dontrun{
#' # R_CARGO # Example disabled since Cargo was not found when installing from source package.
#' # R_CARGO # You can still run the example if you install Cargo. Hint: cargo::install().
#' sweeten(samplesFA[1:3], samplesFA, nIterations=100, nCores=2)
#' # R_CARGO }
#'
sweeten <- function(candidates, samples, nIterations=1000, nCores=0, quiet=FALSE) {
  candidates <- lapply(candidates, function(x) {storage.mode(x) <- "double"; x})
  samples <- lapply(samples, function(x) {storage.mode(x) <- "double"; x})
  .Call(.sweeten, candidates, samples, nIterations, nCores, quiet)
}
