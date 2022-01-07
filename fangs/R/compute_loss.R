#' Compute the Loss for Two Feature Allocations
#'
#' The loss between two feature allocations is computed.
#'
#' @param Z1 something
#' @param Z2 something
#' @param augmented Should the permutations also be provided?
#'
#' @return The loss as a scalar value if `augmented = FALSE` else a list with additional information.
#'
#' @export
#' @examples
#' Z1 <- matrix(c(0,1,1,0,1,1,0,1,1,1,1,1), byrow=TRUE, nrow=6)
#' Z2 <- matrix(c(0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0), byrow=TRUE, nrow=6)
#' compute_loss(Z1,Z2)
#' x <- compute_loss(Z1,Z2,TRUE)
#' sum(cbind(Z1,0) != Z2)
#' sum(cbind(Z1,0)[,x$permutation1] != Z2)
#' sum(cbind(Z1,0) != Z2[,x$permutation2])
#'
compute_loss <- function(Z1, Z2, augmented=FALSE) {
  if ( ! is.numeric(Z1) ) stop("'Z1' should be numeric.")
  if ( ! is.numeric(Z2) ) stop("'Z2' should be numeric.")
  if ( ! is.matrix(Z1) ) stop("'Z1' should be a matrix.")
  if ( ! is.matrix(Z2) ) stop("'Z2' should be a matrix.")
  if ( nrow(Z1) != nrow(Z2) ) stop("'Z1' and 'Z2' must have the same number of rows.")
  if ( isTRUE(augmented) ) {
    .Call(.compute_loss_augmented,Z1,Z2)
  } else {
    .Call(.compute_loss,Z1,Z2)
  }
}
