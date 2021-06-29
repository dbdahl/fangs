#' Compute the Loss for Two Feature Allocations
#'
#' The loss between two feature allocations is computed.
#'
#' @param Z1 something
#' @param Z2 something
#'
#' @return The loss as a scalar value.
#'
#' @export
#' @examples
#' Z1 <- matrix(c(0,0,0,0,0,0,0,0,1,0,0,1), byrow=TRUE, nrow=6)
#' Z2 <- matrix(c(0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0), byrow=TRUE, nrow=6)
#' compute_loss(Z1,Z2)
#'
compute_loss <- function(Z1, Z2) {
  if ( ! is.numeric(Z1) ) stop("'Z1' should be numeric.")
  if ( ! is.numeric(Z2) ) stop("'Z2' should be numeric.")
  if ( ! is.matrix(Z1) ) stop("'Z1' should be a matrix.")
  if ( ! is.matrix(Z2) ) stop("'Z2' should be a matrix.")
  if ( nrow(Z1) != nrow(Z2) ) stop("'Z1' and 'Z2' must have the same number of rows.")
  .Kall(.compute_loss,Z1,Z2)
}
