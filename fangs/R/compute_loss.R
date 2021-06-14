#' @export
compute_loss <- function(Z1, Z2) {
  if ( ! is.numeric(Z1) ) stop("'Z1' should be numeric.")
  if ( ! is.numeric(Z2) ) stop("'Z2' should be numeric.")
  if ( ! is.matrix(Z1) ) stop("'Z1' should be a matrix.")
  if ( ! is.matrix(Z2) ) stop("'Z2' should be a matrix.")
  if ( nrow(Z1) != nrow(Z2) ) stop("'Z1' and 'Z2' must have the same number of rows.")
  .Kall(.compute_loss,Z1,Z2)
}
