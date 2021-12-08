compute_expected_loss <- function(Zs, Z, nCores=0) {
  # mean(sapply(Zs, function(Z2) compute_loss(Z2,Z)))
  .Call(.compute_expected_loss, Z, Zs, nCores)
}
