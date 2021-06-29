compute_expected_loss <- function(Z, Zs) {
  # mean(sapply(Zs, function(Z2) compute_loss(Z,Z2)))
  .Kall(.compute_expected_loss, Z, Zs)
}
