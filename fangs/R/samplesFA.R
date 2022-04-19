#' The Attraction Indian Buffet Distribution
#'
#' Feature allocation samples were obtained from a latent feature model using the AIBD as a prior distribution.
#' The purpose of the model was to use pairwise distance information to identify and predict the presence of
#' Alzheimer's disease in patients.
#'
#' @format A list of length 100 where each list element is a feature allocation object (in binary matrix form).
#' These 100 feature allocation samples are a subset of the original 1000 samples obtained using MCMC in the
#' original simulation study described by Warr et al. (2021).
#'
#' @usage data(samplesFA)
#'
#' @references
#' R. L. Warr, D. B. Dahl, J. M. Meyer, A. Lui (2021),
#' The Attraction Indian Buffet Distribution, Bayesian Analysis Advance Publication 1 - 37,
#' \doi{10.1214/21-BA1279}.
#'
#'
"samplesFA"
