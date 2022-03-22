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
#' @references Warr et al. (2021) Bayesian Analysis 1-37
#' (\href{https://projecteuclid.org/journals/bayesian-analysis/advance-publication/The-Attraction-Indian-Buffet-Distribution/10.1214/21-BA1279.full}{Bayesian Analysis})
#'
#'
"samplesFA"
