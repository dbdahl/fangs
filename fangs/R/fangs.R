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
fangs_old <- function(Zs, nBest=1, k=1, prob1=-1, maxIter) {
  nItems <- nrow(Zs[[1]])
  nSamps <- length(Zs)

  allFeatures <- numeric(nSamps)
  allLosses <- numeric(length(Zs))
  allDensities <- numeric(length(Zs))

  for (i in 1:nSamps) {
    allLosses[i] <- compute_expected_loss(Zs[[i]], Zs)
    allFeatures[i] <- ncol(Zs[[i]])
    allDensities[i] <- mean(Zs[[i]])
  }

  meanDensity <- mean(allDensities)
  nFeaturesMax <- max(allFeatures)
  minIndices <- order(allLosses)[1:nBest]

  if (prob1 == -1) {
    prob1 <- meanDensity
  }

  bestZ <- NA
  bestLoss <- Inf

  for (i in 1:nBest) {
    curZ <- cbind(Zs[[minIndices[i]]], matrix(0, nrow = nItems, ncol = nFeaturesMax - ncol(Zs[[minIndices[i]]])))
    curLoss <- compute_expected_loss(curZ, Zs)

    iter <- 0
    while(iter < maxIter) {
      newZ <- curZ
      sampledIndices <- sample(1:length(curZ), k)
      if (k == 1) {
        newZ[sampledIndices] <- 1 - newZ[sampledIndices]
      }
      else {
        newValues <- sample(c(0,1), k, replace = TRUE, prob = c(1-prob1, prob1))
        # sample until different than current state
        while (identical(newZ[sampledIndices], newValues)) {
          newValues <- sample(c(0,1), k, replace = TRUE, prob = c(1-prob1, prob1))
        }
        newZ[sampledIndices] <- sample(c(0,1), k, replace = TRUE, prob = c(1-prob1, prob1))
      }
      newLoss <- compute_expected_loss(newZ, Zs)
      if (newLoss < curLoss) {
        curLoss <- newLoss
        curZ <- newZ
      }
      if (curLoss < bestLoss) {
        bestLoss <- curLoss
        bestZ <- curZ
      }
      iter <- iter + 1
    }
  }

  # Cut off any columns of all 0s
  if (0 %in% colSums(bestZ)) {
    columnSums <- colSums(bestZ)
    bestZ <- bestZ[,which(columnSums!=0)]
  }

  return(list(Z=bestZ, drawsAnswer=Zs[[minIndices[1]]], loss=bestLoss, nBest=nBest, k=k, prob1=prob1, totalIter=iter))
}
