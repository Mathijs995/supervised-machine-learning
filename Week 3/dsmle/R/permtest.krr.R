#' Permutation test for Kernel Ridge Regresssion
#' 
#' @export permtest.krr
#' @description Performs permutation test for kernel ridge regression.
#' @usage permtest.krr(y, X, k.folds = 10, lambda = 10^seq(-8, 8, length.out = 100), 
#' center = TRUE, scale = TRUE, ...)
#' @param y a numeric vector of responses.
#' @param X a matrix of covariates.
#' @param k.folds the number of folds in k-fold cross validation. Default is \code{k.folds = 10}.
#' @param lambda the vector with lambdas to be used. Default is \code{lambda = 10^seq(-8, 8, length.out = 100)}.
#' @param center logical to indicate to center the predictor variables in \code{X} to have column sums zero. 
#' Default is \code{TRUE}.
#' @param scale logical to indicate to scale the predictor variables in \code{X} to have column variance 1. 
#' Default is \code{TRUE}.
#' @param ... other parameters that may be passed on to \link{krr}. 
#' @return Returns a list of the class \code{cv.krr} with the following fields.
#' \item{\code{yhatu}}{a matrix the predicted out of sample values obtained by k-fold cross validation. 
#' The number of rows is the same as the number of rows of \code{X}  
#' and the number of columns is the same as the length of the vector \code{lambda}.}
#' \item{\code{lambda}}{the vector \code{lambda}.}
#' \item{\code{k.folds}}{number of folds.}
#' \item{\code{rmse}}{vector of root mean squared error comparing \code{y} with k-fold cross 
#' validated predictions \code{yhat}.}
#' \item{\code{lambda.min}}{the \code{lambda} with the smalles \code{rmse}.}
#' \item{\code{edf}}{effective degrees of freedom using \code{krr} with \code{lambda.min} on 
#' all observations.}
#' \item{\code{call}}{string with the call to this function.}
#' @examples data(longley)
#' X   <- as.matrix(longley[, c(1, 2, 4:7)])
#' y   <- longley[, 3]     # Unemployed
#' res <- permtest.krr(y, X , kernel.type = "RBF")

permtest.krr <- function(y, X, nPerm = 100, lambda = 1, 
                         center = TRUE, scale = TRUE, var.subset = NULL, ...){
  if (center) {
    meanX <- colMeans(X)
    X     <- X - outer(rep(1, nrow(X)), meanX)
  }
  if (scale) {
    stdX <- apply(X, 2, sd)
    X     <- X / outer(rep(1, nrow(X)), stdX)
  }
  if (is.null(var.subset)){
    var.subset <- 1:ncol(X)
  }
  res.all <- krr(y, X, center = FALSE, scale = FALSE, lambda = lambda, ...)
  rmse.perm   <- matrix(NA, length(var.subset), nPerm + 1)
  effect.size <- rep(NA, length(var.subset))
  p.value     <- rep(NA, length(var.subset))
  # Do the permutations
  for (j in var.subset){
    for (p in 1:nPerm){
      ind      <- sort(runif(nrow(X)), index.return = TRUE)$ix   # Find random permutation of values 1:n
      X.perm   <- X
      X.perm[, j] <- X[ind, j]
      res.perm <- krr(y, X.perm, center = FALSE, scale = FALSE, lambda = lambda, ...)
      rmse.perm[j, p] <- (sum(y - res.perm$yhat)^2)/length(y)^.5
    }
    res.minj <- krr(y, X[, -j], center = FALSE, scale = FALSE, lambda = lambda, ...)
    effect.size[j] <- sum(y - res.all$yhat)^2 / sum(y - res.minj$yhat)^2
    rmse.perm[j, nPerm + 1] <- (sum(y - res.all$yhat)^2)/length(y)^.5
    p.value[j] <- 1 - sum(as.numeric(rmse.perm[j,] < rmse.perm[j, nPerm + 1]))/(nPerm + 1)
  }  
  mat <- cbind(effect.size, p.value)
  colnames(mat) <- c("effect.size","p-value")
  rownames(mat) <- colnames(X)[var.subset] 
return(mat)
}



