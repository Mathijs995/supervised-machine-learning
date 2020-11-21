#' Cross validation for Kernel Ridge Regresssion
#' 
#' @export
#' @description Performs k-fold cross validation for kernel ridge regression.
#' @usage cv.krr(y, X, k.folds = 10, lambda = 10^seq(-8, 8, length.out = 100), 
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
#' res <- cv.krr(y, X ,kernel.type = "RBF", lambda = 10^seq(-5, 5, length.out = 100))

cv.krr <- function(y, X, k.folds = 10, lambda = 10^seq(-8, 8, length.out = 100), 
                   center = TRUE, scale = TRUE, ...){
  if (center) {
    meanX <- colMeans(X)
    X     <- X - outer(rep(1, nrow(X)), meanX)
  }
  if (scale) {
    stdX <- apply(X, 2, sd)
    X     <- X / outer(rep(1, nrow(X)), stdX)
  }
  
  ind  <- sort(runif(nrow(X)), index.return = TRUE)$ix   # Find random permutation of values 1:n
  fold <- rep(1:k.folds, ceiling(nrow(X)/k.folds))[ind]       # Vector of permuted fold numbers .
  
  yhatu <- matrix(NA, nrow = nrow(X), ncol = length(lambda))
  for (k in 1:k.folds) {
    ind <- fold == k     # Logical vector containing for the holdout sample
    res <- krr(y[!ind], X[!ind, , drop = FALSE], center = FALSE, scale = FALSE, lambda = lambda, ...)
    yhatu[ind,] <- predict(res, Xu = X[ind, , drop = FALSE])
  }  
  rmse <- (colSums((outer(y, rep(1, length(res$lambda))) - yhatu)^2)/length(y))^.5  
  res.all <- krr(y, X, center = FALSE, scale = FALSE, lambda = lambda, ...)
  result <- (list(yhatu = yhatu, lambda = lambda, k.folds = k.folds, rmse = rmse, edf = res.all$edf, 
                  lambda.min = lambda[which.min(rmse)], call = match.call())) 
  class(result) <- "cv.krr"
  return(result)
}


