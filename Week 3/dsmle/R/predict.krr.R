#' Prediction method for Kernel Ridge Regression
#'
#' @export
#' @method predict krr
#' @description Does out-of-sample prediction for kernel ridge regression.
#' @usage krr (y, X, yu = NULL, Xu = NULL, lambda = 10^seq(-8, 8, length.out = 100),
#' kernel.type = c("RBF","linear","nonhompolynom"), kernel.degree = 2, kernel.RBF.sigma = 1,
#' center = TRUE, scale = TRUE)
#' @param res an \code{krr} object returned by \code{krr}.
#' @param Xu a matrix of out-of-sample covariates.
#' @return Returns a matrix the predicted out of sample values.
#' The number of rows is the same as the number of rows of \code{Xu}
#' and the number of columns is the same as the length of the vector \code{lambda}.
#' @examples data(longley)
#' ind <- as.logical(c(T,T,T,T,T,F,T,T,F,T,T,T,F,T,T,T))
#' X  <- as.matrix(longley[ind, c(1, 2, 4:7)])
#' y  <- longley[ind, 3] # Unemployed
#' Xu <- as.matrix(longley[!ind, c(1, 2, 4:7)])
#' yu <- longley[!ind, 3] # Unemployed
#' res <- krr(y, X, kernel.type = "RBF", lambda = 10^seq(-5, 5, length.out = 100))
#' yhatu <- predict(res, Xu)

predict.krr <- function(res, newdata, ...){
  Xu = newdata
  n <- nrow(res$X)
  if (!is.null(res$meanX)) Xu <- Xu - outer(rep(1, nrow(Xu)), res$meanX)
  if (!is.null(res$stdX))  Xu <- Xu / outer(rep(1, nrow(Xu)), res$stdX)
  # Set up out of sample Kernel matrix Ku
  if (res$kernel.type == "RBF"){
    Du <- outer(rowSums(Xu^2), rep(1, n)) + outer(rep(1, nrow(Xu)), rowSums(res$X^2)) - 2*Xu %*% t(res$X)
    Ku <- exp(-Du/res$kernel.RBF.sigma)
  } else if  (res$kernel.type == "nonhompolynom"){
    Ku <- (1 + Xu %*% t(res$X))^res$kernel.degree
  } else if  (res$kernel.type == "linear"){
      Ku <- Xu %*% t(res$X)
  }
  inv.eival <- as.numeric(res$ei$values > 1e-10) * (res$ei$values + as.numeric(res$ei$values <= 1e-10) )^-1
  yhatu <- res$const + ((Ku %*% res$ei$vectors) * outer(rep(1, nrow(Xu)), inv.eival))  %*%
                       t(res$ei$vectors) %*% res$yhat
  return(yhatu)
}

