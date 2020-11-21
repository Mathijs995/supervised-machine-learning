#' Kernel Ridge Regresssion
#' 
#' @export
#' @description Performs kernel ridge regression.
#' @usage krr(y, X, lambda = 10^seq(-8, 8, length.out = 100), 
#' kernel.type = c("RBF","linear","nonhompolynom"), kernel.degree = 2, 
#' kernel.RBF.sigma = 1, center = TRUE, scale = TRUE)
#' @param y a numeric vector of responses.
#' @param X a matrix of covariates.
#' @param yu a numeric vector for out of sample predictions
#' @param Xu a matrix of out-of-sample covariates.
#' @param lambda the vector with lambdas to be used. Default is \code{lambda = 10^seq(-8, 8, length.out = 100)}.
#' @param kernel.type the type of kernel to be used. \code{RBF} defines the radial basis function 
#' or gaussian kernel (default), \code{linear} specifies the linear kernel, \code{nonhompolynom} specifies 
#' nonhomogeneous polynomial kernel. 
#' @param kernel.degree the degree of the \code{nonhompolynom} kernel.
#' @param kernel.RBF.sigma nonnegative hyperparamter for the \code{RBF} kernel.
#' @param center logical to indicate to center the predictor variables in \code{X} to have column sums zero. 
#' Default is \code{TRUE}.
#' @param scale logical to indicate to scale the predictor variables in \code{X} to have column variance 1. 
#' Default is \code{TRUE}.
#' @return Returns a list of the class \code{krr} with the following fields.
#' \item{\code{yhat}}{a matrix with the predicted values (in sample). Has the same length as \code{y} 
#' and the number of columns is the same as the length of the vector \code{lambda}.}
#' \item{\code{lambda}}{the vector \code{lambda}.}
#' \item{\code{edf}}{vector of the effective degrees of freedom corresponding to the  \code{lambda}.}
#' \item{\code{K}}{the kernel matrix computed using \code{X}).}
#' \item{\code{ei}}{list with the eigen \code{vectors} and \code{values} of \code{K}.}
#' \item{\code{meanX}}{vector with means of \code{X} if \code{center == TRUE} and 
#' \code{NULL} otherwise.}
#' \item{\code{stdX}}{vector with standard deviations of \code{X} if \code{scale == TRUE} and 
#' \code{NULL} otherwise.}
#' \item{\code{kernel.type}}{string with the specified kernel type.}
#' \item{\code{kernel.degree}}{degree of the kernel.}
#' @examples data(longley)
#' ind <- as.logical(c(T,T,T,T,T,F,T,T,F,T,T,T,F,T,T,T))
#' X  <- as.matrix(longley[ind, c(1, 2, 4:7)])
#' y  <- longley[ind, 3] # Unemployed
#' Xu <- as.matrix(longley[!ind, c(1, 2, 4:7)])
#' yu <- longley[!ind, 3] # Unemployed
#' res <- krr(y, X, kernel.type = "RBF", lambda = 10^seq(-5, 5, length.out = 100))
#' plot(res, type = "insample")

krr <- function(y, X, lambda = 10^seq(-8, 8, length.out = 100), 
                kernel.type = c("RBF","linear","nonhompolynom"), kernel.degree = 2, kernel.RBF.sigma = 1,
                center = TRUE, scale = TRUE){
  n     <- nrow(X)
  meanX <- NULL
  stdX  <- NULL
  if (center) {
    meanX <- colMeans(X)
    X     <- X - outer(rep(1,n),meanX)
  }
  if (scale) {
    stdX <- apply(X, 2, sd)
    X     <- X / outer(rep(1, n), stdX)
  }
  
  # Kernel ridge regression
  kernel.type <- match.arg(kernel.type, c("RBF","linear","nonhompolynom"), several.ok = FALSE)
  if (kernel.type == "RBF"){
    kernel.RBF.sigma <- kernel.RBF.sigma * ncol(X)
    K  <- exp(-as.matrix(dist(X)^2/kernel.RBF.sigma))
  } else if  (kernel.type == "nonhompolynom"){
    d  <- kernel.degree
    K  <- (1 + X %*% t(X))^d
  } else if  (kernel.type == "linear"){
    K  <- X %*% t(X)
  } 
  # Double center K
  J <- diag(n) - 1/n
  K <- J %*% K %*% J
  ei <- eigen(K, symmetric = TRUE)
  inv.eival <- as.numeric(ei$values > 1e-10) * (ei$values + as.numeric(ei$values <= 1e-10) )^-1
  shrink.mat <- outer(ei$values, rep(1, length(lambda))) / (outer(ei$values, rep(1, length(lambda))) + outer(rep(1,nrow(X)), lambda))
  a <- ei$vectors %*% (shrink.mat * outer(as.vector(t(ei$vectors) %*% y), rep(1, length(lambda))) ) 
  const <- mean(y)
  yhat  <- a + const
  edf   <- colSums(shrink.mat)
  #edf   <- apply(shrink.mat, 2, cumsum)
  result <- (list(yhat = yhat, lambda = lambda, K = K, 
                  y = y, const = const, ei = ei, X = X,  meanX = meanX, stdX = stdX, 
                  kernel.type = kernel.type, kernel.RBF.sigma = kernel.RBF.sigma, 
                  kernel.degree = kernel.degree, edf = edf) ) 
  class(result) <- "krr"
  return(result)
}
