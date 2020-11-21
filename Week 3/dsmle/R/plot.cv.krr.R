#' Fit plot for cross validated kernel ridge regression
#' 
#' @description Makes a k-fold cross validated fit plot for kernel ridge regression for different lambda.
#' @usage plot.cv.krr(res, type = c("rmse", "profile", "edf"), ...)
#' @param res An object of the class \code{cv.krr}.
#' @param type plots the root mean squared error for \code{"rmse"}, provides a profile plot 
#' for each of out-of-sample predicted y value for \code{"profile"}, and a plot of effective 
#' degrees of freedom against \code{lambda} for \code{"edf"}.
#' @param ... additional arguments that are passed on to plot.
#' @examples data(longley)
#' X   <- as.matrix(longley[, c(1, 2, 4:7)])
#' y   <- longley[, 3]     # Unemployed
#' res <- cv.krr(y, X ,kernel.type = "RBF", lambda = 10^seq(-5, 5, length.out = 100))
#' plot(res)                    # for default rmse against lambda
#' plot(res, type = "profile")  # for profile lines per predicted object against lambda
#' plot(res, type = "edf")      # for edf against lambda
#' @export
plot.cv.krr <- function(res, type = c("rmse", "profile", "edf"), ...){
  type <- match.arg(type, c("rmse", "profile", "edf"), several.ok = FALSE)
  if (type == "rmse"){
    plot(res$lambda, res$rmse , type = "l", col = "blue", log = "x", xlab = expression(lambda), 
         ylab = "RMSE", main = "K-fold cross validated prediction", las = "1", ...)
    abline(v = res$lambda[which.min(res$rmse)], col = "grey")
  } else if (type == "profile"){
    n <- nrow(res$yhatu)
    line.col <- rainbow(n)
    y.lim <- c(min(res$yhat), max(res$yhatu)) 
    plot(res$lambda, res$yhatu[1,], type = "l", col = line.col[1], log = "x", xlab = expression(lambda), 
         ylab = "y-hat (out-of-sample)", main = "K-fold cross validated prediction", las = "1", 
         ylim = y.lim, ...)          
    for (i in 2:n) {
      lines(res$lambda, res$yhatu[i,], type = "l", col = line.col[i])
    }
    abline(v = res$lambda[which.min(res$rmse)], col = "grey")
  } else if (type == "edf"){
    ind.best <- which.min(res$rmse)
      if (is.matrix(res$edf)){
      n <- nrow(res$edf)
      line.col <- rainbow(n)
      y.lim <- c(min(res$edf), max(res$edf))   
      plot(res$lambda, res$edf[1,], type = "l", col = line.col[1], log = "x", xlab = expression(lambda), 
           ylab = "effective degrees of freedom", main = "Effective degrees of freedom", las = "1", 
           ylim = y.lim, ...)    
      for (i in 2:n){
        lines(res$lambda, res$edf[i,], type = "l", col = line.col[i])
      }      
      edf.min <- res$edf[n, ind.best]
    } else {
      plot(res$lambda, res$edf, type = "l", col = "blue", log = "x", xlab = expression(lambda), 
           ylab = "effective degrees of freedom", main = "Effective degrees of freedom", las = "1", ...) 
      edf.min <- res$edf[ind.best]
    }
    x <- c(res$lambda[ind.best], res$lambda[ind.best], min(res$lambda))
    y <- c(min(res$edf), edf.min, edf.min)
    lines(x, y, type = "l", col = "black")
  }
}


