#' Fit plot for kernel ridge regression
#' 
#' @description Makes a fit plot for kernel ridge regression for different lambda.
#' @usage plot.krr(res, type = c( "insample", "out-of-sample"))
#' @param res An object of the class krr.
#' @param type Make a plot of the \code{insample} predictions (default) or 
#' \code{out-of-sample} predictions per lambda.
#' @return Returns a list of two items
#' \item{result}{matrix of with values of lambda and the root mean squared error (rmse).}
#' \item{lambda.min}{the values of lambda that had minimal root mean squared error x(rmse).}
#' @export plot.krr
#' @method plot krr
plot.krr <- function(res, type = c( "insample", "out-of-sample")){
  type <- match.arg(type, c("insample", "out-of-sample"), several.ok = FALSE)
  n <- length(res$y)
  if (type == "out-of-sample"){   
    rmse <- (colSums((outer(res$yu, rep(1, length(res$lambda))) - res$yhatu)^2)/n)^.5
    plot(res$lambda, rmse , type = "l", col = "blue", log = "x", xlab = expression(lambda), 
         ylab = "RMSE", main = "Out-of-sample prediction", las = "1")
  } else if (type == "insample"){
    rmse <- (colSums((outer(res$y, rep(1, length(res$lambda))) - res$yhat)^2)/n)^.5
    plot(res$lambda, rmse, type = "l", col = "blue",
         log = "x", xlab = expression(lambda), ylab = "RMSE", main = "In sample prediction", las = "1")
  }
  abline(v = res$lambda[which.min(rmse)], col = "grey")
  result <- cbind(res$lambda, rmse)
  colnames(result) <- c("Lambda", "RMSE")
  invisible(list(result = result, lambda.min = res$lambda[which.min(rmse)]))
}

