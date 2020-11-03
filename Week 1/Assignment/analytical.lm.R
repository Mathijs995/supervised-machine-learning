analytical.lm = function(X, y, intercept=T) {
  # Ordinary linear regression estimator obtained using the analytical solution.
  #
  # Inputs:
  #   X:          Table containing explanatory variables.
  #   y:          Column, vector or list containing dependent variables.
  #   intercept:  Indicates whether to add an intercept to the model, default is
  #               TRUE.
  #
  # Output:
  #   List containing the beta estimates, the p-values for the test of
  #   individual significance, the in-sample residuals, the residual sum of
  #   sum of squares, and the standard errors of the parameter estimates, and
  #   the t-values for the test of individual significance.
  
  # Transform data to numeric
  y = data.matrix(y)
  X = data.matrix(X)
  
  # Add intercept if required
  if (intercept) X = cbind("(Intercept)" = 1, X)
  
  # Construct constants
  inv.Xt.X = solve(crossprod(X))
  N = nrow(X)
  
  # Compute OLS estimator
  b = inv.Xt.X %*% crossprod(X, y)
  
  # Compute standard errors
  e = y - X %*% b
  
  # Compute residual sum of squares
  rss = sum(e ^ 2)
  
  # Compute standard errors of beta
  cov.b = inv.Xt.X * rss / N
  std.b = sqrt(diag(cov.b))
  
  # Compute t-values
  t.b = b / std.b
  
  # Return regression output
  return(list(
    "coefficients" = b,
    "covariance" = cov.b,
    "p-values" = 2 * pnorm(-abs(t.b)),
    "residuals" = e,
    "rss" = rss,
    "r2" = 1 - rss / sum((y - mean(y)) ^ 2),
    "standard errors" = std.b,
    "t-values" = t.b
  ))
}