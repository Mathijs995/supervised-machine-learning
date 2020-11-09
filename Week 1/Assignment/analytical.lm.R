analytical.lm = function(X, y, intercept=T, standardize=F) {
  # Ordinary linear regression estimator obtained using the analytical solution.
  #
  # Inputs:
  #   x:            Table containing numerical explanatory variables.
  #   y:            Column containing a numerical dependent variable.
  #   lambda:       Penalty scaling constant.
  #   alpha:        Scalar of penalty for L1-norm of beta. Note that the scalar
  #                 assigned to the L2-norm is equal to (1 - alpha) / 2.
  #   intercept:    Indicator for whether or not to add an intercept. Default
  #                 is TRUE. If TRUE, standardize is ignored.
  #   standardize:  Indicator for whether or not to scale data. If intercept is
  #                 TRUE, this argument is ignored. Default is FALSE.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   estimated using the analytical solution.
  
  # Import our own shared own functions
  source('../../base.R'); descale = function(beta) descale.beta(beta, x, y)
  
  # Add intercept or standarize data if necessary
  x = create_x(x, intercept, standardize)
  y = create_y(y, intercept, standardize)
  
  # Construct constants
  N = nrow(X)
  
  # Compute OLS estimator
  b = solve(crossprod(X)) %*% crossprod(X, y)
  
  # Compute standard errors
  e = y - X %*% b
  
  # Compute residual sum of squares
  rss = rss(b, X, y)
  
  # Compute standard errors of beta
  cov.b = inv.Xt.X * rss / N
  std.b = sqrt(diag(cov.b))
  
  # Compute t-values
  t.b = b / std.b
  
  # Return regression output
  return(list(
    "adj.r2" = adj.r2(b, X, y),
    "coefficients" = b,
    "covariance" = cov.b,
    "p-values" = 2 * pnorm(-abs(t.b)),
    "residuals" = e,
    "rss" = rss,
    "r2" = r2(b, X, y),
    "standard errors" = std.b,
    "t-values" = t.b
  ))
}