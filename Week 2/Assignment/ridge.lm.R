ridge.lm = function(X, y, lambda, intercept=T, standardize=T, beta.tol=0,
  verbose=0) {
  # Implementation of the analytical solution for the linear regression model
  # with a Ridge penalty term.
  #
  # Inputs:
  #   X:          Table containing numerical explanatory variables.
  #   y:          Column containing a numerical dependent variable.
  #   lambda:     Penalty scaling constant.
  #   intercept:    Indicator for whether or not to add an intercept. Default
  #                 is TRUE. If FALSE, standardize is ignored.
  #   standardize:  Indicator for whether or not to scale data. Default is TRUE.
  #   beta.tol:   Rounding tolerance for beta, default is 0.
  #   verbose:    Integer indicating the step-size of printing progress updates,
  #               default is 0, that is, no progress updates.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   Ridge penalty term solved using the analytical solution
  
  # Add intercept and standarize data if necessary
  if (intercept & standardize) y = scale(y)
  if (intercept) if (standardize) X = cbind(1, scale(X)) else X = cbind(1, X)
  
  # Define constants
  N = nrow(X); P = ncol(X) - 1
  
  # Derive beta estimate
  b.new = solve(crossprod(X) + lambda * diag(P + 1), crossprod(X, y))
  
  # Force elements smaller than beta.tol to zero
  b.new[abs(b.new) < beta.tol] = 0
  
  # Generate summary statistics
  rss = sum((y - X %*% b.new) ^ 2); dof = sum(b.new != 0)
  r2 = 1 - rss / sum((y - mean(y)) ^ 2)
  
  # Return results
  return(list('coefficients' = b.new, 'alpha' = 0, 'lambda' = lambda,
    'loss'= rss / (2 * N) + lambda * sum(b.new[-1] ^ 2),
    'R^2' = r2, 'adjusted R^2' = 1 - (1 - r2) * (N - 1) / (N - dof)))
}