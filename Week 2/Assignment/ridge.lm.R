ridge.lm = function(X, y, lambda, standardize=T, beta.tol=0, verbose=0) {
  # Implementation of the analytical solution for the linear regression model
  # with a Ridge penalty term.
  #
  # Inputs:
  #   X:          Table containing numerical explanatory variables.
  #   y:          Column containing a numerical dependent variable.
  #   lambda:     Penalty scaling constant.
  #   standardize:  Indicator for whether or not to scale data. Default is TRUE.
  #   beta.tol:   Rounding tolerance for beta, default is 0.
  #   verbose:    Integer indicating the step-size of printing progress updates,
  #               default is 0, that is, no progress updates.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   Ridge penalty term solved using the analytical solution
  
  # Ensure data is numerical and create scaled data if necessary
  y = data.matrix(y); X = data.matrix(X);
  y = data.matrix(y); if (standardize) y.scale = scale(y) else y.scale = y
  X = data.matrix(X); if (standardize) X.scale = scale(X) else X.scale = X
  
  # Define constants
  N = nrow(X); P = ncol(X)
  
  # Define helper function for descaling
  descale.beta = function(beta) {
    beta = sd(y) * beta / apply(X, 2, sd)
    beta = c(mean(y) - sum(colMeans(X) * beta), beta)
    names(beta) = c('(Intercept)', colnames(X))
    return(beta)
  }
  
  # Derive beta estimate
  b.new = solve(crossprod(X.scale) + lambda * diag(P)) %*%
    crossprod(X.scale, y.scale)
  
  # Force elements smaller than beta.tol to zero
  b.new[abs(b.new) < beta.tol] = 0
  
  # 'Descale' estimator to original data
  if (standardize) { b.new = descale.beta(b.new); X = cbind(1, X) }
  
  # Generate summary statistics
  e = y - X %*% b.new; rss = sum(e ^ 2); dof = sum(b.new != 0)
  r2 = 1 - rss / sum((y - mean(y)) ^ 2)
  
  # Return results
  return(list('coefficients' = b.new,
    'loss'= rss / (2 * N) + lambda * sum(b.new ^ 2),
    'R^2' = r2, 'adjusted R^2' = 1 - (1 - r2) * (N - 1) / (N - dof)))
}