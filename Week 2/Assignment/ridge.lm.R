ridge.lm = function(X, y, lambda, intercept=F, standardize=T, descale=T, 
  beta.tol=0, verbose=0) {
  # Implementation of the analytical solution for the linear regression model
  # with a Ridge penalty term.
  #
  # Inputs:
  #   X:            Table containing numerical explanatory variables.
  #   y:            Column containing a numerical dependent variable.
  #   lambda:       Penalty scaling constant.
  #   intercept:    Indicator for whether or not to add an intercept. Default
  #                 is FALSE. If TRUE, standardize is ignored.
  #   standardize:  Indicator for whether or not to scale data. Default is TRUE.
  #   beta.tol:     Rounding tolerance for beta, default is 0.
  #   verbose:      Integer indicating the step-size of printing progress
  #                 updates, default is 0, that is, no progress updates.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   Ridge penalty term solved using the analytical solution
  
  # Import our own shared own functions
  source('../../base.R'); descale = function(beta) descale.b(beta, X, y)
  
  # Add intercept or standarize data if necessary
  X = create_X(X, intercept, standardize)
  y = create_y(y, intercept, standardize)
  
  # Derive beta estimate
  b.new = solve(crossprod(X) + lambda * diag(ncol(X)), crossprod(X, y))
  
  # Set elements smaller than beta.tol to zero
  b.new[abs(b.new) < beta.tol] = 0
  
  # Return results
  return(list(
    'coefficients'=if (intercept | !standardize) b.new else descale(b.new),
    'alpha'=0, 'lambda'=lambda,
    'loss'=elastic.net.loss(b.new, X, y, intercept, lambda.l1=0, lambda),
    'R^2'=r2(b.new, X, y), 'adjusted R^2'=adj.r2(b.new, X, y)))
}