mm.lm = function(X, y, intercept=T, standardize=F, b.init=NULL, seed=NULL,
  tol=1e-6, verbose=0) {
  # Ordinary least squares estimator obtained using the MM algorithm. Note that
  # this implementation requires the model to have a constant.
  #
  # Inputs:
  #   X:            Table containing explanatory variables.
  #   y:            Column, vector or list containing dependent variables.
  #   intercept:    Indicator for whether or not to add an intercept. Default
  #                 is FALSE. If TRUE, standardize is ignored.
  #   standardize:  Indicator for whether or not to scale data. If intercept is
  #                 TRUE, this argument is ignored. Default is TRUE.
  #   b.init:       Optional initial values of betas. If NULL, a random beta is
  #                 sampled from the continuous uniform distribution on the
  #                 interval [0, 1] with m non-zero elements for each element m
  #                 in m.vals. Default is NULL.
  #   seed:         Optional seed, used for generating the random initial beta.
  #                 Ignored when an initial beta is provided. Default is NULL.
  #   tol:          Tolerated rounding error, default is 1e-6.
  #   verbose:      Integer indicating the step-size of printing iterations.
  #
  # Output:
  #   Beta estimate of the linear regression model obtained using the MM
  #   algorithm.
  
  
  # Import our own shared own functions
  source('../../base.R'); descale = function(beta) descale.beta(beta, x, y)
  
  # Add intercept or standarize data if necessary
  x = create_x(x, intercept, standardize)
  y = create_y(y, intercept, standardize)
  
  # Define constants
  Xt.X.scale = crossprod(X.scale)
  inv.lambda = 1 / eigen(Xt.X.scale)$values[1]
  inv.lambda.Xt.y.scale = inv.lambda * crossprod(X.scale, y.scale)
  k = 0
  
  # Choose some inital beta_0
  b.new = runif(ncol(X))
  
  # Compute RSS(beta_0)
  rss.new = rss(b.new)
  
  # Update beta_k until convergence
  while (TRUE) {
    
    # Update iteration
    k = k + 1
    
    # Replace old parameters by previous
    b.old = b.new; rss.old = rss.new
    
    # Apply MM-step
    b.new = b.old - inv.lambda * Xt.X.scale %*% b.old + inv.lambda.Xt.y.scale
    rss.new = rss(b.new)
    
    # Compute improvement
    delta = rss.old - rss.new
    
    # Display progress if verbose
    if (verbose & (k %% verbose == 0)) {
      cat(paste0('Iteration: ', k, '\nRSS_new: ', rss.new, '\nRSS_old: ',
                 rss.old, '\nRSS_old - RSS_new: ', delta, '\n\n'))
    }
    
    # Break if improvement smaller than tol; sufficient convergence
    if (delta / rss.old < tol) break
  }
  
  # Ensure information of last iteration is shown
  if (verbose & (k %% verbose != 0)) {
    cat(paste0('Iteration: ', k, '\nRSS_new: ', rss.new, '\nRSS_old: ', rss.old,
               '\nRSS_old - RSS_new: ', delta, '\n\n'))
  }
  
  # Scale estimator to original data
  b = descale(b.new)
  
  # Return beta estimate
  return(b)
}