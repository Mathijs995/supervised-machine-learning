mm.lm = function(X, y, tol=1e-6, verbose=0) {
  # Ordinary least squares estimator obtained using the MM algorithm. Note that
  # this implementation requires the model to have a constant.
  #
  # Inputs:
  #   X:          Table containing explanatory variables.
  #   y:          Column, vector or list containing dependent variables.
  #   tol:        Tolerated rounding error, default is 1e-6.
  #   verbose:    Integer indicating the step-size of printing iterations.
  #
  # Output:
  #   Beta estimate of the linear regression model
  
  
  # Initiallize algorithm and define helper functions
  ## Ensure data is numerical
  y = data.matrix(y)
  X = data.matrix(X)
  
  ## Transform data to scaled numeric so MM algorithm can be applied
  y.scale = scale(y)
  X.scale = scale(X)
  
  ## Define constants
  Xt.X.scale = crossprod(X.scale)
  inv.lambda = 1 / eigen(Xt.X.scale)$values[1]
  inv.lambda.Xt.y.scale = inv.lambda * crossprod(X.scale, y.scale)
  k = 0
  
  ## Define helper function for computing RSS and descaling parameter
  rss = function(beta) sum((y.scale - X.scale %*% beta) ^ 2)
  descale.b = function(b) {
    b = sd(y) * b / apply(X, 2, sd)
    b = c(mean(y) - sum(colMeans(X) * b), b)
    names(b) = c("(Intercept)", colnames(X))
    return(b)
  }
  
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
  b = descale.b(b.new)
  
  # Return beta estimate
  return(b)
}