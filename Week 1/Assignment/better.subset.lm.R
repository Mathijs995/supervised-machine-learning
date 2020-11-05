better.subset.lm = function(X, y, m.vals, b.init=NULL, seed=NULL, tol=1e-6,
  verbose=0) {
  # Implementation of the better subsets selection estimator by Xiong 2014.
  #
  # Inputs:
  #   X:          Table containing explanatory variables.
  #   y:          Column, vector or list containing dependent variables.
  #   m.vals:     Vector containing number of regressors to select.
  #   b.init:     Optional initial values of betas. If NULL, a random beta is
  #               sampled from the continuous uniform distribution on the
  #               interval [0, 1] with m non-zero elements for each element m
  #               in m.vals. Default is NULL.
  #   seed:       Optional seed, used for generating the random initial beta.
  #               Ignored when an initial beta is provided. Default is NULL.
  #   tol:        Tolerated rounding error, default is 1e-6.
  #   verbose:    Integer indicating the step-size of printing iterations.
  #
  # Output:
  #   Dataframe containing the results of the best beta parameter obtained
  #   through better subset regression.
  
  
  # Initiallize algorithm and define helper functions
  ## Ensure data is numerical and scale to speed up MM algorithm
  y = data.matrix(y); X = data.matrix(X); y.scale = scale(y); X.scale = scale(X)
  
  ## Define constants
  rss_tot = sum((y - mean(y)) ^ 2); Xt.X.scale = crossprod(X.scale)
  inv.lambda = 1 / eigen(Xt.X.scale)$values[1]; N = nrow(X); P = ncol(X)
  inv.lambda.Xt.y.scale = inv.lambda * crossprod(X.scale, y.scale)
  best.metric = Inf; if (!is.null(seed)) set.seed(seed)
  
  ## Define helper functions for computing statistics, descaling and printing
  rss = function(beta) sum((y.scale - X.scale %*% beta) ^ 2)
  r2 = function(beta) 1 - sum((y - cbind(1, X) %*% beta) ^ 2) / rss_tot
  adj.r2 = function(beta, m) 1 - (1 - r2(beta)) * (N - 1) / (N - m - 1)
  descale.b = function(b) {
    b = sd(y) * b / apply(X, 2, sd)
    b = c(mean(y) - sum(colMeans(X) * b), b)
    names(b) = c('(Intercept)', colnames(X))
    return(b)
  }
  progress.line = function(var, val, var_width=21, width=15, nsmall=0)
    paste0(format(paste0(var, ':'), width=var_width), format(val, width=width,
      justify='right', nsmall=nsmall), '\n')
  progress.str = function(m, k, rss.new, rss.old, delta) {
    paste0(progress.line('Number of regressors', m),
      progress.line('Iteration', k), progress.line('RSS.new', rss.new,
      nsmall=10), progress.line('RSS.old', rss.old, nsmall=10),
      progress.line('delta', delta, nsmall=10))
  }
  
  # Exercute algorithm for all possible number of explanatory variables
  for (m in m.vals) {
    
    # Specify beta_0 and compute RSS(beta_0)
    if (is.null(b.init)) { b.new = rep(0, P); b.new[sample(P, m)] = runif(m) }
    b.new = b.init[[m]]; rss.new = rss(b.new)
    
    # Update beta_k until convergence
    k = 0; while (TRUE) { k = k + 1;
      
      # Replace old parameters by previous
      b.old = b.new; rss.old = rss.new
      
      # Apply MM-step and update parameters
      u = b.old - inv.lambda * (Xt.X.scale %*% b.old) + inv.lambda.Xt.y.scale
      abs.u = abs(u); abs.u.max = sort(abs.u, decreasing=T)[m]
      b.new = ifelse(abs.u >= abs.u.max, u, 0); rss.new = rss(b.new)
      
      # Retrieve improvement
      delta = rss.old - rss.new
      
      # Display progress if verbose
      if (verbose & (k %% verbose == 0)) {
        cat(progress.str(m, k, rss.new, rss.old, delta), '\n\n')
      }
      
      # Break if improvement smaller than tol; sufficient convergence
      if (delta / rss.old < tol) break
    }
    
    # Ensure information of last iteration is displayed
    if (verbose & (k %% verbose != 0)) {
      cat(progress.str(m, k, rss.new, rss.old, delta), '\n\n')
    }
    
    # 'Descale' estimator to original data
    b = descale.b(b.new)
    
    # Update best estimate if better performance
    metric = adj.r2(b, m)
    if (metric < best.metric) { best.metric = metric; best.b = b }
  }
  
  # Generate summary statistics
  best.r2 = r2(best.b); ids = (best.b != 0); best.b = best.b[ids]
  X.full = cbind(1, X)[, ids]; best.rss = sum((y - (X.full %*% best.b)) ^ 2)
  dof = (N - sum(ids)); best.cov.b = solve(crossprod(X.full)) * best.rss / dof
  best.std.b = sqrt(diag(best.cov.b)); best.t.b = best.b / best.std.b
  
  # Return results best beta
  return(data.frame('coefficients'=best.b, 'standard error'=best.std.b,
    't-values'=best.t.b, 'p-values'=2 * pt(-abs(best.t.b), dof), 'R^2'=best.r2,
    'adjusted R^2'=c(best.metric, rep(NULL, length(best.b) - 1))
  ))
}
