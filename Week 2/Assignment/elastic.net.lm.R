elastic.net.lm = function(X, y, lambda, alpha, standardize=T,
  beta.init=NULL, beta.tol=0, loss.tol=1e-6, eps=1e-6, seed=NULL, verbose=0) {
  # Implementation of the MM algorithm solver for a linear regression model
  # an elastic net penalty term.
  #
  # Inputs:
  #   X:            Table containing numerical explanatory variables.
  #   y:            Column containing a numerical dependent variable.
  #   lambda:       Penalty scaling constant.
  #   alpha:        Scalar of penalty for L1-norm of beta. Note that the scalar
  #                 assigned to the L2-norm is equal to (1 - alpha) / 2.
  #   standardize:  Indicator for whether or not to scale data. Default is TRUE.
  #   beta.init:    Optional initial value of betas. If NULL, a random beta is
  #                 sampled from the continuous uniform distribution on the
  #                 interval [0, 1]. Default is NULL.
  #   beta.tol:     Rounding tolerance for beta, default is 0.
  #   loss.tol:     Tolerated loss, default is 1e-6.
  #   eps:          Correcting value in computation of D matrix, default is
  #                 1e-6.
  #   seed:         Optional seed, used for generating the random initial beta.
  #                 Ignored when an initial beta is provided. Default is NULL.
  #   verbose:      Integer indicating the step-size of printing progress
  #                 updates, default is 0, that is, no progress updates.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   elastic net penalty term solved using the MM algorithm.
  
  # Estimate model with Ridge regression if possible
  if (alpha == 0) return(ridge.lm(X, y, lambda / 2, intercept, standardize,
    beta.tol, verbose))
  
  # Standarize data if necessary
  if (standardize){ y = scale(y); X = scale(X) }
  
  # Define constants
  N = nrow(X); P = ncol(X); double.N = 2 * N
  lambda.l1 = lambda * alpha; lambda.l2 = lambda * (1 - alpha)
  lambda.l2.I = diag(rep(lambda.l2, P))
  inv.N.Xt.X = crossprod(X) / N; inv.N.Xt.y = crossprod(X, y) / N
  
  # Define helper functions for printing progress and computing loss
  progress.line = function(var, val, var_width=21, width=15, nsmall=0)
    paste0(format(paste0(var, ':'), width=var_width), format(val, width=width,
    justify='right', nsmall=nsmall), '\n')
  progress.str = function(k, loss.new, loss.old, delta) paste0(
    progress.line('Iteration', k), progress.line('Loss.new', loss.new,
    nsmall=10), progress.line('Loss.old', loss.old, nsmall=10),
    progress.line('delta', delta, nsmall=10))
  elastic.net.loss = function(beta) sum((y - X %*% beta) ^ 2) / double.N +
    lambda.l1 * sum(abs(beta)) + lambda.l2 / 2 * sum(beta ^ 2)
  
  # Choose some inital beta_0 and compute initial loss
  if(is.null(beta.init)) b.new = runif(P + intercept) else b.new = beta.init
  loss.new = elastic.net.loss(b.new)
  
  # Update beta_k until convergence
  iter = 0; while (TRUE) {
    # Update iteration and replace old parameters by previous
    iter = iter + 1; loss.old = loss.new; b.old = b.new
    
    # Compute A matrix, D matrix, and update parameters
    b.old[abs(b.old) < eps] = eps; lambda.l1.D = diag(lambda.l1 / abs(b.old))
    A = inv.N.Xt.X + lambda.l1.D + lambda.l2.I; b.new = solve(A, inv.N.Xt.y)
    
    # Retrieve improvement
    loss.new = elastic.net.loss(b.new); delta = loss.old - loss.new
    
    # Display progress if verbose
    if (verbose & (iter %% verbose == 0))
      cat(progress.str(iter, loss.new, loss.old, delta), '\n\n')
    
    # Break if improvement smaller than tol, that is, sufficient convergence
    if (delta / loss.old < loss.tol) break
  }
  
  # Ensure information of last iteration is displayed
  if (verbose & (iter %% verbose != 0))
    cat(progress.str(iter, loss.new, loss.old, delta), '\n\n')
  
  # Force elements smaller than beta.tol to zero
  b.new[abs(b.new) < beta.tol] = 0
    
  # Generate summary statistics
  rss = sum((y - X %*% b.new) ^ 2); r2 = 1 - rss / sum((y - mean(y)) ^ 2)
  adj.r2 = 1 - (1 - r2) * (N - 1) / (N - sum(b.new != 0))
  
  # Return results
  return(list('coefficients'=b.new, 'alpha'=alpha, 'lambda'=lambda,
    'loss'=rss / double.N + lambda.l1 * sum(abs(b.new)) + lambda.l2 / 2 *
      sum(b.new ^ 2), 'R^2'=r2, 'adjusted R^2'=adj.r2))
}
