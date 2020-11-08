elastic.net.lm = function(X, y, lambda, alpha, intercept=F, standardize=T,
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
  #   intercept:    Indicator for whether or not to add an intercept. Default
  #                 is FALSE. If TRUE, standardize is ignored.
  #   standardize:  Indicator for whether or not to scale data. If intercept is
  #                 TRUE, this argument is ignored. Default is TRUE.
  #   beta.init:    Optional initial value of betas. If NULL, a random beta is
  #                 sampled from the continuous uniform distribution on the
  #                 interval [0, 1]. Default is NULL.
  #   beta.tol:     Rounding tolerance for beta. Default is 0.
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
  if (alpha == 0) return(ridge.lm(X, y, lambda, intercept, standardize,
    beta.tol, verbose))
  
  # Import our own shared own functions
  source('../../base.R'); descale = function(beta) descale.b(beta, X, y)
  
  # Add intercept or standarize data if necessary
  X = create_X(X, intercept, standardize)
  y = create_y(y, intercept, standardize)
  
  # Define constants
  N = nrow(X); P = ncol(X)
  lambda.l1 = lambda * alpha; lambda.l2 = lambda * (1 - alpha)
  lambda.l2.I = diag(rep(lambda.l2, P)); if (intercept) lambda.l2.I[1,1] = 0
  inv.N.Xt.X = crossprod(X) / N; inv.N.Xt.y = crossprod(X, y) / N
  
  # Define helper functions for computing specific expressions and loss
  lambda.l1.D = function(beta) {
    beta[abs(beta) < eps] = eps
    lambda.l1.D = diag(lambda.l1 / abs(beta))
    if (intercept) lambda.l1.D[1,1] = 0
    return(lambda.l1.D)
  }
  loss = function(beta)
    elastic.net.loss(beta, X, y, intercept, lambda.l1, lambda.l2)
  
  # Choose some inital beta_0 and compute initial loss
  b.new = beta.init; if(is.null(b.new)) b.new = runif(P)
  loss.new = loss(b.new)
  
  # Update iteration and replace old parameters by previous until convergence
  iter = 0; while (TRUE) { iter = iter + 1; loss.old = loss.new; b.old = b.new
  
    # Update parameters, loss and delta
    b.new = solve(inv.N.Xt.X + lambda.l1.D(b.old) + lambda.l2.I, inv.N.Xt.y)
    loss.new = loss(b.new); delta = loss.old - loss.new
    
    # Display progress if verbose
    if (verbose & (iter %% verbose == 0))
      cat(progress.str(iter, loss.new, loss.old, delta), '\n\n')
    
    # Break if improvement smaller than tol, that is, sufficient convergence
    if (delta / loss.old < loss.tol) break
  }
  
  # Ensure information of last iteration is displayed
  if (verbose & (iter %% verbose))
    cat(progress.str(iter, loss.new, loss.old, delta), '\n\n')
  
  # Force elements smaller than beta.tol to zero
  b.new[abs(b.new) < beta.tol] = 0
  
  # Return results
  return(list(
    'coefficients'=if (intercept | !standardize) b.new else descale.b(b.new),
    'alpha'=alpha, 'lambda'=lambda, 'loss'=loss(b.new),
    'R^2'=r2(b.new, X, y), 'adjusted R^2'=adj.r2(b.new, X, y)))
}
