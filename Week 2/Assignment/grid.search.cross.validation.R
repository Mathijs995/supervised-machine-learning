grid.search.cross.validation = function(X, y, estimator, params.list, n.folds,
  ind.metric, comb.metric, verbose=F) {
  # Implementation of hyperparameter tuning using grid search using K-fold
  # cross-validation for given data, a given estimator, and given metrics.
  #
  # Inputs:
  #   X:           Table containing numerical explanatory variables.
  #   y:           Column containing a numerical dependent variable.
  #   estimator:   Estimator to use in hyperparameter tuning.
  #   n.folds:     Number of folds to use in cross-validation.
  #   ind.metric:  Metric for evaluating performance on fold.
  #   comb.metric: Combination function for individual metrics.
  #   verbose:     Indicator for displaying progress bar. Default is FALSE.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   elastic net penalty term solved using the MM algorithm.
  
  # Define constants
  y = data.matrix(y); X = data.matrix(X); N = nrow(X); best.metric = Inf
  metrics = rep(NULL, n.folds); fold.size = ceiling(N / n.folds)
  
  # Shuffle data
  ids = sample(N, N); X = X[ids, ]; y = y[ids]
  
  # Create grid for cross validation search
  grid = expand.grid(params.list); n.combs = nrow(grid)
  
  # If verbose, initialize progress bar
  if (verbose) pb = dplyr::progress_estimated(n.combs * n.folds)
  
  # Apply grid search
  skips = 0
  for (i in 1:n.combs) {
    
    # Apply cross validation
    for (fold in 1:n.folds) {
      
      # If verbose, update progress bar
      if (verbose) pb$tick()$print()
      
      # Create training and test set
      test.ids = ((fold - 1) * fold.size + 1):(min(fold * fold.size, N))
      X.train = X[-test.ids, ]; X.test = X[test.ids, ]
      y.train = y[-test.ids]; y.test = y[test.ids]
      
      # Apply estimator to training data
      beta.train = do.call(estimator, c(list(X=X.train, y=y.train),
        as.list(grid[i, ])))$coefficients
      
      # Store performance on test data
      metrics[fold] = ind.metric(X.test, y.test, beta.train)
    }
    
    # Combine performances on folds to overall performance
    metric = comb.metric(metrics)
    
    # Update best hyperparameters if improvement
    if (metric < best.metric) { best.metric = metric; best.params = grid[i, ]}
  }
  
  # Return best hyperparameters
  return(c(best.params))
}
