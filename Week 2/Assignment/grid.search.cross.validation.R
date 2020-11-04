grid.search.cross.validation = function(X, y, estimator, params.list,
  n.folds=10, ind.metric, comb.metric, fold.id=NULL, verbose=F, ...) {
  # Implementation of hyperparameter tuning using grid search using K-fold
  # cross-validation for given data, a given estimator, and given metrics.
  #
  # Inputs:
  #   X:           Table containing numerical explanatory variables.
  #   y:           Column containing a numerical dependent variable.
  #   estimator:   Estimator to use in hyperparameter tuning.
  #   n.folds:     Number of folds - default is 10. Although n.folds can be as
  #                large as the sample size (leave-one-out CV), it is not
  #                recommended for large datasets.
  #   fold.id:     An optional vector of values between 1 and n.fold
  #                identifying what fold each observation is in. If supplied,
  #                n.fold can be missing.
  #   ind.metric:  Metric for evaluating performance on fold.
  #   comb.metric: Combination function for individual metrics.
  #   verbose:     Indicator for displaying progress bar. Default is FALSE.
  #   ...:         Additional arguments that can be passed to the estimator.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   elastic net penalty term solved using the MM algorithm.
  
  # Define constants
  y = data.matrix(y); X = data.matrix(X); N = nrow(X); best.metric = Inf
  
  # Specify fold ids if not given and initialize metrics and ids vector
  if(is.null(fold.id)) fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]
  else n.folds =length(unique(fold.id))
  metrics = rep(NULL, n.folds); test.ids = matrix(nrow=n.folds, ncol=N)
  for (fold in 1:n.folds) test.ids[fold, ] = (fold.id == fold)
  
  # Create grid for cross validation search
  grid = expand.grid(params.list); n.combs = nrow(grid)
  
  # If verbose, initialize progress bar
  if (verbose) pb = dplyr::progress_estimated(n.combs * n.folds)
  
  # Apply grid search
  for (i in 1:n.combs) {
    
    # Apply cross validation
    for (fold in 1:n.folds) {
      
      # If verbose, update progress bar
      if (verbose) pb$tick()$print()
      
      # Create training and test set
      X.train = X[-test.ids[fold, ], ]; X.test = X[test.ids[fold, ], ]
      y.train = y[-test.ids[fold, ]]; y.test = y[test.ids[fold, ]]
      
      # Apply estimator to training data
      beta.train = do.call(estimator, c(list(X=X.train, y=y.train),
        as.list(grid[i, ]), list(...)))$coefficients
      
      # Store performance on test data
      metrics[fold] = ind.metric(X.test, y.test, beta.train)
    }
    
    # Combine performances on folds to overall performance
    metric = comb.metric(metrics)
    
    # Update best hyperparameters if improvement
    if (metric < best.metric) { best.metric = metric; best.params = grid[i, ]}
  }
  
  # Return best hyperparameters
  if (length(params.list) > 1) return(c(best.params)); return(best.params)
}
