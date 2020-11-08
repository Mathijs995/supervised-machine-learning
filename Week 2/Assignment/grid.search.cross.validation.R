grid.search.cross.validation = function(x, y, estimator, params.list,
  n.folds=10, ind.metric, comb.metric, fold.id=NULL, force=T, verbose=F, ...) {
  # Implementation of hyperparameter tuning using grid search using K-fold
  # cross-validation for given data, a given estimator, and given metrics.
  #
  # Inputs:
  #   x:           Table containing numerical explanatory variables.
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
  #   force:       Indicator whether not to terminate when encountering errors.
  #                Default is T, that is, errors are skipped with warning.
  #   verbose:     Indicator for displaying progress bar. Default is FALSE.
  #   ...:         Additional arguments that can be passed to the estimator.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   elastic net penalty term solved using the MM algorithm.
  
  # Define constants
  y = data.matrix(y); x = data.matrix(x); N = nrow(x); best.metric = Inf
  
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
    
    # Apply cross validation and if verbose, update progress bar
    for (fold in 1:n.folds) { if (verbose) pb$tick()$print()
      
      # Create training and test set
      x.train = x[-test.ids[fold, ], ]; x.test = x[test.ids[fold, ], ]
      y.train = y[-test.ids[fold, ]]; y.test = y[test.ids[fold, ]]
      
      # Apply estimator to training data
      b = tryCatch(
        do.call(estimator, c(list(x=x.train, y=y.train), as.list(grid[i, ]),
          list(...)))$beta,
        error = function(e) {
          warning(paste('Failed for', paste(names(params.list), '=', grid[i, ],
            collapse=', ')))
          if (force & grepl('.*singular.*', e$message, ignore.case=T))
            return(NULL)
          stop(e)
        }
      )
      
      # Store performance on test data
      metrics[fold] = ifelse(is.null(b), Inf, ind.metric(b, x.test, y.test))
    }
    
    # Combine performances on folds to overall performance
    metric = comb.metric(metrics)
    
    # Update best hyperparameters if improvement
    if (metric < best.metric) { best.metric = metric; best.params = grid[i, ]}
  }
  
  # Return best hyperparameters
  if (length(params.list) > 1) return(c(best.params)); return(best.params)
}
