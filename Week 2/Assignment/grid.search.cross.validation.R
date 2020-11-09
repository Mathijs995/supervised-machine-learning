grid.search.cross.validation = function(x, y, estimator, params.list,
  n.folds=10, ind.metric, comb.metric, fold.id=NULL, force=T, verbose=F,
  heatmap=F, heat.scale=NULL, plot.coef=F, ...) {
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
  #   heatmap:     Indicator for whether or not to display a heatmap of the
  #                gridsearch outcomes.
  #   heat.scale:  Optional argument to specify the scale used in the heatmap.
  #                Default is NULL.
  #   plot.coef:   Indicator for whether or not to display optimal coefficients.
  #   ...:         Additional arguments that can be passed to the estimator.
  #
  # Output:
  #   Dataframe containing the results of the linear regression model with
  #   elastic net penalty term solved using the MM algorithm.
  
  # Install and load dependencies
  while (!require('dplyr')) install.packages('dplyr', quiet=T)
  while (!require('ggplot2')) install.packages('ggplot2', quiet=T)
  while (!require('latex2exp')) install.packages('latex2exp')
  
  # Define constants
  y = data.matrix(y); x = data.matrix(x); N = nrow(x); b.metric = Inf
  
  # Specify fold ids if not given and initialize metrics and ids vector
  if(is.null(fold.id)) fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]
  else n.folds =length(unique(fold.id))
  metrics = rep(NULL, n.folds); test.ids = matrix(nrow=n.folds, ncol=N)
  for (fold in 1:n.folds) test.ids[fold, ] = (fold.id == fold)
  
  # Create grid for cross validation search
  grid = expand.grid(params.list)
  n.combs = nrow(grid); metric = rep(NULL, n.combs)
  
  # If verbose, initialize progress bar
  if (verbose) pb = dplyr::progress_estimated(n.combs * n.folds)
  
  # Apply grid search
  for (i in 1:n.combs) {
    
    # Apply cross validation and if verbose, update progress bar
    for (fold in 1:n.folds) { if (verbose) pb$tick()$print()
      
      # Compute individual metric on fold
      metrics[fold] = tryCatch(
        ind.metric(do.call(estimator, c(list(x=x[-test.ids[fold, ], ],
          y=y[-test.ids[fold, ]]), as.list(grid[i, ]), list(...)))$beta,
          x[test.ids[fold, ], ], y[test.ids[fold, ]]),
        error = function(e) {
          warning(paste('Failed for', paste(names(params.list), '=', grid[i, ],
            collapse=', ')))
          if (force & grepl('.*singular.*', e$message)) return(Inf)
          stop(e)
        }
      )
    }
    
    # Combine performances on folds to overall performance
    metric[i] = comb.metric(metrics)
  }
  
  # Extract optimal hyperparameters
  best.id = which(min(metric)); b.metric = metric[best.id]; b.params = grid[i, ]
  
  # Plot heatmaps if required
  combs = combn(names(params.list), 2); grid$metric = metric
  if (heatmap) for (i in 1:ncol(combs)) {
    col.x = combs[1, i]; col.y = combs[2, i]
    p = ggplot(data = grid, aes_string(x=col.x, y=col.y)) + geom_tile(aes(
      color=metric, fill=metric)) + ylab(TeX(paste0('$\\', col.y, '$'))) +
      xlab(TeX(paste0('$\\', col.x, '$')))
    if (!is.null(heat.scale))
      p = p + scale_y_continuous(trans=heat.scale[col.y]) + 
      scale_x_continuous(trans=heat.scale[col.x])
    print(p)
  }
  
  # Estimate best beta and if required, plot outcomes
  best.b = do.call(estimator, c(list(x=x, y=y), as.list(b.params),
    list(...)))$beta
  if (plot.coef) print(ggplot(data.frame(y=colnames(x), b=as.vector(best.b)),
    aes(b, y)) + geom_col() + ylab('Expl. variable') + xlab(TeX('$\\beta$')))
  
  # Reformat optimal hyperparameters
  if (length(params.list) > 1) b.params = c(b.params)
  
  return(list(
    'beta' = best.b,
    'params' = b.params,
    'metric' = b.metric
  ))
}