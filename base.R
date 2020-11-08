rss = function(beta, X, y) sum((y - X %*% beta) ^ 2)
r2 = function(beta, X, y) 1 - rss(beta, X, y) / sum((y - mean(y)) ^ 2)
adj.r2 = function(beta, X, y) {
  N = nrow(X); return(1 - (1 - r2(beta, X, y)) * (N - 1) / (N - sum(beta != 0)))
}

create_X = function(X, intercept, standardize) {
  X = data.matrix(X)
  if (intercept) return(cbind(1, X))
  if (standardize) X = scale(X);
  return(X)
}

create_y = function(y, intercept, standardize) {
  y = data.matrix(y)
  if (!intercept & standardize) y = scale(y)
  return(y)
}

descale.beta = function(beta, X, y) {
  beta = sd(y) * beta / apply(X, 2, sd)
  beta = c(mean(y) - sum(colMeans(X) * beta), beta)
  names(beta) = c('(Intercept)', colnames(X))
  return(beta)
}

progress.str = function(k, loss.new, loss.old, delta) paste0(
    progress.line('Iteration', k),
    progress.line('Loss.new', loss.new, nsmall=10),
    progress.line('Loss.old', loss.old, nsmall=10),
    progress.line('delta', delta, nsmall=10)
  )
progress.line = function(var, val, var_width=21, width=15, nsmall=0) paste0(
    format(paste0(var, ':'), width=var_width),
    format(val, width=width, justify='right', nsmall=nsmall),
    '\n'
  )

elastic.net.loss = function(beta, X, y, intercept, lambda.l1, lambda.l2)
  rss(beta, X, y) / (2 * nrow(X)) + 
    lambda.l1 * sum(abs(beta[(1 + intercept):ncol(X)])) +
    lambda.l2 * sum(beta[(1 + intercept):ncol(X)] ^ 2) / 2