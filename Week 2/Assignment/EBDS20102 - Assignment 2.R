################################################################################
# COURSE: Supervised Machine Learning
# STUDENTS:
#   Yuchou Peng
#   Chao Liang
#   Mathijs de Jong
#   Eva Mynott
#
# DATE: 2020-11-08
################################################################################

################################################################################
# Initialize local settings
################################################################################

# Specify working directory
WEEK = 'Week 2'
setwd(paste0(BASE.DIR, 'Supervised Machine Learning/', WEEK, '/Assignment'))

# Specify options
options(scipen=999)
set.seed(42)

################################################################################
# Load dependencies
################################################################################

# Install and load packages
while (!require('glmnet')) install.packages('glmnet', quiet=T)
while (!require('SVMMaj')) install.packages('SVMMaj', quiet=T)
while (!require('xtable')) install.packages('xtable', quiet=T)

# Load dependencies
source('../../base.R')
source('elastic.net.lm.R')
source('grid.search.cross.validation.R')
source('ridge.lm.R')


################################################################################
# Pre-process data
################################################################################

# Load data
df = SVMMaj::supermarket1996; df = df[sort(colnames(df))]
df = subset(df, select=-c(CITY, GROCCOUP_sum, SHPINDX, STORE, ZIP))

# Define dependent and independent variables
dep.var = 'GROCERY_sum'; y = df[dep.var]; x = df[colnames(df) != dep.var]

# OPTIONAL: Remove duplicate columns
x = x[, -which(duplicated(t(x)))]

# Specify hyperparameter values to consider
params.length = 100; params.list = list(
  'alpha' = seq(0, 1, length.out=params.length),
  'lambda' = c(0, 10 ^ seq(-5, 5, length.out=params.length - 1))
)
heat.scale = c('alpha'='identity', 'lambda'='log10')

# Specify fold ids
N = nrow(x); n.folds = 5; fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]

################################################################################
# Hyperparameter tuning
################################################################################

# Define metric functions
lm.rmse = function(beta, x, y) sqrt(mean(sum((y - x %*% beta) ^ 2)))

# Hyperparameter tuning using estimator based on MM algorithm
gscv.own = grid.search.cross.validation(scale(x), scale(y), elastic.net.lm,
  params.list, ind.metric=lm.rmse, comb.metric=mean, fold.id=fold.id,
  verbose=T, force=T, heatmap=T, heat.scale=heat.scale, plot.coef=T,
  intercept=F, standardize=F)
progress.str(list(c('Alpha', gscv.own$params$alpha), c('Lambda',
  gscv.own$params$lambda)))

# Hyperparameter tuning using glmnet estimator
gscv.glm = grid.search.cross.validation(scale(x), scale(y), glmnet,
  params.list, ind.metric=lm.rmse, comb.metric=mean, fold.id=fold.id,
  verbose=T, force=T, heatmap=T, heat.scale=heat.scale, plot.coef=T,
  intercept=F, standardize=F)
progress.str(list(c('Alpha', gscv.glm$params$alpha), c('Lambda',
  gscv.glm$params$lambda)))


################################################################################
# Generate results
################################################################################

# Estimate on all data for optimal values of lambda and alpha
res.own.own = elastic.net.lm(scale(x), scale(y), alpha=gscv.own$params$alpha,
  lambda=gscv.own$params$lambda, standardize=F)
res.own.glm = elastic.net.lm(scale(x), scale(y), alpha=gscv.glm$params$alpha,
  lambda=gscv.glm$params$lambda, standardize=F)
res.glm.own = glmnet(scale(x), scale(y), alpha=gscv.own$params$alpha,
  lambda=gscv.own$params$lambda, standardize=F)
res.glm.glm = glmnet(scale(x), scale(y), alpha=gscv.glm$params$alpha,
  lambda=gscv.glm$params$lambda, standardize=F)

# Compare implementation of MM algorithm to glmnet outcome
res.table = data.frame(
  'own.own' = c(res.own.own$a0, res.own.own$beta),
  'own.glm' = c(res.own.glm$a0, res.own.glm$beta),
  'glm.own' = c(res.glm.own$a0, as.vector(res.glm.own$beta)),
  'glm.glm' = c(res.glm.glm$a0, as.vector(res.glm.glm$beta))
)
rownames(res.table) = paste0('\\texttt{', c('Intercept', colnames(x)), '}')

# Export table to LaTeX
xtable(res.table, digits=6)
