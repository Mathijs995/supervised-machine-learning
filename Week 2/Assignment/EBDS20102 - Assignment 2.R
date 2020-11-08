################################################################################
# COURSE: Supervised Machine Learning
# STUDENTS:
#   Yuchou Peng
#   Chao Liang
#   Mathijs de Jong
#   Eva Mynott
#
# DATE: 2020-11-03
################################################################################

################################################################################
# Initialize local settings
################################################################################

# Specify working directory
BASE.DIR = '~/Google Drive/Tinbergen - MPhil/'
WEEK = 'Week 2'
setwd(paste0(BASE.DIR, 'Supervised Machine Learning/', WEEK, '/Assignment'))

# Specify options
options(scipen=999)

################################################################################
# Load dependencies
################################################################################

# Install packages
if (!require('dplyr')) install.packages('dplyr', quiet=T)
if (!require('glmnet')) install.packages('glmnet', quiet=T)
if (!require('SVMMaj')) install.packages('SVMMaj', quiet=T)

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
while (any(duplicated(t(x)))) x = x[, -max(which(duplicated(t(x))))]

# Specify hyperparameter values to consider
params.list = list(
  'alpha' = seq(0, 1, length.out=100),
  'lambda' = 10 ^ seq(-5, 5, length.out=100)
)

# Specify fold ids
N = nrow(x); n.folds = 5; fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]

################################################################################
# Hyperparameter tuning
################################################################################

# Define metric functions
lm.mse.scale = function(beta, x, y) mean(sum((y - x %*% beta) ^ 2))
root.mean = function(x) sqrt(mean(x))

# Hyperparameter tuning using estimator based on MM algorithm
gscv.own = grid.search.cross.validation(scale(x), scale(y), elastic.net.lm,
  params.list, ind.metric=lm.mse.scale, comb.metric=root.mean,
  fold.id=fold.id, verbose=T, force=T, intercept=F, standardize=F)

# Display optimal hyperparameters according to own implementation
print('Hyperparameter tuning using own estimator and own tuner')
progress.str(list(c('Alpha', gscv.own$alpha), c('Lambda', gscv.own$lambda)))

# Hyperparameter tuning using glmnet estimator
gscv.glm = grid.search.cross.validation(scale(x), scale(y), glmnet,
  params.list, ind.metric=lm.mse.scale, comb.metric=root.mean,
  fold.id=fold.id, verbose=T, force=T, intercept=F, standardize=F)

# Display optimal hyperparameters according to glmnet
print('Hyperparameter tuning using glmnet and own tuner')
progress.str(list(c('Alpha', gscv.glm$alpha), c('Lambda', gscv.glm$lambda)))


################################################################################
# Generate results
################################################################################

# Estimate on all data for optimal values of lambda and alpha
res.own = elastic.net.lm(scale(x), scale(y), alpha=gscv.own$alpha,
  lambda=gscv.own$lambda, standardize=F)
res.glmnet = glmnet(scale(x), scale(y), alpha=gscv.glm$alpha,
  lambda=gscv.glm$lambda, standardize=F)

# Compare implementation of MM algorithm to glmnet outcome
data.frame(
  'own' = c(res.own$a0, res.own$beta),
  'glmnet' = c(res.glmnet$a0, as.vector(res.glmnet$beta))
)
