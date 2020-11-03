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

# Set working directory
BASE.DIR = '~/Google Drive/Tinbergen - MPhil'
WEEK = 'Week 2'
setwd(paste0(BASE.DIR, '/Supervised Machine Learning/', WEEK, '/Assignment'))

# Specify options
options(scipen=999)

################################################################################
# Load dependencies
################################################################################

# Install packages
if (!require('glmnet')) install.packages('glmnet', quiet=T)
if (!require('dplyr')) install.packages('dplyr', quiet=T)

# Load dependencies
source('elastic.net.lm.R')
source('grid.search.cross.validation.R')
source('ridge.lm.R')


################################################################################
# Generate results
################################################################################

# Load data
load('supermarket1996.rdata')
df = supermarket1996[sort(colnames(supermarket1996))]; rm(supermarket1996)
df = subset(df, select = -c(CITY, GROCCOUP_sum, SHPINDX, STORE, ZIP))

# Define dependent and independent variables
dep.var = 'GROCERY_sum'; y = df[dep.var]; X = df[colnames(df) != dep.var]

# OPTIONAL: Remove duplicate columns
while (any(duplicated(t(X)))) X = X[, -min(which(duplicated(t(X))))]

# Specify hyperparameter values to consider
params.list = list(
  'alpha' = 10 ^ seq(-5, 0, length.out = 100),
  'lambda' = 10 ^ seq(-5, 5, length.out = 100)
)

# Specify fold ids
N = nrow(X); n.folds = 5; fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]

# Define metric functions
mse = function(X, y, beta) mean(sum((y - X %*% beta) ^ 2))
root.mean = function(x) sqrt(mean(x))

# Hyperparameter tuning using grid search 5-fold cross-validation
gscv.res = grid.search.cross.validation(X, y, elastic.net.lm, params.list,
  n.folds=n.folds, ind.metric=mse, comb.metric=root.mean, fold.id=fold.id,
  verbose=T)

# Compare outcome with glmnet package
cv.fit = cv.glmnet(data.matrix(scale(X)), data.matrix(scale(y)), nfolds=n.folds,
  foldid=fold.id, gamma=gscv.res$alpha, lambda=params.list$lambda)

# Display optimal hyperparameters
cat('Optimal lambda: ', cv.fit$lambda.min, '\n')
cat('Optimal lambda: ', gscv.res$lambda, '\nOptimal alpha:  ',
  gscv.res$alpha, '\n')

# Estimate model on all data for optimal values of lambda and alpha
elastic.net.lm(X, y, lambda = gscv.res$lambda, alpha = gscv.res$alpha)

# Display glmnet results
plot(cv.fit)
coef(cv.fit, s = "lambda.min")