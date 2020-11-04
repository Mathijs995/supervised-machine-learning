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
source('../../init.R')
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
  'alpha' = seq(0, 1, length.out=100),
  'lambda' = 10 ^ seq(-5, 5, length.out=100)
)

# Specify fold ids
N = nrow(X); n.folds = 5; fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]

# Define metric functions
lm.mse = function(X, y, beta) mean(sum(y - cbind(1, X) %*% beta) ^ 2)
root.mean = function(x) sqrt(mean(x))

# Hyperparameter tuning using grid search 5-fold cross-validation
gscv.res = grid.search.cross.validation(scale(X), scale(y), elastic.net.lm,
  params.list, n.folds=n.folds, ind.metric=lm.mse, comb.metric=root.mean,
  fold.id=fold.id, verbose=T, intercept=T, standardize=F)

# Compare outcome with glmnet package
cv.fit = cv.glmnet(scale(X), scale(y), nfolds=n.folds, foldid=fold.id,
  gamma=params.list$alpha, lambda=params.list$lambda)

# Display optimal hyperparameters
cat('Optimal lambda: ', cv.fit$lambda.min, '\n')
cat('Optimal lambda: ', gscv.res$lambda, '\nOptimal alpha:  ',
  gscv.res$alpha, '\n')

# Estimate model on all data for optimal values of lambda and alpha
elastic.net.lm(scale(X), scale(y), alpha=gscv.res$alpha, lambda=gscv.res$lambda)
glmnet(scale(X), scale(y), alpha=gscv.res$alpha, lambda=gscv.res$lambda)

# Display glmnet results
plot(cv.fit)
coef(cv.fit, s = "lambda.min")