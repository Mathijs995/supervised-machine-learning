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
# Load dependencies
################################################################################

# Set working directory
BASE.DIR = '~/Google Drive/Tinbergen - MPhil'
WEEK = 'Week 2'
setwd(paste0(BASE.DIR, '/Supervised Machine Learning/', WEEK, '/Assignment'))


################################################################################
# Load dependencies
################################################################################

# Install packages
if (!require('glmnet')) install.packages('glmnet', quiet=T)

# Load dependencies
source('elastic.net.lm.R')
source('grid.search.cross.validation.R')

# Specify options
options(scipen=999)


################################################################################
# Generate results
################################################################################

# Load data
load('supermarket1996.rdata')
df = supermarket1996[sort(colnames(supermarket1996))]
df = subset(df, select = -c(CITY, GROCCOUP_sum, SHPINDX, STORE, ZIP))

# Define dependent and independent variables
dep.var = 'GROCERY_sum'
y = df[dep.var]
X = df[colnames(df) != dep.var]

# OPTIONAL: Remove duplicate columns
while (any(duplicated(t(X)))) X = X[, -min(which(duplicated(t(X))))]

# Specify hyperparameter values to consider
params.list = list(
  'alpha' = c(0, 0.1, 0.25, 0.5, 0.75, 0.9), #10 ^ seq(-3, 0, length.out = 100),
  'lambda' = 10 ^ seq(-5, 5, length.out = 100)
)

# Define metric functions
mse = function(X, y, beta) sum((y - X %*% beta) ^ 2) / nrow(X)
root.mean = function(x) sqrt(mean(x))

# Hyperparameter tuning using grid search 5-fold cross-validation
gscv.res = grid.search.cross.validation(X, y, elastic.net.lm, params.list,
  n.folds=5, ind.metric=mse, comb.metric=root.mean, verbose=T)
cat('Optimal lambda: ', gscv.res$lambda, '\nOptimal alpha:  ', gscv.res$alpha, '\n')

# Estimate model on all data for optimal values of lambda and alpha
elastic.net.lm(X, y, lambda = gscv.res$lambda, alpha = gscv.res$alpha)

# Compare outcome with glmnet package
cvfit = cv.glmnet(data.matrix(scale(X)), data.matrix(scale(y)), nfolds=5,
  gamma=params.list$alpha, lambda=params.list$lambda)
plot(cvfit)
cvfit$lambda.min
coef(cvfit, s = "lambda.min")
