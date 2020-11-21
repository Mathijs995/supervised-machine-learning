################################################################################
# COURSE: Supervised Machine Learning
# STUDENTS:
#   Yuchou Peng
#   Chao Liang
#   Mathijs de Jong
#   Eva Mynott
#
# DATE: 2020-11-16
################################################################################

################################################################################
# Initialize local settings
################################################################################

# Specify working directory
WEEK = 'Week 3'
setwd(paste0(BASE.DIR, '/Supervised Machine Learning/', WEEK, '/Assignment'))

# Specify options
options(scipen=999)
set.seed(42)

################################################################################
# Load dependencies
################################################################################

# Install and load packages
install.packages(setdiff(c('car', 'caret', 'devtools', 'e1071', 'Ecdat',
  'ISLR', 'plotrix'), installed.packages()))

# Install and load latest version of own package
devtools::install_github('Accelerytics/mlkit', upgrade='always', force=T)
library(mlkit)

# Install and load dsmle package
install.packages("dsmle_1.0-4.tar.gz", repos=NULL, type="source")
library(dsmle)

################################################################################
# Pre-process data
################################################################################

# Load data
df = Ecdat::Airline
formula = output ~ 0 + .

# Specify hyperparameter values to consider
params.length = 10
lambdas = c(0, 10 ^ seq(-5, 5, length.out=params.length - 1))
kernels = c('lin', 'pol', 'rbf')
consts = c(0, 0.5, 1, 2, 10, 100, 1000)
degrees = c(0.5, 1, 2, 10, 100, 1000)
scales = c(0.5, 1, 2, 10, 100, 1000)
params.list = list(
  'lambda' = lambdas,
  'kernel' = kernels,
  'const' = consts,
  'degree' = degrees,
  'scale' = scales
)

# Specify fold ids
N = nrow(df); n.folds = 5; fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]

################################################################################
# Comparison of different implementations
################################################################################

# Define metric functions
rmse = function(y.hat, y) sqrt(mean((y.hat - y) ^ 2))

# Hyperparameter tuning using estimator based on mlkit implementation
gscv.mlkit = mlkit::grid.search.cross.validation(formula,
  as.data.frame(scale(df)), mlkit::dual.ridge.lm, params.list, ind.metric=rmse,
  fold.id=fold.id, verbose=T, force=T)
print(gscv.mlkit)

# Hyperparameter tuning using estimator based on dsmle implementation
krr.function = function(formula, data, lambda, kernel, const, degree, scale) {
  y = data.matrix(data[, all.vars(formula)[1]]);
  X = stats::model.matrix(formula, data)
  kernel.type = ifelse(kernel == 'lin', 'linear', ifelse(kernel == 'rbf',
    'RBF', 'nonhompolynom'))
  return(dsmle::krr(y, X, lambda, kernel.type, degree, scale, center=F,
    scale=F))
}
gscv.dsmle = mlkit::grid.search.cross.validation(formula,
 as.data.frame(scale(df)), krr.function, params.list, ind.metric=rmse,
 fold.id=fold.id, verbose=T, force=T)
print(gscv.dsmle)
