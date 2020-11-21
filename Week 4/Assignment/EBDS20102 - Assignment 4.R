################################################################################
# COURSE: Supervised Machine Learning
# STUDENTS:
#   Yuchou Peng
#   Chao Liang
#   Mathijs de Jong
#   Eva Mynott
#
# DATE: 2020-11-23
################################################################################

################################################################################
# Initialize local settings
################################################################################

# Specify working directory
WEEK = 'Week 4'
setwd(paste0(BASE.DIR, '/Supervised Machine Learning/', WEEK, '/Assignment'))

# Specify options
options(scipen=999)
set.seed(42)

################################################################################
# Load dependencies
################################################################################

# Install and load packages
install.packages(setdiff(c('devtools', 'elasticnet', 'SVMMaj'),
  installed.packages()))
install.packages(
  'https://cran.r-project.org/src/contrib/Archive/regsel/regsel_0.2.tar.gz',
  repos = NULL,
  method = 'libcurl'
)

# Install and load latest version of own package
devtools::install_github('Accelerytics/mlkit', upgrade='always', force=T)
library(mlkit)

################################################################################
# Pre-process data
################################################################################

# Load data
df = regsel::bank
formula = y ~ 0 + .

# Drop rows containing NaN values and take random sample of 1000 clients
df = df[complete.cases(df), ]
df = df[sample(nrow(df), 1000), ]
df[colnames(df) != 'y'] = scale(df[colnames(df) != 'y'])

################################################################################
# Comparison of different implementations
################################################################################

res = SVMMaj::svmmaj(df[, colnames(df) != 'y'], df$y, hinge='absolute')
mlkit::svm.bin(formula, df, lambda=1, loss='abs', verbose=25,
  v.init=c(0, res$beta))

mlkit::svm.bin(formula, df, lambda=1, verbose=1)
SVMMaj::svmmaj(df[, colnames(df) != 'y'], df$y, scale='zscore',
  hinge='quadratic')

res = SVMMaj::svmmaj(df[, colnames(df) != 'y'], df$y, hinge='huber',
  hinge.delta=2)
mlkit::svm.bin(formula, df, lambda=1, loss='hub', huber.k=2, verbose=1,
  v.init=c(0, res$beta))
