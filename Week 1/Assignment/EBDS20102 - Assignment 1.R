################################################################################
# COURSE: Supervised Machine Learning
# STUDENTS:
#   Yuchou Peng
#   Chao Liang
#   Mathijs de Jong
#   Eva Mynott
#
# DATE: 2020-11-02
################################################################################

################################################################################
# Initialize environment
################################################################################

# Specify working directory
source('../../init.R')
WEEK = 'Week 1'
setwd(paste0(BASE.DIR, '/Supervised Machine Learning/', WEEK, '/Assignment'))

# Specify optional options
set.seed(123)
options(scipen=999)

################################################################################
# Load dependencies
################################################################################

# Install packages
if (!require('Ecdat')) install.packages('Ecdat', quiet=T)

# Load dependencies
source('better.subset.lm.R')

################################################################################
# Generate results
################################################################################

# Load data
df = Ecdat::Airq

# Define dependent and independent variables
y = df$airq
X = subset(df, select = -c(airq))

# Transform yes/no variables to 1/0
X = apply(X, 2, function(x) {
  if (setequal(x, c('yes', 'no'))) return(ifelse(x == 'yes', 1L, 0L))
  return(as.numeric(x))
})

# Append transformed explanatory variables
transforms = list('sqrt'=sqrt, 'log'=log, 'square'=function(x) x ^ 2,
  'cube'=function(x) x ^ 3)
for (transform in names(transforms)) {
  X.cont = X[, apply(X, 2, function(x) !setequal(x, c(0, 1)))]
  colnames(X.cont) = paste(transform, colnames(X.cont), sep='_')
  X = cbind(X, transforms[[transform]](X.cont))
}

# Specify m.vals and initial betas
P = ncol(X); m.vals = c(1:P); b.init = list()
for (m in m.vals) {
  b.init[[m]] = rep(0, P); b.init[[m]][sample(P, m)] = runif(m)
}

# Estimate and show results of better subset regression
apply(better.subset.lm(X, y, m.vals=m.vals, verbose=1, b.init=b.init), 2,
  function(x) format(x, nsmall=10))
