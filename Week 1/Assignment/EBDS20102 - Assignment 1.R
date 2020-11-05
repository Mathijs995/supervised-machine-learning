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
transforms = list('sqrt'=sqrt, 'log'=log, 'square'=function(x) x ^ 2)
X.orig = X
for (transform in names(transforms)) {
  X.cont = X.orig[, apply(X.orig, 2, function(x) !setequal(x, c(0, 1)))]
  colnames(X.cont) = paste(transform, colnames(X.cont), sep='_')
  X = cbind(X, transforms[[transform]](X.cont))
}

# Specify m.vals and initial betas
runs = 1:100
m.max = ncol(X)
P = ncol(X); m.vals = c(1:m.max); b.init = list()
for (i in runs) {
  b.init[[i]] = list()
  for (m in m.vals)
    b.init[[i]][[m]] = rep(0, P); b.init[[i]][[m]][sample(P, m)] = runif(m)
}

# Estimate and show results of better subset regression
best.metric = Inf
for (i in runs) {
  res = tryCatch(
    better.subset.lm(X, y, m.vals=m.vals, verbose=0, b.init=b.init[[i]]),
    error = function(e) return(NULL)
  )
  if (is.null(res)) next
  adj.r2 = res$adjusted.R.2[1]
  if (adj.r2 < best.metric) { best.res = res; best.metric = adj.r2 }
}
print(best.res)