################################################################################
# COURSE: Supervised Machine Learning
# STUDENTS:
#   Yuchou Peng
#   Chao Liang
#   Mathijs de Jong
#   Eva Mynott
#
# DATE: 2020-10-27
################################################################################


################################################################################
# Load dependencies
################################################################################

# Load packages
install.packages(c("Ecdat"), quiet=T)

# Set working directory and laod implementations
setwd(
  paste0(
    getwd(),
    "/Google Drive/Tinbergen - MPhil/Supervised Machine Learning/Week 1")
)

# Load dependencies
source('better.subset.lm.R')
  
################################################################################
# Compare estimators for air quality data
################################################################################

# Load data
df = Ecdat::Airq

# Extract dimensions of data
N = nrow(df)
P = ncol(df)

# Sanity check: Display first few rows of data
head(df)

# Define dependent and independent variables
y = df$airq
X = subset(df, select = -c(airq))

# Estimate OLS estimator using the MM algorithm
mm.lm(X, y, verbose=1)

# Estimate OLS estimator using standard OLS as provided in R
summary(lm('airq ~ 1 + .', as.data.frame(data.matrix(df))))

# Estimate OLS estimator using own version of standard OLS
analytical.lm(X, y)

# Estimate better subset OLS estimator
best.subset.lm(X, y, m.vals=c(1:ncol(X)), verbose=1000)