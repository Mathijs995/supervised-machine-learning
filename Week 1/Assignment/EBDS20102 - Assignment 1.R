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
# Load dependencies
################################################################################

# Specify working directory
source('../../init.R')
WEEK = 'Week 1'
setwd(paste0(BASE.DIR, '/Supervised Machine Learning/', WEEK, '/Assignment'))


################################################################################
# Load dependencies
################################################################################

# Install packages
if (!require('Ecdat')) install.packages('Ecdat', quiet=T)

# Load dependencies
source('better.subset.lm.R')

# Specify options
options(scipen=999)


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

# Estimate and show results of better subset regression
apply(better.subset.lm(X, y, m.vals=c(1:ncol(X)), verbose=0), 2,
  function(x) format(x, nsmall=10))
