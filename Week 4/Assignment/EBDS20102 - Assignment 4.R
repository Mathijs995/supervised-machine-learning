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
install.packages(setdiff(c('devtools', 'elasticnet', 'kernlab', 'SVMMaj'),
  installed.packages()))
install.packages(
  'https://cran.r-project.org/src/contrib/Archive/regsel/regsel_0.2.tar.gz',
  repos = NULL,
  method = 'libcurl'
)

# Install and load latest version of own package
devtools::install_github('Accelerytics/mlkit', upgrade='always', force=T)

################################################################################
# Pre-process data
################################################################################

# Specify sample size
n.train = 400; n.test = 100

# Load data, drop rows containing NaN values and take sample n.obs clients
df = regsel::bank[complete.cases(regsel::bank), ]
df = df[sample(nrow(df), n.train + n.test), ]

# Reformat data
df = as.data.frame(cbind(y=ifelse(df$y, 1L, -1L),
  scale(df[, colnames(df) != 'y'])))

# Extract test and train data
test.ids = sample(nrow(df), n.test)
df.test = df[test.ids, ]; df.train = df[-test.ids, ]

# Define formula
formula = y ~ 0 + .


################################################################################
# Pre-process data
################################################################################

# Specify hyperparameter values to consider
params.length = 3
params.list = list(
  lambda = 2 ^ seq(5, -5, length.out = 20),
  kernel = c(kernlab::vanilladot, kernlab::polydot, kernlab::rbfdot),
  kernel.offset = 1,
  kernel.degree = c(0.5, 2, 3),
  kernel.scale =  1,
  kernel.sigma = c(0.5, 1, 2),
  hinge = c('absolute', 'quadratic', 'huber'),
  hinge.delta = c(0.5, 2)
)

# Specify fold ids
N = nrow(df.train); n.folds = 3; fold.id = ((1:N) %% n.folds + 1)[sample(N, N)]

################################################################################
# Hyperparameter tuning
################################################################################

# Define metric functions
mis.rate = function(y.hat, y) sum(y != ifelse(y.hat >= 0, 1L, -1L)) / length(y)

# Grid search 3-fold cross-validation
gscv.svm = grid.search.cross.validation(formula, df.train, SVMMaj::svmmaj,
  params.list, ind.metric=mis.rate, fold.id=fold.id, force=T, verbose=T,
  use.formula=F)

# Display results
print(gscv.svm)
