################################################################################
# COURSE:     Supervised Machine Learning
# STUDENTS:   Mathijs de Jong
# DATE:       2020-12-23
################################################################################


################################################################################
# Initialize local settings
################################################################################

# Initialize working directory
BASE.DIR = '/Users/mathijs/Google Drive/Tinbergen - MPhil'
setwd(paste0(BASE.DIR, '/Supervised Machine Learning/Final assignment'))

# Specify printing settings and set seed
options(scipen=999)
set.seed(42)

# Clean up environment
rm(BASE.DIR)

################################################################################
# Load dependencies
################################################################################

# Install public packages
install.packages(setdiff(c('censReg', 'devtools', 'gbm', 'glmnet', 'lmtest',
  'progress', 'kernlab', 'KRLS', 'randomForest', 'rpart', 'sampleSelection',
  'sandwich', 'SVMMaj'), installed.packages()), verbose=F, quiet=T)

# Install and load latest version of own (mlkit) package
if ('mlkit' %in% installed.packages()) remove.packages('mlkit')
devtools::install_github('Accelerytics/mlkit', force=T, upgrade='always')


################################################################################
# Pre-process data
################################################################################

# Load and pre-process
df.raw = readxl::read_excel('dataset13.xlsx')
df.full = df.raw[df.raw$mail == 1, ]
for (i in nrow(df.full)) if (df.full[i, 'mailid'] != 7)
  df.full[i, 'lastresp'] = df.full[i - 1, 'resp']
df.full = df.full[sample(1:nrow(df.full), 1e4), ]

# Specify dependent and exogenous variables
full.form = formula(amount ~ 1 + lastresp + avresp + avamount + urblvl + hhsize
  + lowinc + highinc)
select.form = formula(resp ~ 1 + lastresp + avresp + urblvl + hhsize
  + lowinc + highinc)
amount.form = formula(amount ~ 1 + lastresp + avamount + urblvl + hhsize
  + lowinc + highinc)

df.full$amount = ifelse(df.full$amount > 0, log(df.full$amount), 0)
train.ids = caret::createDataPartition(df.full$resp, p=.7, list=F)
df = list(train = df.full[train.ids, ], test = df.full[-train.ids, ])

n.train = nrow(df$train)
x = list()
y = list()

for (i in c('train', 'test')) {
  y[[i]] = list(
    amount = log(df[[i]][df[[i]]$amount > 0, 'amount']),
    full = ifelse(df[[i]]$amount > 0, log(df[[i]]$amount), 0),
    resp = df[[i]]$resp
  )
  x[[i]] = list(
    amount = model.matrix(amount.form, data=df[[i]][df[[i]]$amount > 0, ]),
    full = model.matrix(full.form, data=df[[i]]),
    select = model.matrix(select.form, data=df[[i]])
  )
}

n.folds = 5

# Clean up environment
rm(i, df.raw, df.full, train.ids)


################################################################################
# Benchmark model - Tobit I
################################################################################

# Tobit I - MLE using censReg package
censReg.tobit1 = censReg::censReg(full.form, left=0, right=Inf, data=df$train)
print(summary(censReg.tobit1))

# Tobit I - MLE using own implementation
tobit1_log_likelihood = function(theta, y, x) {
  length.theta = length(theta)
  sigma = exp(theta[length.theta])
  x.beta = x %*% theta[1:(length.theta - 1)]
  return(sum(!y * pnorm(-x.beta / sigma, log.p=T) - (y > 0) * (0.5 * log(2 * pi)
    + log(sigma) + ((y - x.beta) ^ 2) / (2 * sigma ^ 2))))
}
ml.tobit1 = optim(
  par = censReg.tobit1$estimate, fn = function(theta) -tobit1_log_likelihood(
    theta, y$train$full, x$train$full)
)$par
print(ml.tobit1)


################################################################################
# Benchmark model - Tobit II
################################################################################

# Tobit II - Heckman Two-Step Procedure using sampleSelection package
two.step.tobit2.res = sampleSelection::selection(
  selection = select.form,
  outcome = amount.form,
  data = df$train,
  method='2step'
)
print(summary(two.step.tobit2.res))

# Tobit II - MLE using sampleSelection package
ml.tobit2.res = sampleSelection::selection(
  selection = select.form,
  outcome = amount.form,
  data = df$train,
  method='ml'
)
print(summary(ml.tobit2.res))

# Tobit II - Heckman Two-Step Procedure using own implementation
select.probit = glm(select.form, df$train, family=binomial(link='probit'))
summary(select.probit)
x.beta.select = x$train$select %*% select.probit$coefficients
inv.mills = dnorm(x.beta.select) / pnorm(x.beta.select)
ids = (df$train$amount != 0)
amount.res = lm(update(amount.form, ~ . + inv.mills), cbind(df$train,
  inv.mills)[ids, ])
print(lmtest::coeftest(amount.res, vcov=sandwich::vcovHC(amount.res,
  type='HC0')))
eta = amount.res$residuals
sigma12 = amount.res$coefficients[length(amount.res$coefficients)]
sigma22 = sum(eta ^ 2 + sigma12 ^ 2 * inv.mills[ids] * (x.beta.select[ids]
  + inv.mills[ids])) / (sum(ids) - 1)
sqrt.sigma22 = sqrt(sigma22)
rho = sigma12 / sqrt.sigma22
res.df = data.frame('estimate' = c(sqrt.sigma22, rho))
rownames(res.df) = c('sigma', 'rho')
print(res.df)

# Tobit II - MLE using own implementation
theta0 = c(sqrt.sigma22, rho, select.probit$coefficients,
  amount.res$coefficients[1:ncol(x$train$amount)])
names(theta0) = c('sigma', 'rho', paste0('x.select.', colnames(x$train$select)),
  paste0('x.amount.', colnames(x$train$amount)))

tobit2_log_likelihood = function(theta, y, x.select, x.amount) {
  sigma = theta[1]
  rho = theta[2]
  d = (y == 0)
  k.select = ncol(x.select)
  x.beta.select = x.select %*% theta[3:(2 + k.select)]
  x.beta.amount = x.amount %*% theta[(3 + k.select):length(theta)]
  return(sum(!d * pnorm(-x.beta.select, log.p=T) + d * pnorm(x.beta.select
    + rho / sigma * (y - x.beta.amount) / sqrt(1 - rho ^ 2), log.p=T)
    - 0.5 * log(2 * pi) - log(sigma) - ((y - x.beta.amount) ^ 2
    / (2 * sigma ^ 2))))
}

ml.tobit2 = optim(
  par = theta0,
  fn = function(theta) -tobit2_log_likelihood(theta, y$train$full,
    x$train$select, model.matrix(amount.form, data=df$train))
)$par
print(ml.tobit2)

# Clean up environment
rm(amount.res, eta, ids, inv.mills, res.df, rho, sigma12, sigma22, sqrt.sigma22,
   theta0, x.beta.select, tobit1_log_likelihood, tobit2_log_likelihood)

################################################################################
# Two-Part Models - Classifiers
################################################################################

logit.classifier = function(formula, data)
  glm(formula, family = binomial(link='logit'), data)
probit.classifier = function(formula, data)
  glm(formula, family = binomial(link='probit'), data)
svm.classifier = function(formula, data, lambda=1, kernel=kernlab::vanilladot,
  kernel.sigma=1, kernel.scale=1, kernel.degree=1, kernel.offset=1,
  hinge='quadratic', hinge.delta=1e-08)
  SVMMaj::svmmaj(
    X = model.matrix(formula, data),
    y = unlist(data[, all.vars(formula)[1]]),
    lambda = lambda,
    scale = 'zscore',
    kernel = kernel,
    kernel.sigma = kernel.sigma,
    kernel.scale = kernel.scale,
    kernel.degree = kernel.degree,
    kernel.offset = kernel.offset,
    hinge = hinge,
    hinge.delta = hinge.delta
  )
tree.classifier = function(formula, data, cp)
  rpart::rpart(
    formula = formula,
    data = data,
    method='class',
    control = rpart::rpart.control(cp=cp)
  )
forest.classifier = function(formula, data, ntree, mtry, importance)
  randomForest::randomForest(
    x = model.matrix(formula, data),
    y = factor(unlist(data[, all.vars(formula)[1]])),
    ntree = ntree,
    mtry = mtry,
    importance = importance
  )

classifiers = list(
  logit = list(
    name = 'logistic regression classifier',
    method = logit.classifier,
    params.list = NULL
  ),
  probit = list(
    name = 'probit regression classifier',
    method = probit.classifier,
    params.list = NULL
  ),
  svm.linear = list(
    name = 'linear SVM',
    method = svm.classifier,
    params.list = list(
      lambda = 2 ^ seq(5, -5, length.out = 10),
      kernel = c(kernlab::vanilladot),
      hinge = c('quadratic')
    )
  ),
  tree = list(
    name = 'classification decision tree',
    method = tree.classifier,
    params.list = list(
      cp = sapply(1:10, function(x) 10^-x)
    )
  ),
  bag.forest = list(
    name = 'bagged forest classification',
    method = forest.classifier,
    params.list = list(
      mtry = c(1:3),
      ntree = c(50, 100, 250, 500),
      importance = F
    )
  ),
  random.forest = list(
    name = 'random forest classification',
    method = forest.classifier,
    params.list = list(
      mtry = c(1:3),
      ntree = c(50, 100, 250, 500),
      importance = T
    )
  ),
  boosted.trees = list(
    name = 'boosted classification trees',
    method = gbm::gbm,
    params.list = list(
      n.trees = c(50, 100, 250, 500),
      shrinkage = 2 ^ seq(5, -5, length.out = 10),
      interaction.depth = c(1:5),
      distribution = c('bernoulli')
    )
  ),
  logit.boost = list(
    name = 'LogicBoost algorithm',
    method = mlkit::logit.boost,
    params.list = list(
      n.iter = 10 ^ seq(2, 3, 5)
    )
  )
)

# Define metric functions
miss.rate = function(y.hat, y) {
  if (is.matrix(y.hat))
    y.hat = apply(y.hat, 1, function(x) which.max(x) - 1)
    return(sum(y != y.hat) / length(y))
  if (is.factor(y.hat)) return(sum(y != y.hat) / length(y))
  return(sum(y != ifelse(y.hat > 0, 1, 0), y.hat)) / length(y)
}

# Grid search 3-fold cross-validation
for (classifier in names(classifiers)) {
  print(classifier)
  classifier.obj = classifiers[[classifier]]
  if (is.null(classifier.obj[['params.list']]))
    classifiers[[classifier]][['model']] = classifier.obj[['method']](
      select.form,
      df$train
    )
  else
    classifiers[[classifier]][['model']] = mlkit::grid.search.cross.validation(
      formula = select.form,
      data = df$train,
      estimator = classifier.obj[['method']],
      params.list = classifier.obj[['params.list']],
      ind.metric = miss.rate,
      n.folds = n.folds,
      force = T,
      verbose = T
    )
}


################################################################################
# Two-Part Models - Regression models
################################################################################

df.train.scaled = as.data.frame(scale(df$train[df$train$amount > 0, ]))

elastic.net.regression = function(formula, data, alpha, lambda)
  glmnet::glmnet(
    x = model.matrix(formula, data),
    y = unlist(data[, all.vars(formula)[1]]),
    lambda = lambda,
    alpha = alpha
  )

kernel.ridge.regression = function(formula, data, kernel, lambda, sigma)
  KRLS::krls(
    X = model.matrix(formula, data),
    y = unlist(data[, all.vars(formula)[1]]),
    whichkernel = kernel,
    lambda = lambda,
    sigma = sigma
  )

tree.regression = function(formula, data, cp)
  rpart::rpart(
    formula = formula,
    data = data,
    method='anova',
    control = rpart::rpart.control(cp=cp)
  )
forest.regression = function(formula, data, ntree, mtry, importance)
  randomForest::randomForest(
    x = model.matrix(formula, data),
    y = unlist(data[, all.vars(formula)[1]]),
    ntree = ntree,
    mtry = mtry,
    importance = importance
  )

regression.models = list(
  elastic.net.regression = list(
    name = 'Elastic Net regression',
    method = elastic.net.regression,
    params.list = list(
      alpha = seq(0, 1, length.out=11),
      lambda = c(0, 10 ^ seq(-5, 5, length.out=10))
    )
  ),
  tree = list(
    name = 'decision tree regression',
    method = tree.regression,
    params.list = list(
      cp = sapply(1:10, function(x) 10^-x)
    )
  ),
  bag.forest = list(
    name = 'bagged forest regression',
    method = forest.regression,
    params.list = list(
      mtry = c(1:3),
      ntree = c(50, 100, 250, 500),
      importance = F
    )
  ),
  random.forest = list(
    name = 'random forest regression',
    method = forest.regression,
    params.list = list(
      mtry = c(1:3),
      ntree = c(50, 100, 250, 500),
      importance = T
    )
  ),
  boosted.trees = list(
    name = 'boosted regression trees',
    method = gbm::gbm,
    params.list = list(
      n.trees = c(50, 100, 250, 500),
      shrinkage = 2 ^ seq(5, -5, length.out = 10),
      interaction.depth = c(1:5),
      distribution = c('gaussian', 'laplace')
    )
  )
)

# Define metric functions
rmse = function(y.hat, y) sqrt(sum((y - y.hat) ^ 2))

# Grid search 3-fold cross-validation
for (model in names(regression.models)) {
  model.obj = regression.models[[model]]
  if (is.null(model.obj[['params.list']]))
    regression.models[[model]][['model']] = model.obj[['method']](
      amount.form,
      df.train.scaled
    )
  else
    regression.models[[model]][['model']] = 
      mlkit::grid.search.cross.validation(
        formula = amount.form,
        data = df.train.scaled,
        estimator = model.obj[['method']],
        params.list = model.obj[['params.list']],
        ind.metric = rmse,
        n.folds = n.folds,
        force = T,
        verbose = T
      )
}


################################################################################
# Direct Models - Censored Models
################################################################################

mixed.models = list(
  tree = list(
    name = 'decision tree regression',
    method = tree.regression,
    params.list = list(
      cp = sapply(1:10, function(x) 10^-x)
    )
  ),
  bag.forest = list(
    name = 'bagged forest regression',
    method = forest.regression,
    params.list = list(
      mtry = c(1:3),
      ntree = c(50, 100, 250, 500),
      importance = F
    )
  ),
  random.forest = list(
    name = 'random forest regression',
    method = forest.regression,
    params.list = list(
      mtry = c(1:3),
      ntree = c(50, 100, 250, 500),
      importance = T
    )
  ),
  boosted.trees = list(
    name = 'boosted regression trees',
    method = gbm::gbm,
    params.list = list(
      n.trees = c(50, 100, 250, 500),
      shrinkage = 2 ^ seq(5, -5, length.out = 10),
      interaction.depth = c(1:5),
      distribution = c('gaussian', 'laplace')
    )
  )
)

# Define metric functions
rmse = function(y.hat, y) sqrt(sum((y - y.hat) ^ 2))

# Grid search 3-fold cross-validation
for (model in names(mixed.models)) {
  model.obj = mixed.models[[model]]
  if (is.null(model.obj[['params.list']]))
    mixed.models[[model]][['model']] = model.obj[['method']](
      amount.form,
      df$train
    )
  else
    mixed.models[[model]][['model']] = mlkit::grid.search.cross.validation(
      formula = amount.form,
      data = df$train,
      estimator = model.obj[['method']],
      params.list = model.obj[['params.list']],
      ind.metric = rmse,
      n.folds = n.folds,
      force = T,
      verbose = T
    )
}
