rescale <- function(res){
  # Rescale homals results to proper scaling
  n <- nrow(res$objscores)
  m <- length(res$catscores)
  res$objscores     <- (n * m)^.5 * res$objscores
  res$scoremat      <- (n * m)^.5 * res$scoremat
  res$catscores     <- lapply(res$catscores,     FUN = function(x) (n * m)^.5 * x)
  res$cat.centroids <- lapply(res$cat.centroids, FUN = function(x) (n * m)^.5 * x)
  res$low.rank      <- lapply(res$low.rank,      FUN = function(x) n^.5 * x)
  res$loadings      <- lapply(res$loadings,      FUN = function(x) m^.5 * x)
  res$discrim       <- lapply(res$discrim,       FUN = function(x) (n * m)^.5 * x)
  res$eigenvalues   <- n * res$eigenvalues 
  return(res)
}
