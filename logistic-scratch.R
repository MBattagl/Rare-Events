library(ggplot2)
library(dplyr)

sigmoid <- function(z) {
  
  1 / (1 + exp(-z))
  
}

#cost function
cost <- function(theta, X, y) {
  
  m <- length(y) # number of training examples
  
  h <- sigmoid(X %*% theta)
  J <- (t(-y) %*% log(h) - t(1 - y) %*% log(1 - h)) / m
  J
  
}

#gradient function
grad <- function(theta, X, y) {
  
  m <- length(y) 
  
  h <- sigmoid(X %*% theta)
  grad <- (t(X) %*% (h - y)) / m
  grad
  
}

logisticReg <- function(X, y) {
  
  temp <- bind_cols(X, list(y = y)) %>%
    na.omit() %>%
    mutate(bias = 1)
  
  X <- temp %>% 
    select(bias, everything(), -y) %>%
    as.matrix()
  
  y <- temp %>%
    pull(y)
  
  #initialize theta
  theta <- matrix(rep(0, ncol(X)), nrow = ncol(X))
  #use the optim function to perform gradient descent
  costOpti <- optim(matrix(rep(0, ncol(X)), nrow = ncol(X)), cost, grad, X=X, y=y)
  
  costOpti$par[,1]
  
}

logisticPred <- function(X, coef) {
  
  X %>%
    na.omit() %>%
    mutate(bias = 1) %>%
    select(bias, everything()) %>%
    as.matrix(.) %*% coef %>%
    as.vector() %>%
    sigmoid()
  
}
