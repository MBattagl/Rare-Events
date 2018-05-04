library(tidyverse)
source("logistic-scratch.R")

signal_to_noise <- 1
n <- 500

beta_1 <- sqrt(signal_to_noise)

dat <- data_frame(x1 = rnorm(n),
                  intercept = qnorm(.05, sd = sqrt(beta_1 ^ 2 + 1)),
                  y_signal = beta_1 * x1,
                  y = intercept + y_signal + rnorm(n),
                  y_resp = y >= 0,
                  true_prob = pnorm(y_signal + intercept))

train <- dat %>%
  sample_frac(.66)

X_train <- train %>%
  select(x1)

y_train <- train %>%
  pull(y_resp)

fit <- glm(as.factor(y_resp) ~ x1, data = train, family = "binomial")

coefs <- fit %>% 
  coef() %>% 
  as.numeric()

V <- vcov(fit)

pred <- X_train %>%
  logisticPred(coefs)

X_matrix <- X_train %>%
  mutate(bias = 1) %>%
  select(bias, everything()) %>%
  as.matrix()

W <- diag(pred * (1 - pred))

diag(W)

Q <- X_matrix %*% solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix)

e <- -0.5 * diag(Q) * (2 * pred - 1)

bias <- (solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix) %*% W %*% e) %>%
  as.numeric()

unbiased_coefs <- coefs - bias

unbiased_coef_pred <- X_train %>%
  logisticPred(unbiased_coefs)

t <- data_frame(true = train$true_prob, pred = pred, unbiased_coef_pred = unbiased_coef_pred)

t %>% 
  mutate(diff = unbiased_coef_pred - pred) %>%
  ggplot(aes(x = true, y = diff)) +
  stat_summary(geom = "point")


