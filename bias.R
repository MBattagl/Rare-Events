library(tidyverse)
source("logistic-scratch.R")

simulation <- function(n, signal_to_noise) {
  
  beta_1 <- sqrt(signal_to_noise)
  
  dat <- data_frame(x1 = rnorm(n),
                    intercept = qnorm(.05, sd = sqrt(beta_1 ^ 2 + 1)),
                    y_signal = beta_1 * x1,
                    y = intercept + y_signal + rnorm(n),
                    y_resp = y >= 0,
                    true_prob = pnorm(y_signal + intercept))
  
  train <- dat %>%
    sample_frac(.66)
  
  test <- dat %>%
    anti_join(train, by = c("x1", "y", "y_resp", "true_prob"))
  
  X_train <- train %>%
    select(x1)
  
  X_test <- test %>%
    select(x1)
  
  y_train <- train %>%
    pull(y_resp)
  
  y_test <- test %>%
    pull(y_resp)
  
  fit <- glm(as.factor(y_resp) ~ x1, data = train, family = "binomial")
  
  coefs <- fit %>% 
    coef() %>% 
    as.numeric()
  
  V <- vcov(fit)
  
  pred <- X_train %>%
    logisticPred(coefs)
  
  test_pred <- X_test %>%
    logisticPred(coefs)
  
  X_matrix <- X_train %>%
    mutate(bias = 1) %>%
    select(bias, everything()) %>%
    as.matrix()
  
  X_test_matrix <- X_test %>%
    mutate(bias = 1) %>%
    select(bias, everything()) %>%
    as.matrix()
  
  W <- diag(pred * (1 - pred))
  
  Q <- X_matrix %*% solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix)
  
  e <- 0.5 * diag(Q) * (2 * pred - 1)
  
  bias <- (solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix) %*% W %*% e) %>%
    as.numeric()
  
  unbiased_coefs <- coefs - bias
  
  unbiased_coef_pred <- X_test %>%
    logisticPred(unbiased_coefs)
  
  updated_var <- (nrow(X_train) / (nrow(X_train) + ncol(X_train) + 1)) ^ 2 * V
  
  C <- (.5 - unbiased_coef_pred) * unbiased_coef_pred * (1 - unbiased_coef_pred) * diag((X_test_matrix %*% updated_var) %*% t(X_test_matrix))
  
  unbiased_pred <- unbiased_coef_pred - C
  bayesian_pred <- unbiased_coef_pred + C
  
  pred_dat <- data_frame(y_resp = test$y_resp, true_prob = test$true_prob, 
                         pred = test_pred, unbiased_coef_pred = unbiased_coef_pred, 
                         unbiased_pred = unbiased_pred, intercept = coefs[1], slope = coefs[2],
                         intercept_adjusted = unbiased_coefs[1], slope_adjusted = unbiased_coefs[2])
  
  pred_dat %>%
    mutate(original_pred_error = test_pred - true_prob,
           unbiased_coef_error = unbiased_coef_pred - true_prob,
           unbiased_pred_error = unbiased_pred - true_prob,
           bayesian_pred_error = bayesian_pred - true_prob)
  
}

mapping_wrapper <- function(signal_to_noise, samp_size, reps) {
  
  rep(samp_size, reps) %>%
    map_df(~ simulation(., signal_to_noise = signal_to_noise)) %>%
    mutate(signal_to_noise = as.character(signal_to_noise))
  
}

sim_results <- c(0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5) %>%
  map_df(~mapping_wrapper(signal_to_noise = ., samp_size = 500, reps = 5000))

sim_results %>% select(pred, unbiased_coef_pred, unbiased_pred, intercept, intercept_adjusted, slope, slope_adjusted) %>% colMeans()

sim_results %>%
  group_by(signal_to_noise) %>%
  summarise_all(mean)

sim_results %>%
  group_by(signal_to_noise) %>%
  summarise_all(~ (.) ^ 2 %>% mean())



# sim_results %>%
#   group_by(y_resp) %>%
#   summarise_all(mean)
# 
# sim_results %>%
#   group_by(y_resp) %>%
#   summarise_all(~ (.)^2 %>% mean())
