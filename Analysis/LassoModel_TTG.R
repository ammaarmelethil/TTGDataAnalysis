rm(list = ls())
data <- read.csv("~/Downloads/Capstone Data  - Capstone Data.csv", header=TRUE)

#Data Cleaning 

data <- data[!apply(data, 1, function(x) any(x == "#VALUE!")), ]

library(glmnet)
library(dplyr)
library(lubridate)

data$When <- mdy_hm(data$When)

data$Net.Profit <- as.numeric(as.character(data$Net.Profit))
data$SPY <- as.numeric(as.character(data$SPY))
data$VIX <- as.numeric(as.character(data$VIX))
data$TLT <- as.numeric(as.character(data$TLT))
data$NDX <- as.numeric(as.character(data$NDX))
data$SPX <- as.numeric(as.character(data$SPX))

data <- na.omit(data[, c("Net.Profit", "SPY", "VIX", "TLT", "SPX", "NDX")])

#lasso model 

x <- as.matrix(data[, c("SPY", "VIX", "TLT", "SPX", "NDX")])
y <- data$Net.Profit

cv_model <- cv.glmnet(x, y, alpha=1)
best_lambda <- cv_model$lambda.min
lasso_model <- glmnet(x, y, alpha=1, lambda=best_lambda)

print(coef(lasso_model))

#graphing
coef_lasso <- coef(lasso_model, s = best_lambda)
coef_matrix <- as.matrix(coef_lasso)
coef_df <- data.frame(Variable = rownames(coef_matrix), Coefficient = coef_matrix[,1])
coef_df <- coef_df[-1, ]

library(ggplot2)

ggplot(coef_df, aes(x = Variable, y = Coefficient)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  coord_flip() + 
  xlab("Variables") +
  ylab("Coefficient Value") +
  ggtitle("Lasso Regression Coefficients")

#bootstrap

n_bootstrap <- 1000

coef_matrix <- matrix(NA, nrow = n_bootstrap, ncol = ncol(data) - 1)
colnames(coef_matrix) <- names(data)[-which(names(data) == "Net.Profit")]

for (i in 1:n_bootstrap) {

  sample_data <- data[sample(nrow(data), replace = TRUE), ]
  
  x <- as.matrix(sample_data[, c("SPY", "VIX", "TLT", "SPX", "NDX")])
  y <- sample_data$Net.Profit
  
  lasso_model <- glmnet(x, y, alpha = 1, lambda = cv_model$lambda.min)
  
  coef_matrix[i, ] <- as.numeric(coef(lasso_model, s = cv_model$lambda.min)[-1])
}

coef_mean <- apply(coef_matrix, 2, mean, na.rm = TRUE)
coef_sd <- apply(coef_matrix, 2, sd, na.rm = TRUE)


print(coef_mean)
print(coef_sd)

#graphing bootstrap

coef_data <- data.frame(
  Variable = names(coef_mean),
  CoefMean = coef_mean,
  CoefSD = coef_sd)

coef_data$LowerBound = coef_data$CoefMean - coef_data$CoefSD
coef_data$UpperBound = coef_data$CoefMean + coef_data$CoefSD

library(ggplot2)

ggplot(coef_data, aes(x = Variable, y = CoefMean)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = LowerBound, ymax = UpperBound), width = 0.2) +
  theme_minimal() +
  coord_flip() + 
  xlab("Variables") +
  ylab("Coefficient Value") +
  ggtitle("Bootstrap Lasso Regression Coefficients")


