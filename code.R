library(readr)
library(stats)
library(ggplot2)

data <- read_csv("data-23.csv")
t <- data$t
y <- data$`y(t)`
ggplot(data, aes(x = t, y = y)) +   geom_point(color = "blue") +        
  labs(title = "Plot of t vs y(t)", x = "t",y = "y(t)") +
  theme_minimal() 

################### Least Square Estimators ###################
# Fit Model 1: y(t) = α0 + α1 * exp(β1 * t) + α2 * exp(β2 * t) + e(t) 
loss_function_1 <- function(params) {
  a0 <- params[1]
  a1 <- params[2]
  b1 <- params[3]
  a2 <- params[4]
  b2 <- params[5]
  
  # Predicted values based on current parameters
  y_pred <- a0 + a1 * exp(b1 * t) + a2 * exp(b2 * t)
  
  # Sum of squared residuals
  sum((y - y_pred)^2)
}

# Initial guesses for the parameters
start_params_1 <- c(a0 = 1,a1 = 2, b1 = -0.01, a2 = 2, b2 = -0.01)

# Use optim to minimize the loss function
optim_results_1 <- optim(par = start_params_1, fn = loss_function_1, method = "BFGS")

# Extract the estimated parameters
a0_est <- optim_results_1$par[1]
a1_est <- optim_results_1$par[2]
b1_est <- optim_results_1$par[3]
a2_est <- optim_results_1$par[4]
b2_est <- optim_results_1$par[5]

pred1 <-  a0_est + a1_est * exp(b1_est * t) + a2_est * exp(b2_est * t)

# Fit Model 2: y(t) = (α0 + α1 * t) / (β0 + β1 * t) + e(t)
loss_function_2 <- function(params) {
  a0 <- params[1]
  a1 <- params[2]
  b0 <- params[3]
  b1 <- params[4]
  
  # Predicted values based on current parameters
  y_pred <- (a0 + a1 * t) / (b0 + b1 * t)
  
  # Sum of squared residuals
  sum((y - y_pred)^2)
}

# Initial guesses for the parameters
start_params_2 <- c(a0 = 1, a1 = 0.1, b0 = 1, b1 = 0.1)

# Use optim to minimize the loss function
optim_results_2 <- optim(par = start_params_2, fn = loss_function_2, method = "BFGS")

# Extract the estimated parameters
a0_est <- optim_results_2$par[1]
a1_est <- optim_results_2$par[2]
b0_est <- optim_results_2$par[3]
b1_est <- optim_results_2$par[4]

pred2 <- (a0_est + a1_est * t) / (b0_est + b1_est * t)

# Fit Model 3: y(t) = β0 + β1 * t + β2 * t^2 + β3 * t^3 + β4 * t^4 + e(t)
loss_function_3 <- function(params) {
  b0 <- params[1]
  b1 <- params[2]
  b2 <- params[3]
  b3 <- params[4]
  b4 <- params[5]
  
  # Predicted values based on current parameters
  y_pred <- b0 + b1*t + b2*(t^2) + b3*(t^3) + b4*(t^4)
  
  # Sum of squared residuals
  sum((y - y_pred)^2)
}

# Initial guesses for the parameters
start_params_3 <- c(b0 = mean(y), b1 = 1, b2 = 1, b3 = 1, b4 = 1)

# Use optim to minimize the loss function
optim_results_3 <- optim(par = start_params_3, fn = loss_function_3, method = "BFGS",hessian = TRUE)

# Extract the estimated parameters
b0_est <- optim_results_3$par[1]
b1_est <- optim_results_3$par[2]
b2_est <- optim_results_3$par[3]
b3_est <- optim_results_3$par[4]
b4_est <- optim_results_3$par[5]

pred3 <- b0_est + b1_est*t + b2_est*(t^2) + b3_est*(t^3) + b4_est*(t^4)

################### AIC and RSS ###################
calculate_rss <- function(actual, predicted){
  rss <- sum((actual - predicted)^2)
  return(rss)
}
rss1 <- calculate_rss(y, pred1)
rss2 <- calculate_rss(y, pred2)
rss3 <- calculate_rss(y, pred3)

n <- length(y)
k1 <- 5  #no of parameters in model-1
k2 <- 4  #no of parameters in model-2
k3 <- 5  #no of parameters in model-3

AIC_1 <- n*log(rss1/n) + 2*k1
AIC_2 <- n*log(rss2/n) + 2*k2
AIC_3 <- n*log(rss3/n) + 2*k3

## RSS3 and AIC_3 are low value, so Model-3 is the best fitted model


################### Estimated sigma2 ###################
# as the model-3 is best, we will consider model-3's residuals 
n <- length(y)
k3 <- 5  # Number of parameters in Model 3
residuals <- y - pred3
sigma2 <- sum((residuals)^2) / (n - k3)  # Residual variance
cat("Estimated sigma2 is", sigma2)

################### Confidence Intervals Based on Fisher Information Matrix ###################
# Calculate the estimated variance-covariance matrix
hessian_matrix <- optim_results_3$hessian
fisher_information_matrix <- solve(hessian_matrix)  # Inverse of the Hessian
cov_matrix <- fisher_information_matrix * sigma2  # Adjust by sigma^2 to get variances

# Calculate standard errors of the parameter estimates
std_errors <- sqrt(diag(cov_matrix))

# Compute 95% confidence intervals for each parameter
z_value <- qnorm(0.975)  # 1.96 for 95% confidence level
conf_intervals <- data.frame(
  Parameter = c("b0", "b1", "b2", "b3", "b4"),
  Estimate = c(b0_est, b1_est, b2_est, b3_est, b4_est),
  Lower_95_CI = c(b0_est, b1_est, b2_est, b3_est, b4_est) - z_value * std_errors,
  Upper_95_CI = c(b0_est, b1_est, b2_est, b3_est, b4_est) + z_value * std_errors
)
conf_intervals

################### Residuals plot ###################
residuals <- y - pred3
data_residual <- data.frame(Index = 1:length(residuals), Residuals = residuals)

# Create the plot
ggplot(data_residual, aes(x = Index, y = Residuals)) +
  geom_point(shape = 16) +         # Add points with solid circles
  geom_line(col="blue") +                    # Add a line connecting points
  geom_hline(yintercept = 0,       # Add a horizontal line at y = 0
             color = "black")+
  labs(title = "Residuals Plot", x = "Index", y = "Residuals") +
  theme_minimal()



################### Normality Verfitication ###################
# 1. Shapiro-Wilk Test 
shapiro_test <- shapiro.test(residuals)

# Print Shapiro-Wilk Test results
cat("Shapiro-Wilk Test p-value:", shapiro_test$p.value, "\n")
if (shapiro_test$p.value > 0.05) {
  cat("Residuals are likely normally distributed (fail to reject H0 at 5% level).\n")
} else {
  cat("Residuals are not normally distributed (reject H0 at 5% level).\n")
}

# 2. Q-Q Plot 
# Create the Q-Q plot
ggplot(data_residual, aes(sample = residuals)) +
  stat_qq(color="red") +
  stat_qq_line(col = "blue") +  # Black line
  ggtitle("Q-Q Plot of Residuals") +
  theme_minimal()


################### plot btw observed value and fitted curve ###################
plot(t,y,pch=16,main = "Fitted model and Data",type="b",col="#4da6ff")
lines(t,pred3,col="red",lwd=1.5)
legend("topleft", legend = c("Model-3","Data"), col = c("red","#4da6ff"), lwd = 2)

