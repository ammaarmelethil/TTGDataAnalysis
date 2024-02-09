import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Corrected import

# Read the CSV file
data = pd.read_csv("DATA.csv")

# Data Cleaning
data = data[data.apply(lambda x: x != "#VALUE!").all(axis=1)]
data['When'] = pd.to_datetime(data['When'])

# Convert columns to numeric and remove missing values
numeric_cols = ["Net Profit", "SPY", "VIX", "TLT", "SPX", "NDX"]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=numeric_cols)

# Split data into features (X) and target (y)
X = data[["SPY", "VIX", "TLT", "SPX", "NDX"]]
y = data["Net Profit"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Lasso regression model with cross-validation
lasso_cv_model = LassoCV(cv=5,max_iter=2700)
lasso_cv_model.fit(X_train, y_train)

# Print coefficients
lasso_coefs = dict(zip(X.columns, lasso_cv_model.coef_))
print(lasso_coefs)

# Get coefficients from the Lasso model
coef_lasso = lasso_cv_model.coef_
variables = ["SPY", "VIX", "TLT", "SPX", "NDX"]  # Assuming these are the variable names

# Create DataFrame for coefficients
coef_df = pd.DataFrame({"Variable": variables, "Coefficient": coef_lasso})

# Remove intercept (if present)
coef_df = coef_df.dropna()

# Plot coefficients using matplotlib
plt.figure(figsize=(8, 6))
plt.barh(coef_df["Variable"], coef_df["Coefficient"], color='skyblue')
plt.xlabel("Coefficient Value")
plt.ylabel("Variables")
plt.title("Lasso Regression Coefficients")
plt.grid(axis='x')
plt.show()
