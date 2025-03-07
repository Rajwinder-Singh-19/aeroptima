import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

path = os.getcwd() + "/src/database/NACA_PCA_perturbed_nn_dataset.csv"
df = pd.read_csv(path)

df.dropna(inplace=True)

X = df.iloc[:, :-2]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GradientBoostingRegressor(
    n_estimators=1000,  # Number of boosting stages
    learning_rate=0.01,  # Lower values prevent overfitting
    max_depth=8,  # Depth of each tree
    subsample=0.9,  # Use 80% of data per tree
    min_samples_split=3, # Avoid overfitting
    min_samples_leaf=2,  # Regularization
    random_state=42,
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R² Score: {r2:.6f}")
