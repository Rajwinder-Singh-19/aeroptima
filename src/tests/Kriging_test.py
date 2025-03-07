import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

path = os.getcwd() + "/src/database/NACA_PCA_perturbed_nn_dataset.csv"
df = pd.read_csv(path)
df.dropna(inplace=True)

N_SAMPLES = 5000
assert N_SAMPLES <= df.shape[0]
X = df.iloc[:N_SAMPLES, :-2].values
y = df.iloc[:N_SAMPLES, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40, alpha=1e-6)

gpr.fit(X_train, y_train)
y_pred = gpr.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(gpr, f"gpr_{N_SAMPLES}.pkl")

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R² Score: {r2:.6f}")