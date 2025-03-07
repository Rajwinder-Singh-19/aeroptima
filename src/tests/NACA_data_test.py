import os
import numpy as np
import pandas as pd
from xfoil.analysis import aero_analysis
from scipy.interpolate import Rbf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
from classes.bezierfoil import BezierFoil

path = os.getcwd() + "/src/database/NACA_PCA_perturbed_nn_dataset.csv"

df = pd.read_csv(path)
df.dropna(inplace=True)

X = df.iloc[:, :-2].values
y = df.iloc[:, -1].values

pca = PCA(n_components=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50 ,test_size=0.3, shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

thin_plate_interp = Rbf(*X_train_pca.T, y_train, function='thin_plate', smooth=0.1)

y_pred = thin_plate_interp(*X_test_pca.T)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R² Score: {r2:.6f}")

TEST_FOIL = BezierFoil(UDB['ah81k144_dat'], 10, pca_components="NACA/naca_pca_components.npy",
    mean_pca_foil="NACA/naca_pca_mean_airfoil.npy")

upper = TEST_FOIL.upper_control.flatten()
lower = TEST_FOIL.lower_control.flatten()
test_control = np.hstack((upper, lower))
test_control_scaled = scaler.transform(test_control.reshape(1, -1))
test_control_pca = pca.transform(test_control_scaled.reshape(1,-1))
y_plate = thin_plate_interp(*test_control_pca.T)

TEST_FOIL.save_foil("TestFoil", "Foil", "Foil.dat", 10, 10)

cd = aero_analysis("Foil", "Foil.dat", 1, 10, 300, 6e6, 10, 1000, 0, 5)[1]

print(f"Predicted: {y_plate}")
print(f"Xfoil: {cd}")