import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from classes.bezierfoil import BezierFoil
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
from xfoil.analysis import aero_analysis

# Load the dataset
df = pd.read_csv(os.getcwd() + "/src/database/naca_pca_perturbed_nn_dataset.csv")
df.dropna(inplace=True)

# Extract input (Bezier control points) and output (Cl, Cd)
X = df.iloc[:, :-2].values  # Bezier control points
y = df.iloc[:, -2:].values  # Cl, Cd

# Normalize data (Using MinMaxScaler to avoid distorting small values)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

class BestAirfoilNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BestAirfoilNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 263),
            nn.BatchNorm1d(263),
            nn.ReLU(),
            nn.Dropout(0.10079807725112308),
            nn.Linear(263, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)
# Define the improved Neural Network
class AirfoilNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(AirfoilNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),  # Normalizes layer outputs
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model
input_size = X.shape[1]
output_size = y.shape[1]
model = BestAirfoilNN(input_size, output_size)

# Define loss function and optimizer
criterion = nn.HuberLoss()  # Using MAE (less sensitive to outliers)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.85)

# Training loop with validation and learning rate adjustment
num_epochs = 500
best_val_loss = float('inf')
early_stopping_patience = 50
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_predictions = model(X_batch)
            val_loss += criterion(val_predictions, y_batch).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch}: Train Loss = {train_loss:.5f}, Val Loss = {val_loss:.5f}")

    scheduler.step(val_loss)  # Adjust learning rate if needed

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_airfoil_nn.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load("best_airfoil_nn.pth"))

# Evaluate on test data
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.5f}")

# Function to predict aerodynamics for a new shape
def predict_airfoil(control_points):
    control_points = np.array(control_points).reshape(1, -1)
    control_points = scaler_X.transform(control_points)
    control_tensor = torch.tensor(control_points, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prediction = model(control_tensor).numpy()

    return scaler_y.inverse_transform(prediction)

# Example: Predict aerodynamics for a new shape
test_foil = BezierFoil(UDB['ah63k127_dat'], 10)
test_control = [*test_foil.upper_control, *test_foil.lower_control]
predicted_aero = predict_airfoil(test_control)
print("Predicted Cl, Cd, Cl/Cd:", predicted_aero)
test_foil.save_foil("TestFoil", "Foil", "Foil.dat", 10, 8)
actual = aero_analysis("Foil", "Foil.dat", 1, 10, 300, 6e6, 10, 1000, 0)
print("Actual Cl, Cd, Cl/Cd:", actual[0:2])