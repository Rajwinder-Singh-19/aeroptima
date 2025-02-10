import numpy as np
import os

mean_aerofoil = np.load(os.getcwd()+"/src/database/pca_mean_airfoil.npy")
pca_components = np.load(os.getcwd()+"/src/database/pca_components.npy")

# Print their shapes
print("Mean airfoil shape:", mean_aerofoil.shape)
print("PCA components shape:", pca_components.shape)  # Should be (n_components, 160) or (n_components, 80)

# Print some sample values
print("\nMean Airfoil Data (First 10 values):", mean_aerofoil[:10])
print("\nPCA Component 0 (First 10 values):", pca_components[0, :10])  # First component