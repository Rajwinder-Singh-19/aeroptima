import numpy as np
from sklearn.decomposition import PCA
import os
dataset = np.load(os.getcwd() + "/src/database/PCA_files/NACA/NACA_bezier_controls.npy")

n_components = 5  # Use top 5 shape modes
pca = PCA(n_components=n_components)
pca.fit(dataset)
np.save(os.getcwd()+"/src/database/pca_files/naca/naca_pca_components.npy", pca.components_)
np.save(os.getcwd()+"/src/database/pca_files/naca/naca_pca_mean_airfoil.npy", pca.mean_)


