from sklearn import datasets
import numpy as np

datasets_dict = {
    "noisy circles": datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8),
    "noisy moons": datasets.make_moons(n_samples=1500, noise=0.05, shuffle=True, random_state=42),
    "blobs": datasets.make_blobs(n_samples=1200, n_features=2,centers=3, cluster_std=1.2, center_box=(-10.0, 10.0), shuffle=True, random_state=42, return_centers=False),
    "anisotropic": (np.dot(datasets.make_blobs(n_samples=1500, centers=2, random_state=42, shuffle=True, center_box=(-10.0, 10.0))[0], [[0.6, -0.6], [-0.4, 0.8]]), datasets.make_blobs(n_samples=1500, centers=2, random_state=42, shuffle=True, center_box=(-10.0, 10.0))[1]),
    "varied": datasets.make_classification(n_samples=1500, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)
}

base_x, base_y = datasets.make_blobs(n_samples=1500, centers=2, random_state=42,shuffle=True, center_box=(-10.0, 10.0))

# Adjusting anisotropic data to ensure the transformation is correctly applied
new_x = np.dot(base_x, [[0.6, -0.6], [-0.4, 0.8]])
datasets_dict["anisotropic"] = (new_x, base_y)