import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate sample data
X, true_labels = make_moons(n_samples=500, noise=0.05)

# Apply DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = db.labels_

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Highlighting the core samples
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
plt.scatter(X[core_samples_mask, 0], X[core_samples_mask, 1], s=50, c=labels[core_samples_mask], edgecolor="black", marker="o")

# Plot configuration
plt.title('DBSCAN Clustering')
plt.show()
