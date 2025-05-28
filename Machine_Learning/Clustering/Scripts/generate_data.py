import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_classification

# Create directory to save data
os.makedirs("../Data", exist_ok=True)

# Phase 1: Well-separated spherical blobs (K-Means friendly)
X1, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
df1 = pd.DataFrame(X1, columns=["X1", "X2"])
df1.to_csv("../Data/phase1_blobs.csv", index=False)

# Phase 2: Overlapping elliptical blobs (challenging for K-Means)
X2, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.5, 1.5, 1.5], random_state=42)
df2 = pd.DataFrame(X2, columns=["X1", "X2"])
df2.to_csv("../Data/phase2_overlap.csv", index=False)

# Phase 3: Sparse data with noise (good for DBSCAN)
X3, _ = make_blobs(n_samples=250, centers=3, cluster_std=1.0, random_state=42)
noise = np.random.uniform(low=-10, high=10, size=(50, 2))
X3 = np.vstack([X3, noise])
df3 = pd.DataFrame(X3, columns=["X1", "X2"])
df3.to_csv("../Data/phase3_noise.csv", index=False)

# Phase 4: Uneven density / bandwidth
X4, _ = make_blobs(n_samples=[100, 200, 50], centers=[[0, 0], [5, 5], [2, -2]], cluster_std=[0.3, 1.0, 2.5], random_state=42)
df4 = pd.DataFrame(X4, columns=["X1", "X2"])
df4.to_csv("../Data/phase4_varied.csv", index=False)

# Phase 5: Nested or chaining structure (good for Hierarchical)
X5, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
df5 = pd.DataFrame(X5, columns=["X1", "X2"])
df5.to_csv("../Data/phase5_nested.csv", index=False)
