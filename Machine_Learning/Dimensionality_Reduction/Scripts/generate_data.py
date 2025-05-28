import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_classification

# ===== Create directory if not exists =====
os.makedirs('../Data', exist_ok=True)

# ===== PARAMETERS =====
n_samples = 1000
n_informative = 10
n_redundant = 10
n_repeated = 0
n_features = 100  # Total features
n_clusters_per_class = 1
random_state = 42

# ===== Generate Classification-Like High-Dimensional Data =====
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_repeated=n_repeated,
    n_classes=2,
    n_clusters_per_class=n_clusters_per_class,
    flip_y=0.05,  # Add some noise
    class_sep=1.0,
    shuffle=True,
    random_state=random_state
)

df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
df["Y"] = y  

# ===== Add Low-Variance Features Manually =====
for i in range(5):
    df[f"lowvar_{i}"] = 0.01 * np.random.randn(n_samples)

# ===== Add Sparse Features (Mostly Zeros) =====
for i in range(5):
    sparse_col = np.random.binomial(1, 0.05, n_samples) * np.random.randn(n_samples)
    df[f"sparse_{i}"] = sparse_col

# ===== Save to CSV =====
df.to_csv("../Data/dim_reduction_data.csv", index=False)
print("Synthetic high-dimensional dataset generated and saved to Data/")
