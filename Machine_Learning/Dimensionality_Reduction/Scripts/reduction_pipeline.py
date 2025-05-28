import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ========== SECTION 1: LOAD & STANDARDIZE ==========
def load_data():
    df = pd.read_csv('../Data/dim_reduction_data.csv')
    X = df.drop(columns='Y', errors='ignore')  # remove Y if exists
    y = df['Y'] if 'Y' in df.columns else None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# ========== SECTION 2: REDUCTION METHODS ==========
def apply_pca(X):
    model = PCA(n_components=2)
    return model.fit_transform(X)

def apply_tsne(X):
    model = TSNE(n_components=2, random_state=42)
    return model.fit_transform(X)

# Optional placeholders for non-compatible methods
def apply_umap(X):
    raise NotImplementedError("UMAP is not available in Python 3.13.")

def apply_autoencoder(X):
    raise NotImplementedError("TensorFlow autoencoder not supported in Python 3.13.")

# ========== SECTION 3: VISUALIZATION ==========
def visualize(X_2d, y, method):
    plt.figure(figsize=(8,6))
    if y is not None:
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y, palette='Set1', alpha=0.7)
    else:
        plt.scatter(X_2d[:,0], X_2d[:,1], alpha=0.6)
    plt.title(f'{method} Projection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    os.makedirs('../Images', exist_ok=True)
    plt.savefig(f'../Images/{method}_projection.png')
    plt.close()
    print(f'{method} projection saved to Images/')

# ========== MAIN ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        help='pca | tsne')
    args = parser.parse_args()

    X_scaled, y = load_data()

    if args.method == 'pca':
        X_2d = apply_pca(X_scaled)
    elif args.method == 'tsne':
        X_2d = apply_tsne(X_scaled)
    else:
        raise ValueError("Only 'pca' and 'tsne' are currently supported in Python 3.13.")

    visualize(X_2d, y, args.method)
