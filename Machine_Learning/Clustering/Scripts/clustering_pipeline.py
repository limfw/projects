import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ====== MODEL SELECTION ======
def build_model(name):
    if name == 'kmeans':
        return KMeans(n_clusters=3, random_state=42)
    elif name == 'dbscan':
        return DBSCAN(eps=1.5, min_samples=5)
    elif name == 'gmm':
        return GaussianMixture(n_components=3, random_state=42)
    elif name == 'meanshift':
        return MeanShift()
    elif name == 'hierarchical':
        return AgglomerativeClustering(n_clusters=3)
    else:
        raise ValueError("Unsupported model: Choose from kmeans, dbscan, gmm, meanshift, hierarchical")

# ====== CLUSTERING + SILHOUETTE ======
def run_clustering(X, model_type):
    model = build_model(model_type)

    if model_type == 'gmm':
        cluster_labels = model.fit_predict(X)
    else:
        model.fit(X)
        cluster_labels = model.labels_

    score = silhouette_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else -1
    print(f"Silhouette Score: {score:.2f}")

    os.makedirs("../Model", exist_ok=True)
    joblib.dump(model, f"../Model/{model_type}_{args.phase}.pkl")

    return cluster_labels

# ====== VISUALIZATION ======
def visualize_clusters(X, labels, model_type, phase):
    os.makedirs("../Images", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set2", s=60, edgecolor="k")
    plt.title(f"{model_type.upper()} Clustering - {phase}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(f"../Images/{model_type}_{phase}.png")
    plt.close()
    print(f"Plot saved to ../Images/{model_type}_{phase}.png")

# ====== MAIN ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True,
                        help="Dataset phase: phase1_blobs, phase2_overlap, etc.")
    parser.add_argument("--model", type=str, required=True,
                        help="Clustering model: kmeans, dbscan, gmm, meanshift, hierarchical")
    args = parser.parse_args()

    df = pd.read_csv(f"../Data/{args.phase}.csv")
    X = df.values

    cluster_labels = run_clustering(X, args.model)
    visualize_clusters(X, cluster_labels, args.model, args.phase)
