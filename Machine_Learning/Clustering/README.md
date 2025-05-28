### Clustering Modeling Project (ML/Clustering)

This module implements five core clustering algorithms to investigate how different data structures influence clustering performance and interpretability.

### Included Algorithms

- K-Means
- DBSCAN
- Gaussian Mixture Model (GMM)
- Mean Shift
- Hierarchical Clustering

---

### Learning Objectives

- Understand how each clustering algorithm identifies structure in unlabeled data.
- Evaluate how data characteristics (e.g., shape, density, noise) impact clustering outcomes.
- Visualize clustering results using scatter plots and dendrograms where applicable.

---

### Project Structure

```text
ML/
└── Clustering/
    ├── Data/                        # Contains generated datasets
    ├── Model/                       # Optional: model files or cluster assignments
    ├── Images/                      # Cluster plots and dendrograms
    ├── Scripts/
    │   ├── generate_data.py         # Synthetic data generator for clusters
    │   ├── clustering_pipeline.py   # Apply and evaluate clustering algorithms
    │   └── model.py                 # Optional: reuse models or predict cluster labels
    ├── requirements.txt
    └── README.md


### Dataset Scenarios
### Dataset Scenarios

| Phase          | Dataset Type                  | Purpose                          |
|----------------|-------------------------------|----------------------------------|
| phase1_blobs   | Well-separated spherical blobs | Best fit for K-Means             |
| phase2_overlap | Overlapping elliptical blobs   | GMM handles this well            |
| phase3_noise   | Sparse data with outliers      | Test DBSCAN robustness           |
| phase4_varied  | Uneven densities or bandwidths | Good for DBSCAN / Mean Shift     |
| phase5_nested  | Nested or chaining structure   | Best visualized with Hierarchical |

Each dataset is generated using make_blobs(), make_moons(), or make_classification() from sklearn.datasets and custom configurations.

### How to Run
Step 1: Generate All Datasets
```bash
python Scripts/generate_data.py
```

Step 2: Train & Visualize Clustering
```bash
python Scripts/clustering_pipeline.py --phase phase1_blobs --model kmeans
```

### Output
- Cluster assignments visualized in Images/
- Optional model config or predictions in Model/
- Evaluation via silhouette scores and visual inspection

### Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

### Requirements.txt
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib


### Model Saving Formats
|Format	|Use Case|	Library|
|.pkl	|Store fitted clustering models	|joblib|
Other formats (e.g., .onnx) are less relevant in unsupervised settings.