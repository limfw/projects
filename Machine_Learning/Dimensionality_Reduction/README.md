# Dimensionality Reduction Module (ML/DimReduction)

This module demonstrates four powerful dimensionality reduction techniques using a synthetic dataset crafted to simulate real-world challenges like high correlation, low variance, noise, and high sparsity.

---

## Motivation

In many machine learning applications, especially with high-dimensional data, irrelevant or redundant features can:
- Slow down training
- Cause overfitting
- Make interpretation difficult
- Obscure meaningful patterns

Dimensionality reduction and feature selection are critical for:
- **Noise reduction**
- **Model efficiency**
- **Visualization**
- **Improved generalization**

---

## Dataset Characteristics

Our synthetic dataset mimics the following real-world data issues:

| Feature Issue              | Purpose                                                      |
|----------------------------|--------------------------------------------------------------|
| Low variance               | To demonstrate removal of non-informative features           |
| High mutual correlation    | To illustrate feature redundancy                             |
| High noise                 | Simulated as irrelevant columns and slight Gaussian noise    |
| Sparse features            | Mimics many real-world datasets (e.g., text or genomic data) |
| Block-wise correlation     | Feature groups with strong internal correlation              |
| High-dimensionality        | Forces the need for projection or compression                |

---

## Included Techniques

| Method         | Description                                                   | Use Case                            |
|----------------|---------------------------------------------------------------|-------------------------------------|
| PCA            | Projects data into uncorrelated directions with max variance  | Linear projection, denoising        |
| t-SNE          | Preserves local structure for visualization                   | Nonlinear, low-dimensional plots    |
| UMAP*          | Better global structure, faster than t-SNE                    | Manifold learning, visualization    |
| Autoencoder*   | Neural network-based compression and reconstruction           | Nonlinear, deep structure discovery |

> \* Requires Python 3.10


---

## Project Structure

```text
ML/
└── DimReduction/
    ├── Data/                        # Generated high-dim synthetic datasets
    ├── Model/                       # Saved encoder models (for Autoencoder)
    ├── Images/                      # Correlation heatmaps, embedding plots
    ├── Scripts/
    │   ├── generate_data.py         # Create high-dimensional structured data
    │   ├── reduction_pipeline.py    # Run PCA, t-SNE
    │   └── model.py                 # Defines and loads Autoencoder
    ├── requirements.txt
    └── README.md
```

## How to Run
Step 1: Generate Data
```bash
python Scripts/generate_data.py
```

Step 2: Run a Dimensionality Reduction Technique
```bash
python Scripts/reduction_pipeline.py --method pca
python Scripts/reduction_pipeline.py --method tsne

```

## Outputs
- Embedding plots saved in Images/
- Autoencoder models saved in Model/
- Intermediate statistics (e.g., explained variance) printed

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## requirements.txt
numpy
pandas
scikit-learn
matplotlib
seaborn
umap-learn
joblib
tensorflow  # For Autoencoder

## Notes
- t-SNE and UMAP are primarily used for visualization, not predictive modeling.
- PCA is useful for both analysis and model input.
- Autoencoders can be fine-tuned to perform denoising and dimensionality reduction.