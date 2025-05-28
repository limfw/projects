## Suggested Folder Structure

```text
ML/
└── Classification/
    ├── Data/                    # Contains generated datasets
    ├── Model/                   # Trained model files (.pkl)
    ├── Images/                  # Confusion matrices, decision boundary plots
    ├── Scripts/
    │   ├── generate_data.py     # Creates binary classification datasets
    │   ├── classification_pipeline.py  # Combined training + evaluation
    │   └── model.py             # Used only for predicting new data later
    ├── requirements.txt
    └── README.md
```

---

## README.md

````markdown
# Classification Modeling Project (ML/Classification)

This module implements three core classification algorithms to explore how data structure affects classification performance.

### Included Algorithms
- K-Nearest Neighbors (K-NN)
- Support Vector Machine (SVM)
- Logistic Regression

---

## Learning Objectives

- Understand the fundamental behavior of 3 different classification paradigms:
  - Distance-based (KNN)
  - Margin-based (SVM)
  - Probabilistic linear model (Logistic Regression)
- Compare performance across different data structures (linearly separable, noisy, overlapping, imbalanced).
- Visualize decision boundaries and confusion matrices.

---

## Dataset Scenarios

| Phase | Dataset Type           | Purpose                             |
|-------|------------------------|-------------------------------------|
| phase1_clean       | Linearly separable, balanced       | Baseline performance |
| phase2_overlap     | Overlapping class boundaries        | Test robustness      |
| phase3_noise       | Noisy and slightly imbalanced       | Evaluate regularization |
| phase4_highdim     | High-dimensional features           | Analyze overfitting |

Each dataset is generated using `make_classification()` from `sklearn.datasets`.

---

## How to Run

### Step 1: Generate All Datasets
```bash
python Scripts/generate_data.py
````

### Step 2: Train, Evaluate, Visualize

```bash
python Scripts/classification_pipeline.py --phase phase1_clean --model knn
```

---

## Output

- Trained model saved in `Model/`
- Plot saved in `Images/`
- Metrics printed: Accuracy, Precision, Recall, F1, Confusion Matrix

---

## Requirements

See `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Model Saving Formats

| Format  | Use Case                      | Library  |
| ------- | ----------------------------- | -------- |
| `.pkl`  | Generic model persistence     | `joblib` |
| `.onnx` | Cross-framework compatibility | `onnx`   |

This project uses `.pkl` for simplicity and speed.

---


````
---
## Requirements.txt

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
````

---
