#  Regression Modeling Project (ML/Regression)

This module explores regression methods using Python — with a focus on **how data structure affects model performance**. We compare three fundamental regression techniques:

- Linear Regression
- Lasso Regression (L1 regularization)
- Ridge Regression (L2 regularization)

---

##  Learning Objectives

1. Understand core regression techniques through hands-on coding.
2. Analyze how different data structures (e.g., noise, sparsity, collinearity) influence model behavior and performance.
3. Compare model results visually and statistically across various conditions.
4. Develop insights into model selection based on data quality and dimensionality.

---

##  Dataset

We use synthetic datasets generated using Python (e.g., `sklearn.datasets.make_regression`). These allow precise control over:

- Noise
- Collinearity
- Sparsity
- Dimensionality

---

##  Experiment Flow

Each experiment involves **the same base dataset**, modified under different structural conditions:

###  Phase 1: Normal Data (clean structure)

- No noise, no collinearity
- Apply Linear, Lasso, and Ridge regression
- Compare coefficients, R², and prediction accuracy

---

###  Phase 2: High Collinearity

- Add highly correlated features
- Observe overfitting and instability in Linear Regression
- Observe how Lasso/Ridge manage multicollinearity

---

###  Phase 3: Sparse Data

- Introduce many irrelevant or zero-value features
- Observe Lasso's feature selection behavior
- Compare model performance and interpretability

---

###  Phase 4: High-Dimensional Data

- Increase number of features >> number of samples
- Analyze behavior of all three regressors
- Discuss overfitting and regularization effectiveness

---

##  Folder Structure

```text
ML/Regression/
├── Data/              # Synthetic datasets (generated)
├── Scripts/           # All modeling code
├── Model/             # Saved models (pkl/joblib)
├── Images/            # Visualization results
├── requirements.txt   # Install dependencies
└── README.md          # Project overview (this file)
```

##  How to Run
Train model for a given scenario:

```bash
python Scripts/train.py --scenario normal --model linear
```

## Evaluate and visualize:
```bash
python Scripts/evaluate.py --scenario normal --model linear
```

Repeat the same for lasso, ridge, and other scenarios (collinear, sparse, highdim).

## Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R^2 Score
- Coefficient sparsity (for Lasso)


## Using a Trained Model on New Data

After training, you can apply your saved model to real or unseen data using:

```bash
python Scripts/model.py
```
---

##  Model Saving and Deployment

After training, each model is saved in the `Model/` folder using the `.pkl` format. This allows you to reuse trained models later without retraining.

### Example:

```bash
# Saved automatically after training:
Model/linear_phase1_normal.pkl
Model/ridge_phase2_collinear.pkl
...
```

These files can be loaded in other scripts or applications for prediction:

```python
import joblib
model = joblib.load('Model/ridge_phase2_collinear.pkl')
```

### Supported Formats

| Format    | Use Case                                 | Library               |
| --------- | ---------------------------------------- | --------------------- |
| `.pkl`    | Generic Python objects                   | `pickle`, `joblib`    |
| `.joblib` | Like `.pkl`, better for large arrays     | `joblib`              |
| `.h5`     | Deep learning models                     | `keras`, `tensorflow` |
| `.onnx`   | Framework-agnostic model exchange format | `onnx`                |

Choose the format based on your project's requirements. For regression models built with `scikit-learn`, `.pkl` and `.joblib` are typically the best choices.

---
- Discussions (for Students)
- When does Linear Regression fail?
- Why does Lasso set some coefficients to zero?
- How does Ridge handle correlated features differently?
- Why do regularized models outperform in high-dimensional spaces?
- What data structure patterns call for regularization?

## Requirements
See requirements.txt for all dependencies:
```bash
pip install -r requirements.txt
```

```
