import pandas as pd
import joblib
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ===== MODEL SELECTION =====
def build_model(model_type='knn'):
    model_type = model_type.lower()
    if model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'svm':
        return SVC(kernel='rbf', probability=True)
    elif model_type == 'logistic':
        return LogisticRegression()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ===== TRAINING =====
def train_model(phase, model_type):
    df = pd.read_csv(f'../Data/{phase}.csv')
    X, y = df.drop(columns='Y'), df['Y']
    model = build_model(model_type)
    model.fit(X, y)
    
    os.makedirs('../Model', exist_ok=True)
    joblib.dump(model, f'../Model/{model_type}_{phase}.pkl')
    print(f" Model trained and saved for {model_type} on {phase}")
    return model, X, y

# ===== EVALUATION =====
def evaluate_model(model, X, y, phase, model_type):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    print(f"=== Classification Report for {model_type} on {phase} ===")
    print(classification_report(y, y_pred))

    os.makedirs('../Images', exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_type} on {phase}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'../Images/confmat_{model_type}_{phase}.png')
    plt.close()

    return y_pred

# ===== VISUALIZATION: CLASS IMBALANCE + DECISION BOUNDARY =====
def visualize(X, y, model, phase, model_type):
    os.makedirs('../Images', exist_ok=True)

    plt.figure()
    sns.countplot(x=y)
    plt.title(f'Class Distribution: {model_type} on {phase}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(f'../Images/classdist_{model_type}_{phase}.png')
    plt.close()
    
    if X.shape[1] == 2:
        plot_decision_boundary(X, y, model, phase, model_type)

def plot_decision_boundary(X, y, model, phase, model_type):
    import numpy as np

    h = .02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_input = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    Z = model.predict(grid_input)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(f'Decision Boundary: {model_type} on {phase}')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.savefig(f'../Images/decision_boundary_{model_type}_{phase}.png')
    plt.close()

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, required=True,
                        help='phase1_clean, phase2_overlap, phase3_noise, phase4_highdim')
    parser.add_argument('--model', type=str, required=True,
                        help='knn, svm, logistic')
    args = parser.parse_args()

    model, X, y = train_model(args.phase, args.model)
    y_pred = evaluate_model(model, X, y, args.phase, args.model)
    visualize(X, y, model, args.phase, args.model)
