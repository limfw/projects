# Scripts/generate_data.py
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import os

def generate_phase1_normal(path):
    X, y = make_regression(n_samples=100, n_features=5, noise=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df['Y'] = y
    df.to_csv(os.path.join(path, 'phase1_normal.csv'), index=False)


def generate_phase2_collinear(path):
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X_collinear = np.hstack([X, X[:, [0]] * 0.9 + np.random.randn(100, 1)*0.1])
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 2
    df = pd.DataFrame(X_collinear, columns=['X0', 'X1', 'X2', 'X3_collinear'])
    df['Y'] = y
    df.to_csv(os.path.join(path, 'phase2_collinear.csv'), index=False)



def generate_phase3_sparse(path):
    X, y = make_regression(n_samples=100, n_features=20, noise=10, random_state=42)
    X[:, 10:] = 0 
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df['Y'] = y
    df.to_csv(os.path.join(path, 'phase3_sparse.csv'), index=False)



def generate_phase4_highdim(path):
    X, y = make_regression(n_samples=50, n_features=100, noise=15, random_state=42)
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df['Y'] = y
    df.to_csv(os.path.join(path, 'phase4_highdim.csv'), index=False)



if __name__ == '__main__':
    out_path = '../Data'
    os.makedirs(out_path, exist_ok=True)
    generate_phase1_normal(out_path)
    generate_phase2_collinear(out_path)
    generate_phase3_sparse(out_path)
    generate_phase4_highdim(out_path)
    