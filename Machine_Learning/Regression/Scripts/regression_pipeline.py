import pandas as pd
import joblib
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# ===== MODEL SELECTION =====
def build_model(model_type='linear', alpha=1.0):
    model_type = model_type.lower()
    if model_type == 'linear':
        return LinearRegression()
    elif model_type == 'lasso':
        return Lasso(alpha=alpha)
    elif model_type == 'ridge':
        return Ridge(alpha=alpha)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ===== SECTION 1: TRAINING =====
def train_model(phase, model_type):
    df = pd.read_csv(f'../Data/{phase}.csv')
    X, y = df.drop(columns='Y'), df['Y']
    model = build_model(model_type)
    model.fit(X, y)
    
    os.makedirs('../Model', exist_ok=True)
    joblib.dump(model, f'../Model/{model_type}_{phase}.pkl')
    print(f"Model trained and saved for {model_type} on {phase}")
    return model, X, y

# ===== SECTION 2: EVALUATION =====
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    return y_pred, mse, r2

# ===== SECTION 3: VISUALIZATION =====
def visualize(y, y_pred, phase, model_type):
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.7)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_type.upper()} on {phase}')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    os.makedirs('../Images', exist_ok=True)
    plot_path = f'../Images/{model_type}_{phase}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, required=True,
                        help='phase1_normal, phase2_collinear, phase3_sparse, phase4_highdim')
    parser.add_argument('--model', type=str, required=True,
                        help='linear, lasso, ridge')
    args = parser.parse_args()

    model, X, y = train_model(args.phase, args.model)
    y_pred, mse, r2 = evaluate_model(model, X, y)
    visualize(y, y_pred, args.phase, args.model)
