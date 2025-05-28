from sklearn.datasets import make_classification
import pandas as pd
import os

def save_dataset(X, y, filename, path):
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    df['Y'] = y
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, filename), index=False)
    print(f"Saved: {filename}")

def generate_phase1_clean(path):
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                               n_clusters_per_class=1, class_sep=2.0, flip_y=0,
                               weights=[0.5, 0.5], random_state=42)
    save_dataset(X, y, 'phase1_clean.csv', path)

def generate_phase2_overlap(path):
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                               n_clusters_per_class=1, class_sep=0.5, flip_y=0.05,
                               weights=[0.5, 0.5], random_state=42)
    save_dataset(X, y, 'phase2_overlap.csv', path)

def generate_phase3_noise(path):
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                               n_clusters_per_class=1, class_sep=1.0, flip_y=0.2,
                               weights=[0.7, 0.3], random_state=42)
    save_dataset(X, y, 'phase3_noise.csv', path)

def generate_phase4_highdim(path):
    X, y = make_classification(n_samples=200, n_features=20, n_informative=5, 
                               n_redundant=10, class_sep=1.0, flip_y=0.05,
                               weights=[0.5, 0.5], random_state=42)
    save_dataset(X, y, 'phase4_highdim.csv', path)

if __name__ == '__main__':
    out_dir = '../Data'
    generate_phase1_clean(out_dir)
    generate_phase2_overlap(out_dir)
    generate_phase3_noise(out_dir)
    generate_phase4_highdim(out_dir)
