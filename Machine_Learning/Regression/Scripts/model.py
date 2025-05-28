# Scripts/model.py
import joblib
import pandas as pd

def load_model(model_path):
    return joblib.load(model_path)

def predict_new_data(model, input_data_path):
    df = pd.read_csv(input_data_path)
    predictions = model.predict(df)
    return predictions

if __name__ == "__main__":
    # Example usage
    model_file = '../Model/ridge_phase1_normal.pkl'
    new_data_file = 'Data/new_data.csv'  # Must have matching columns

    model = load_model(model_file)
    preds = predict_new_data(model, new_data_file)

    print("Predictions:")
    print(preds)
