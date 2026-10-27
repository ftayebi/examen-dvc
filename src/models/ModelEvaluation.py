from pathlib import Path

import pickle

import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file paths

data_dir = Path("data/processed_data")

models_dir = Path("models")

# Load test data

X_test = pd.read_csv(data_dir / "X_test_scaled.csv")

y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

# Load trained model

with open(models_dir / "trained_model.pkl", "rb") as file:

    model = pickle.load(file)

# Predict target values

y_pred = model.predict(X_test)

# Calculate evaluation metrics

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

# Display results

print("Model Evaluation Results")

print("------------------------")

print(f"MAE:  {mae:.4f}")

print(f"RMSE: {rmse:.4f}")

print(f"R²:   {r2:.4f}")