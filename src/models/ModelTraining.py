
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Define paths
data_dir = Path("data/processed_data")
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

# 2. Load training data
X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()

# 3. Load best parameters from Grid Search
with open(models_dir / "best_params.pkl", "rb") as file:
    best_params = pickle.load(file)

print("Best parameters:", best_params)

# 4. Create and train the model
model = LinearRegression(**best_params)
model.fit(X_train, y_train)

# 5. Save the trained model
model_path = models_dir / "trained_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("Model training completed!")
print("Trained model saved to:", model_path)
