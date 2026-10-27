
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# 1. Load training data
data_dir = Path("data/processed_data")
X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()

# 2. Define parameters to test
param_grid = {
    "fit_intercept": [True, False],
    "positive": [True, False],
}

# 3. Search for the best parameters
grid = GridSearchCV(
    estimator=LinearRegression(),
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    refit=True,
)

grid.fit(X_train, y_train)

# 4. Save the best parameters
output = Path("models")

output.mkdir(parents=True, exist_ok=True)

with open(output / "trained_model.pkl", "wb") as file:

    pickle.dump(grid.best_estimator_, file)

print("Best parameters:", grid.best_params_)

print("Saved to:", output / "trained_model.pkl")