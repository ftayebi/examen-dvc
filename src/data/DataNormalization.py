from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.DataSplitting import X_test, X_train

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

output = Path("data/processed_data")
output.mkdir(parents=True, exist_ok=True)

pd.DataFrame(
    X_train_scaled, columns=X_train.columns
).to_csv(output / "X_train_scaled.csv", index=False)

pd.DataFrame(
    X_test_scaled, columns=X_test.columns
).to_csv(output / "X_test_scaled.csv", index=False)

print("Data normalization completed!")