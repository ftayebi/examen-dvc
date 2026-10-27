from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
df = pd.read_csv(url)

#print(df.head())
#print(df.columns.tolist())
#print(df.dtypes)
#print(df.select_dtypes(exclude="number").columns.tolist())


X = df.drop(columns=["date", "silica_concentrate"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

output = Path("data/processed_data")
output.mkdir(parents=True, exist_ok=True)

X_train.to_csv(output / "X_train.csv", index=False)
X_test.to_csv(output / "X_test.csv", index=False)
y_train.to_frame().to_csv(output / "y_train.csv", index=False)
y_test.to_frame().to_csv(output / "y_test.csv", index=False)

print("Data splitting completed!")