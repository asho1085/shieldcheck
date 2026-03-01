import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CSV_PATH = "datasetml.csv"
OUT_PATH = "model.pkl"


def train_and_save(csv_path: str = CSV_PATH, out_path: str = OUT_PATH) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    y = df["Result"].astype(int)
    X = df.drop(columns=["Result"])

    if "index" in X.columns:
        X = X.drop(columns=["index"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Test accuracy: {acc:.4f}")

    payload = {"model": model, "columns": list(X.columns)}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    train_and_save()