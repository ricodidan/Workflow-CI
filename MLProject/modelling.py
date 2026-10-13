import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {filepath}")
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def main(data_path: str, n_estimators: int, max_depth: int):
    print(f"Loading data from: {data_path}")
    X_train, X_test, y_train, y_test = load_data(data_path)

    # Memulai MLflow run
    with mlflow.start_run():
        # Logging Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        print(f"Training RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Logging Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Logging Sklearn Model Artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print("=" * 60)
        print("Training Selesai & Sukses!")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model within MLflow Project")
    parser.add_argument("--data_path", default="telco_churn_preprocessing/customer_churn_clean.csv", help="Path ke data")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah estimators")
    parser.add_argument("--max_depth", type=int, default=10, help="Maksimal depth tree")
    args = parser.parse_args()

    main(args.data_path, args.n_estimators, args.max_depth)
