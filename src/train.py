import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

mlflow.set_experiment("CreditCard_Default_Prediction")

def run_training(n_estimators=100, max_depth=5, min_samples_split=2):
    df = pd.read_csv("data/raw/credit_card.csv")

    target_col = "Y"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}_s{min_samples_split}"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, "model")

        print(f"n_estimators={n_estimators}, max_depth={max_depth} -> ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

if __name__ == "__main__":
    run_training(n_estimators=50,  max_depth=3)
    run_training(n_estimators=100, max_depth=5)
    run_training(n_estimators=200, max_depth=7)
    run_training(n_estimators=100, max_depth=10)
    run_training(n_estimators=200, max_depth=10, min_samples_split=5)
