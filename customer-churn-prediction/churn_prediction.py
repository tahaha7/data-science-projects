# Customer churn prediction on the Telco dataset (7043 customers).
# Tries 3 models - Logistic Regression, Random Forest, Gradient Boosting -
# and compares them using ROC-AUC. Run: python churn_prediction.py

from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report, RocCurveDisplay, PrecisionRecallDisplay)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DATA_PATH = Path(__file__).resolve().parent / "telco_churn.csv"
DATA_URL = ("https://huggingface.co/datasets/sanjaycusat/telco-customer-churn3/"
            "resolve/main/WA_Fn-UseC_-Telco-Customer-Churn.csv")
RANDOM_STATE = 42


def load_data():
    # download the csv if we don't have it yet
    if not DATA_PATH.exists():
        print("Downloading Telco Customer Churn dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # TotalCharges has some blanks (new customers), fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df = df.drop(columns=["customerID"])  # id column, not useful for prediction
    df["Churn"] = (df["Churn"] == "Yes").astype(int)  # 1 = churned, 0 = stayed
    return df


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_data()
    print(f"Dataset: {df.shape[0]} customers, {df.shape[1]-1} features, "
          f"churn rate = {df['Churn'].mean():.1%}")

    # quick look: churn rate by contract type
    print("\nChurn rate by contract:")
    print(df.groupby("Contract")["Churn"].mean().round(3).to_string())

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # scale the numbers, one-hot encode the categories
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    results, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocess), ("clf", clf)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results.append({
            "model": name,
            "cv_roc_auc_mean": round(cv_auc.mean(), 4),
            "cv_roc_auc_std": round(cv_auc.std(), 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
            "test_pr_auc": round(average_precision_score(y_test, proba), 4),
        })
        fitted[name] = (pipe, proba)
        print(f"{name}: CV ROC-AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f} | "
              f"Test ROC-AUC={roc_auc_score(y_test, proba):.4f}")

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    results_df.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    print("\n" + results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipe, best_proba = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best_pipe.predict(X_test),
                                target_names=["Stayed", "Churned"]))

    # ROC and PR curves for all three models
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (_, proba) in fitted.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=axes[1])
    axes[0].set_title("ROC Curves"); axes[1].set_title("Precision-Recall Curves")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "roc_pr_curves.png", dpi=120); plt.close(fig)

    # confusion matrix of the best model
    cm = confusion_matrix(y_test, best_pipe.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stayed", "Churned"]); ax.set_yticklabels(["Stayed", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=120); plt.close(fig)

    # which features matter most (using random forest)
    rf_pipe, _ = fitted["RandomForest"]
    feat_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    importances = rf_pipe.named_steps["clf"].feature_importances_
    top = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#1f6feb")
    ax.set_title("Top 12 Feature Importances — RandomForest")
    ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "feature_importance.png", dpi=120); plt.close(fig)
    print("\nTop churn drivers:\n" + top.to_string())

    print(f"\nArtifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
