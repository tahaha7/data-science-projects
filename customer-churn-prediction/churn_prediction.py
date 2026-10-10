"""
Customer Churn Prediction
=========================
Predicts which telecom customers are likely to churn (cancel their
subscription) using synthetic customer data.

Pipeline:
  1. Generate a realistic synthetic telecom dataset
  2. Exploratory data analysis (churn rate, distributions)
  3. Feature engineering + preprocessing
  4. Train & compare: Logistic Regression, Random Forest, Gradient Boosting
  5. Evaluate with ROC-AUC, PR-AUC, confusion matrix, feature importance
  6. Save plots and a model-comparison CSV to artifacts/

Run:
    pip install -r requirements.txt
    python churn_prediction.py
"""

from pathlib import Path
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
RANDOM_STATE = 42
N_SAMPLES = 5000


def generate_data(n=N_SAMPLES, seed=RANDOM_STATE):
    """Build a synthetic telecom customer dataset with realistic churn drivers."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(20, 120, n), 2)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.45, 0.30, 0.25])
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                         n, p=[0.35, 0.2, 0.2, 0.25])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.45, 0.35, 0.20])
    support = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
    senior = rng.choice([0, 1], n, p=[0.8, 0.2])

    # Churn probability driven by known risk factors + noise
    logit = (-1.6
             - 0.035 * tenure
             + 0.012 * monthly
             + 0.9 * (contract == "Month-to-month")
             - 0.5 * (contract == "Two year")
             + 0.4 * (internet == "Fiber optic")
             - 0.45 * (support == "Yes")
             + 0.3 * (payment == "Electronic check")
             + 0.25 * senior
             + rng.normal(0, 0.5, n))
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "InternetService": internet, "TechSupport": support, "SeniorCitizen": senior,
        "Churn": churn,
    })


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = generate_data()
    print(f"Dataset: {df.shape[0]} customers, churn rate = {df['Churn'].mean():.1%}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_cols = [c for c in X.columns if c not in num_cols]

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

    # ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (_, proba) in fitted.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=axes[1])
    axes[0].set_title("ROC Curves"); axes[1].set_title("Precision-Recall Curves")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "roc_pr_curves.png", dpi=120); plt.close(fig)

    # Confusion matrix (best model)
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

    # Feature importance (best tree model if available)
    tree_model = fitted.get("RandomForest", fitted.get("GradientBoosting"))
    if tree_model:
        pipe = tree_model[0]
        feat_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pipe.named_steps["clf"].feature_importances_
        top = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(9, 6))
        top.sort_values().plot(kind="barh", ax=ax, color="#1f6feb")
        ax.set_title("Top 12 Feature Importances — RandomForest")
        ax.set_xlabel("Importance")
        fig.tight_layout(); fig.savefig(ARTIFACTS / "feature_importance.png", dpi=120); plt.close(fig)
        print("\nTop churn drivers:\n" + top.to_string())

    print(f"\nArtifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
