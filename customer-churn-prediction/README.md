# 📉 Customer Churn Prediction — Telco Dataset

Predicts which telecom customers are likely to churn, so retention efforts can target the right people.

## Dataset
**Telco Customer Churn** — 7,043 customers, 20 features, 26.5% churn rate.
Source: [Hugging Face](https://huggingface.co/datasets/sanjaycusat/telco-customer-churn3) (mirror of the classic Kaggle dataset). Downloaded automatically on first run.

## What it does
1. **Cleaning** — coerces blank `TotalCharges` to numeric, encodes the target
2. **Preprocessing** — scales numerics, one-hot encodes categoricals via `ColumnTransformer`
3. **Three models** — Logistic Regression vs Random Forest vs Gradient Boosting, compared with 5-fold CV ROC-AUC
4. **Winner:** Gradient Boosting — 0.843 ROC-AUC on the held-out test set
5. **Interpretability** — feature importances reveal TotalCharges, tenure, MonthlyCharges, and month-to-month contracts as the top churn drivers

## Tech
Python · pandas · scikit-learn · matplotlib

## Run it
```bash
pip install -r requirements.txt
python churn_prediction.py
```

Plots and `model_comparison.csv` are written to `artifacts/`.
