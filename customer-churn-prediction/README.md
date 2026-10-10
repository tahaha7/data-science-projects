# 📉 Customer Churn Prediction

Predicts which telecom customers are likely to churn, so retention efforts can target the right people.

## What it does
1. **Synthetic data** — generates 5,000 realistic telecom customers (tenure, charges, contract, payment method, internet service, tech support…) with churn driven by known risk factors
2. **Preprocessing** — scales numerics, one-hot encodes categoricals via a `ColumnTransformer` pipeline
3. **Three models** — Logistic Regression vs Random Forest vs Gradient Boosting, compared with 5-fold CV ROC-AUC
4. **Evaluation** — test ROC-AUC, PR-AUC, ROC & precision-recall curves, confusion matrix, classification report
5. **Interpretability** — top-12 feature importances showing what actually drives churn

## Tech
Python · pandas · scikit-learn · matplotlib

## Run it
```bash
pip install -r requirements.txt
python churn_prediction.py
```

Plots and `model_comparison.csv` are written to `artifacts/`.

## Sample output
- CV ROC-AUC per model (mean ± std), test ROC-AUC / PR-AUC
- `roc_pr_curves.png`, `confusion_matrix.png`, `feature_importance.png`
