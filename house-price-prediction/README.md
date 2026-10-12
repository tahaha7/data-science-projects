# 🏠 California House Price Prediction

Predicts median house values (in $100k) for California districts from 1990 US Census features — useful as a regression-modeling template for pricing problems.

## Dataset
**California Housing** — 20,640 districts, 8 numeric features (median income, house age, average rooms, location, …).
Source: [scikit-learn datasets docs](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (`sklearn.datasets.fetch_california_housing`). Downloaded automatically on first run.

## What it does
1. **EDA** — target distribution, correlation heatmap, geographic price scatter
2. **Preprocessing** — median imputation + standard scaling inside each `Pipeline` (no leakage)
3. **Four models** — Ridge vs Lasso vs Random Forest vs Gradient Boosting, compared with 5-fold CV RMSE
4. **Winner:** Gradient Boosting — CV RMSE 0.485 (vs 0.510 Random Forest, 0.721 Ridge/Lasso); test RMSE 0.488, MAE 0.328, R² 0.819
5. **Diagnostics** — predicted-vs-actual plot, residual analysis, feature importances (median income dominates)

## Tech
Python · pandas · scikit-learn · matplotlib

## Run it
```bash
pip install -r requirements.txt
python house_price_prediction.py
```

Plots and `model_comparison.csv` are written to `artifacts/`.
