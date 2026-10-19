# 📈 Store Sales Forecasting

Forecasts daily grocery sales for Corporación Favorita (Ecuador) using 4.5 years of real store data — useful as a time-series forecasting template for retail demand planning.

## Dataset
**Store Sales – Time Series Forecasting** (Kaggle competition) — 3M+ rows: daily sales by store and product family (33 families, 54 stores, 2013-01-01 to 2017-08-15), plus oil prices, holidays/events, and transactions.
Source: [kaggle.com/competitions/store-sales-time-series-forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). Downloaded automatically on first run (needs a Kaggle API key at `~/.kaggle/kaggle_api_key` — competition rules must be accepted on the Kaggle page).

## What it does
1. **EDA** — top product families by sales, weekly seasonality (weekend peaks)
2. **Feature engineering** — sales aggregated to national daily level per family; lag (16/28/56-day) and rolling-mean features built strictly from history (no leakage), plus calendar, oil-price, and holiday features
3. **Time-based split** — trained on everything before 2017-07-31, validated on the last 16 days (same window as the Kaggle test set)
4. **Three models** — Naive (sales 16 days ago) vs Ridge vs LightGBM, compared on RMSLE (the competition metric), RMSE, MAE, MAPE
5. **Winner:** LightGBM — RMSLE 0.3373 (vs 0.5653 Ridge, 0.6392 Naive); MAPE 20.91%
6. **Diagnostics** — forecast-vs-actual plots for the two biggest families, LightGBM feature importances (recent rolling averages dominate)

## Tech
Python · pandas · scikit-learn · LightGBM · matplotlib

## Run it
```bash
pip install -r requirements.txt
python sales_forecasting.py
```

Plots and `model_comparison.csv` are written to `artifacts/`.
