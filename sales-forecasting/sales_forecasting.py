# Store sales forecasting on the Corporacion Favorita grocery data (Kaggle).
# Aggregates daily sales per product family, builds lag + calendar features,
# and compares a naive baseline, Ridge and LightGBM on the last 16 days.
# Run: python sales_forecasting.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
ARTIFACTS = BASE / "artifacts"
RANDOM_STATE = 42

FILES = ["train.csv", "oil.csv", "holidays_events.csv", "transactions.csv",
         "stores.csv", "test.csv", "sample_submission.csv"]
COMPETITION = "store-sales-time-series-forecasting"
HORIZON = 16  # days to forecast, same as the Kaggle test window


def download_data():
    # competition data needs auth, so grab it with the Kaggle API key if
    # the data folder is empty
    import urllib.request, zipfile

    if all((DATA / f).exists() for f in FILES):
        return
    DATA.mkdir(exist_ok=True)
    key_path = Path.home() / ".kaggle" / "kaggle_api_key"
    if not key_path.exists():
        raise SystemExit("No Kaggle key at ~/.kaggle/kaggle_api_key - can't fetch data")
    key = key_path.read_text().strip()
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    zip_path = DATA / "store_sales.zip"
    print("Downloading the competition data (one-off, ~22MB)...")
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
        f.write(resp.read())
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)
    zip_path.unlink()
    print("  download done")


def load_data():
    # one row per day per family across all stores (national level)
    download_data()
    print("Loading train.csv...")
    train = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    daily = (train.groupby(["date", "family"], as_index=False)
                  .agg(sales=("sales", "sum"),
                       onpromotion=("onpromotion", "sum")))
    print(f"  {len(daily):,} rows, {daily['family'].nunique()} families, "
          f"{daily['date'].min().date()} to {daily['date'].max().date()}")

    oil = pd.read_csv(DATA / "oil.csv", parse_dates=["date"])
    # oil prices stop a few weeks before the sales data ends, so stretch
    # the last known price over the full date range
    full_dates = pd.DataFrame({"date": pd.date_range(daily["date"].min(),
                                                     daily["date"].max())})
    oil = (full_dates.merge(oil, on="date", how="left")
                    .assign(dcoilwtico=lambda d: d["dcoilwtico"].ffill().bfill()))

    hol = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    # transferred holidays moved to a new date; use the actual celebration days
    nat = hol[(hol["locale"] == "National") &
              (hol["type"].isin(["Holiday", "Additional", "Bridge"])) &
              (hol["transferred"] == False)][["date"]].drop_duplicates()
    nat["is_holiday"] = 1

    daily = daily.merge(oil, on="date", how="left")
    daily = daily.merge(nat, on="date", how="left")
    daily["is_holiday"] = daily["is_holiday"].fillna(0).astype(int)
    daily = daily.sort_values(["family", "date"]).reset_index(drop=True)
    return daily


def add_features(df):
    # everything is built from data available at prediction time:
    # lags start at HORIZON so nothing leaks from the forecast window
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekend"] = (df["dow"] >= 5).astype(int)
    g = df.groupby("family")["sales"]
    for lag in [HORIZON, 28, 56]:
        df[f"lag_{lag}"] = g.shift(lag)
    # rolling stats on the shifted series, per family (so windows never
    # bleed into another family's series)
    lagged = g.shift(HORIZON)
    fam = df["family"]
    roll = lambda s, w, f: s.groupby(fam).transform(
        lambda x: x.rolling(w).agg(f))
    df["roll7"] = roll(lagged, 7, "mean")
    df["roll28"] = roll(lagged, 28, "mean")
    df["rollstd28"] = roll(lagged, 28, "std")
    df["promo_roll7"] = roll(df.groupby("family")["onpromotion"].shift(HORIZON),
                             7, "mean")
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    # the model predicts log(sales), so put the sales-based features on the
    # same log scale - a linear model can't fit log(y) from raw-scale lags
    for col in ["lag_16", "lag_28", "lag_56", "roll7", "roll28",
                "rollstd28", "promo_roll7", "onpromotion"]:
        df[col] = np.log1p(df[col])
    return df


def rmsle(y_true, y_pred):
    # the Kaggle competition metric
    return np.sqrt(np.mean((np.log1p(y_pred.clip(min=0)) - np.log1p(y_true)) ** 2))


def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask])


def eda(df):
    ARTIFACTS.mkdir(exist_ok=True)
    # top 10 families by total sales
    top = df.groupby("family")["sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    top.head(10).plot(kind="barh", ax=ax, color="#58a6ff")
    ax.invert_yaxis()
    ax.set_xlabel("Total sales")
    ax.set_title("Top 10 product families by total sales")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "top_families.png", dpi=150)
    plt.close(fig)

    # weekly pattern for the biggest family
    big = top.index[0]
    sub = df[df["family"] == big].copy()
    sub["dow_name"] = sub["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    weekly = sub.groupby("dow_name")["sales"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4))
    weekly.plot(kind="bar", ax=ax, color="#3fb950")
    ax.set_xlabel("Day of week")
    ax.set_ylabel(f"Avg daily sales ({big})")
    ax.set_title("Weekly seasonality - sales peak on weekends")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "weekly_seasonality.png", dpi=150)
    plt.close(fig)


def split(df, features):
    # last HORIZON days are the validation window, like the Kaggle test set
    cutoff = df["date"].max() - pd.Timedelta(days=HORIZON - 1)
    train = df[df["date"] < cutoff].dropna()
    valid = df[df["date"] >= cutoff].dropna()
    return (train[features], np.log1p(train["sales"]),
            valid[features], valid["sales"])


def main():
    df = load_data()
    df = add_features(df)
    eda(df)

    features = [c for c in df.columns
                if c not in ("date", "family", "sales")]
    X_tr, y_tr, X_va, y_va = split(df, features)
    print(f"Train rows: {len(X_tr):,}, validation rows: {len(X_va):,}")

    results = []

    # naive baseline: sales from HORIZON days ago
    # (lag features are stored log-transformed, so convert back)
    naive = np.expm1(X_va[f"lag_{HORIZON}"].fillna(0).clip(lower=0).values)
    results.append(("Naive", naive))

    # ridge on the same features (scaled, since features have very
    # different magnitudes)
    print("Training Ridge...")
    ridge = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    ridge.fit(X_tr, y_tr)
    results.append(("Ridge", np.maximum(np.expm1(ridge.predict(X_va)), 0)))

    # lightgbm
    print("Training LightGBM...")
    lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                         min_child_samples=100, subsample=0.8,
                         random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgbm.fit(X_tr, y_tr)
    results.append(("LightGBM", np.maximum(np.expm1(lgbm.predict(X_va)), 0)))

    rows = []
    for name, pred in results:
        row = {"model": name,
               "RMSLE": round(rmsle(y_va.values, pred), 4),
               "RMSE": round(np.sqrt(mean_squared_error(y_va, pred)), 2),
               "MAE": round(mean_absolute_error(y_va, pred), 2),
               "MAPE": round(mape(y_va.values, pred), 4)}
        rows.append(row)
        print(f"  {name}: RMSLE={row['RMSLE']}, MAPE={row['MAPE']:.2%}")
    comp = pd.DataFrame(rows)
    comp.to_csv(ARTIFACTS / "model_comparison.csv", index=False)

    # metric comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(comp["model"], comp["RMSLE"], color=["#8b949e", "#58a6ff", "#3fb950"])
    ax.set_ylabel("RMSLE (lower is better)")
    ax.set_title(f"Model comparison on the last {HORIZON} days")
    for i, v in enumerate(comp["RMSLE"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "model_comparison.png", dpi=150)
    plt.close(fig)

    # forecast vs actual for two big families
    top2 = (df.groupby("family")["sales"].sum()
              .sort_values(ascending=False).index[:2])
    cutoff = df["date"].max() - pd.Timedelta(days=HORIZON - 1)
    preds = pd.Series(results[2][1], index=X_va.index)  # LightGBM predictions
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, fam in zip(axes, top2):
        hist = df[(df["family"] == fam) & (df["date"] >= cutoff - pd.Timedelta(days=60))
                  & (df["date"] < cutoff)]
        fut = df[(df["family"] == fam) & (df["date"] >= cutoff)]
        ax.plot(hist["date"], hist["sales"], color="#8b949e", label="History")
        ax.plot(fut["date"], fut["sales"], color="black", label="Actual")
        ax.plot(fut["date"], preds.loc[fut.index].values, color="#f85149",
                linestyle="--", label="LightGBM forecast")
        ax.set_title(fam)
        ax.legend(fontsize=9)
    fig.suptitle(f"LightGBM forecast vs actual - last {HORIZON} days")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)

    # feature importances
    imp = pd.Series(lgbm.feature_importances_, index=features).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", ax=ax, color="#58a6ff")
    ax.set_xlabel("Importance")
    ax.set_title("LightGBM - top features")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    print("Done. Plots and model_comparison.csv are in artifacts/")


if __name__ == "__main__":
    main()
