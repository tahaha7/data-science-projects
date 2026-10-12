"""
California House Price Prediction
=================================
Predicts median house prices (in $100k) for California districts using the
1990 US Census California Housing dataset (20,640 districts, 8 features).

Dataset: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
(sklearn.datasets.fetch_california_housing — downloaded automatically on first run)

Pipeline:
  1. Load dataset (auto-download via sklearn fetch)
  2. EDA: distributions, correlation heatmap, geographic scatter
  3. Preprocessing: median-impute + standardize features (Pipeline)
  4. Train & compare: Ridge, Lasso, Random Forest, Gradient Boosting
     with 5-fold cross-validated RMSE / R²
  5. Winner evaluation: residual analysis, predicted-vs-actual plot,
     feature importances
  6. Save plots + model_comparison.csv to artifacts/

Run:
    pip install -r requirements.txt
    python house_price_prediction.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42

MODEL_SPECS = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=5000),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, random_state=RANDOM_STATE),
}


def load_data():
    """Fetch California Housing (auto-downloads on first run)."""
    print("Loading California Housing dataset (downloads on first run)...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    print(f"  shape={df.shape}, features={list(data.feature_names)}")
    return df, data.target.name, data.feature_names


def eda(df, target, features):
    """Exploratory plots: target distribution, correlation, geography."""
    ARTIFACTS.mkdir(exist_ok=True)

    # 1. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[target], bins=50, color="#58a6ff", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Median house value ($100k)")
    axes[0].set_ylabel("Districts")
    axes[0].set_title("Distribution of house values")
    # clipped at 5.0 ($500k cap in the 1990 census data)
    axes[1].boxplot(df[target].sample(min(3000, len(df)), random_state=RANDOM_STATE))
    axes[1].set_ylabel("Median house value ($100k)")
    axes[1].set_title("Target boxplot (note the $500k cap)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Correlation heatmap (features + target)
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    ax.set_title("Feature correlation matrix")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "correlation.png", dpi=150)
    plt.close(fig)

    # 3. Geographic scatter colored by price
    sample = df.sample(min(8000, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(sample["Longitude"], sample["Latitude"], c=sample[target],
                    cmap="plasma", s=3, alpha=0.6, vmin=0, vmax=5)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("House value by location (coastal = pricier)")
    fig.colorbar(sc, ax=ax, label="Median house value ($100k)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "geography.png", dpi=150)
    plt.close(fig)

    # capped values
    capped = (df[target] >= 5.0).mean()
    print(f"  {capped:.1%} of districts hit the $500k collection cap (target clipped)")


def build_pipelines():
    """Median-impute + standardize, then model — inside each pipeline."""
    return {
        name: Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]) for name, model in MODEL_SPECS.items()
    }


def compare_models(X_train, y_train):
    """5-fold CV RMSE for every model."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in build_pipelines().items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="neg_root_mean_squared_error",
                                 n_jobs=-1)
        rmse = -scores
        results[name] = {"mean_rmse": rmse.mean(), "std_rmse": rmse.std()}
        print(f"  {name:>16}: 5-fold CV RMSE = {rmse.mean():.4f} ± {rmse.std():.4f}")
    return results


def evaluate_winner(name, X_train, X_test, y_train, y_test, features):
    """Fit the winner on train, evaluate on test, plot diagnostics."""
    pipe = build_pipelines()[name]
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"\nWinner: {name}  |  Test RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # Predicted vs actual
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, pred, s=4, alpha=0.25, color="#58a6ff")
    lims = [0, 5.2]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual value ($100k)"); ax.set_ylabel("Predicted value ($100k)")
    ax.set_title(f"{name}: predicted vs actual (R²={r2:.3f})")
    ax.legend(); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # Residuals
    resid = y_test - pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(pred, resid, s=4, alpha=0.25, color="#f0883e")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_xlabel("Predicted value ($100k)"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs predicted")
    axes[1].hist(resid, bins=60, color="#3fb950", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "residuals.png", dpi=150)
    plt.close(fig)

    # Feature importances (tree models expose them)
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot.barh(ax=ax, color="#58a6ff", edgecolor="black")
        ax.set_xlabel("Importance"); ax.set_title(f"{name}: feature importances")
        fig.tight_layout()
        fig.savefig(ARTIFACTS / "feature_importance.png", dpi=150)
        plt.close(fig)
        print("  Top 3 features:", ", ".join(imp.tail(3).index[::-1]))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df, target, features = load_data()
    features = list(features)

    print("\n[1/4] EDA")
    eda(df, target, features)

    X = df[features]; y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    print("\n[2/4] Model comparison (5-fold CV)")
    results = compare_models(X_train, y_train)
    winner = min(results, key=lambda k: results[k]["mean_rmse"])

    print("\n[3/4] Winner evaluation on held-out test set")
    test_metrics = evaluate_winner(winner, X_train, X_test, y_train, y_test, features)

    print("\n[4/4] Saving comparison table")
    rows = [{"model": n, "cv_rmse_mean": v["mean_rmse"], "cv_rmse_std": v["std_rmse"]}
            for n, v in results.items()]
    pd.DataFrame(rows).to_csv(ARTIFACTS / "model_comparison.csv", index=False)

    print(f"\nDone. Plots + model_comparison.csv in {ARTIFACTS}/")
    print(f"FINAL: winner={winner} test_RMSE={test_metrics['rmse']:.4f} "
          f"test_MAE={test_metrics['mae']:.4f} test_R2={test_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
