# Credit card fraud detection on the ULB dataset.
# Two unsupervised models on the same task: Isolation Forest, and an
# autoencoder (a small neural net trained to rebuild normal transactions).
# Fraud is rare here (0.17%), so accuracy is useless — a model that calls
# everything legit already scores 99.83%. Both models are judged on
# precision-recall at a low false-positive rate instead.
# Run it with: python fraud_detection.py

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HF_DATASET = "David-Egea/Creditcard-fraud-detection"
# Same ULB data, hosted as a plain CSV (used only if the HF mirror fails).
CSV_MIRROR = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
BASE = Path(__file__).parent
ARTIFACTS = BASE / "artifacts"


def load_data():
    # First try the Hugging Face mirror — no login or key needed.
    try:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET)
        return ds["train"].to_pandas()
    except Exception as e:
        print(f"HF mirror failed ({e}), trying the CSV mirror...")
        return pd.read_csv(CSV_MIRROR)


def recall_at_fpr(y_true, scores, fpr_target):
    # How much fraud we catch while keeping false alarms under fpr_target.
    order = np.argsort(-scores)
    y = np.asarray(y_true)[order]
    n_fraud = y.sum()
    n_legit = len(y) - n_fraud
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fpr = fp / n_legit
    recall = tp / n_fraud
    ok = fpr <= fpr_target
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(name, scores, y_test):
    # Scores are anomaly scores: higher means "looks more like fraud".
    ap = average_precision_score(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    # Best F1 point on the curve, so the threshold is picked from the data.
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1s[:-1]))  # last point has no threshold
    thr = float(thresholds[best])
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "average_precision": round(float(ap), 4),
        "roc_auc": round(float(auc), 4),
        "recall_at_fpr_0.1pct": round(recall_at_fpr(y_test, scores, 0.001), 4),
        "recall_at_fpr_1pct": round(recall_at_fpr(y_test, scores, 0.01), 4),
        "best_f1": round(float(f1s[best]), 4),
        "best_threshold": thr,
        "precision_at_best_f1": round(float(precision[best]), 4),
        "recall_at_best_f1": round(float(recall[best]), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    args = parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    df = load_data()
    print(f"loaded {len(df):,} rows, {int(df['Class'].sum())} fraud "
          f"({df['Class'].mean() * 100:.3f}%)")

    X = df[FEATURES].copy()
    y = df["Class"].values
    # Amount and Time are on a different scale than V1-V28, so scale them.
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])
    X = X.values

    # Stratified split, so the test set keeps the same tiny fraud rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if args.train_limit:
        X_train, y_train = X_train[: args.train_limit], y_train[: args.train_limit]
    if args.test_limit:
        X_test, y_test = X_test[: args.test_limit], y_test[: args.test_limit]

    # --- Isolation Forest: trained on all of the training rows ---
    print("training isolation forest...")
    iso = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42, n_jobs=-1
    )
    iso.fit(X_train)
    iso_scores = -iso.decision_function(X_test)
    iso_results = evaluate("isolation_forest", iso_scores, y_test)

    # --- Autoencoder: trained only on legit transactions ---
    # It learns what normal looks like, so fraud stands out as a big
    # rebuild error. Capped at 60k rows to keep the run short on CPU.
    print("training autoencoder...")
    normal = X_train[y_train == 0]
    if len(normal) > 60000:
        rng = np.random.default_rng(42)
        normal = normal[rng.choice(len(normal), 60000, replace=False)]
    ae = MLPRegressor(
        hidden_layer_sizes=(16, 8, 16),
        activation="relu",
        solver="adam",
        batch_size=512,
        max_iter=25,
        early_stopping=True,
        random_state=42,
    )
    ae.fit(normal, normal)
    ae_scores = np.mean((ae.predict(X_test) - X_test) ** 2, axis=1)
    ae_results = evaluate("autoencoder", ae_scores, y_test)

    results = {"isolation_forest": iso_results, "autoencoder": ae_results}
    winner = max(results, key=lambda k: results[k]["average_precision"])
    results["winner"] = winner
    results["test_size"] = int(len(y_test))
    results["test_fraud"] = int(y_test.sum())
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2))

    # Save both models + scaler, so the model can be reloaded and reused.
    with open(ARTIFACTS / "fraud_model.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "isolation_forest": iso,
                "autoencoder": ae,
                "features": FEATURES,
                "thresholds": {
                    k: v["best_threshold"] for k, v in results.items() if isinstance(v, dict)
                },
            },
            f,
        )

    # Precision-recall curves for both models, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, scores) in zip(
        axes, [("Isolation Forest", iso_scores), ("Autoencoder", ae_scores)]
    ):
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax.plot(recall, precision)
        ax.set_title(f"{name} (AP {results[name.lower().replace(' ', '_')]['average_precision']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "pr_curves.png", dpi=120)
    plt.close(fig)

    print(json.dumps(results, indent=2))
    print(f"winner on average precision: {winner}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-limit", type=int, default=None,
                   help="use only the first N training rows (quick test)")
    p.add_argument("--test-limit", type=int, default=None,
                   help="use only the first N test rows (quick test)")
    return p.parse_args()


if __name__ == "__main__":
    main()
