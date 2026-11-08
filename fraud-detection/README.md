# 💳 Credit Card Fraud Detection

Spots fraudulent card transactions in the real ULB credit card dataset (284,807 European transactions, only 492 fraud — 0.17%). Two unsupervised models go head to head: an Isolation Forest and an autoencoder, both judged on precision-recall at a low false-positive rate, not accuracy.

## Dataset
**ULB / Worldline Credit Card Fraud Detection** — 284,807 real cardholder transactions over two days in September 2013, with 492 fraud cases (0.172%). Each row has 30 inputs: V1–V28 (PCA-transformed, anonymised) plus Time and Amount; the target is Class (1 = fraud, 0 = legitimate).
Source: [huggingface.co/datasets/David-Egea/Creditcard-fraud-detection](https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection) — a public mirror of the Kaggle dataset (mlg-ulb/creditcardfraud). Downloaded automatically on first run via `datasets.load_dataset` — if that fails, the script falls back to a plain CSV mirror of the same data. No manual download needed.

## What it does
1. **Data** — loads the full dataset, scales Time and Amount (V1–V28 are already standardised), and takes a stratified 80/20 train/test split so the test set keeps the same tiny fraud rate
2. **Isolation Forest** — 200 trees, trained on all training rows; anomaly score is the negative decision function
3. **Autoencoder** — a small neural net (30 → 16 → 8 → 16 → 30) trained only on legitimate transactions, capped at 60,000 rows to keep the run short on CPU; anomaly score is the rebuild error
4. **Evaluation** — average precision, ROC-AUC, recall at 0.1% and 1% false-positive rate, and the best-F1 threshold picked from the precision-recall curve, all on the 56,962-transaction test set (see `artifacts/` and `metrics.json`)
5. **Saved model** — both trained models plus the scaler in `artifacts/fraud_model.pkl`, ready to load and score new transactions

## Results
**Autoencoder wins: average precision 0.5268** vs 0.1578 for Isolation Forest, on the 56,962-transaction test set (98 fraud cases).

| Model | Avg precision | ROC-AUC | Recall @ 0.1% FPR | Recall @ 1% FPR | Best F1 |
|---|---|---|---|---|---|
| Isolation Forest | 0.1578 | 0.9537 | 0.2551 | 0.6531 | 0.2922 |
| Autoencoder | 0.5268 | 0.9669 | 0.5510 | 0.8776 | 0.5217 |

At its best-F1 threshold the autoencoder catches 54 of the 98 fraud cases (55% recall) with about half its fraud flags correct (precision 0.50) — 55 false alarms across ~57,000 transactions. That is the honest trade-off on this dataset: accuracy would say 99.83% for a model that catches nothing, which is why this project is scored on precision-recall instead.

## Tech
Python · scikit-learn · pandas · matplotlib · Hugging Face Datasets

## Run it
```bash
pip install -r requirements.txt
python fraud_detection.py
```

For a quick smoke test on a small slice of the data:
```bash
python fraud_detection.py --train-limit 20000 --test-limit 5000
```

Plots, `metrics.json`, and `fraud_model.pkl` are written to `artifacts/`.
