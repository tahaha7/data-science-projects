# News topic classification on AG News.
# Two models on the same task: a fast TF-IDF + LinearSVC baseline,
# and an optional DistilBERT fine-tune (slower, needs a GPU or patience).
# Run the baseline with: python news_topic_classification.py
# Add --bert to also fine-tune DistilBERT on a small subsample.

import argparse
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
DATA_DIR = Path(__file__).parent / "data"
ARTIFACTS = Path(__file__).parent / "artifacts"
FASTAI_URL = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"


def load_ag_news():
    # If the mirror CSVs are already on disk (e.g. from a previous run), reuse them.
    folder = DATA_DIR / "ag_news_csv"
    if (folder / "train.csv").exists():
        train = pd.read_csv(folder / "train.csv", header=None, names=["label", "title", "text"])
        test = pd.read_csv(folder / "test.csv", header=None, names=["label", "title", "text"])
        train["label"] = train["label"] - 1  # CSV labels are 1-4, we want 0-3
        test["label"] = test["label"] - 1
        train["text"] = train["title"].fillna("") + " " + train["text"].fillna("")
        test["text"] = test["title"].fillna("") + " " + test["text"].fillna("")
        return train[["text", "label"]], test[["text", "label"]]

    # Otherwise try Hugging Face Datasets, it handles the download for us.
    try:
        from datasets import load_dataset

        ds = load_dataset("fancyzhx/ag_news")
        train = pd.DataFrame(ds["train"])
        test = pd.DataFrame(ds["test"])
        train["text"] = train["title"] + " " + train["description"]
        test["text"] = test["title"] + " " + test["description"]
        return train[["text", "label"]], test[["text", "label"]]
    except Exception as e:
        print(f"Hugging Face download failed ({e}), falling back to the fastai mirror.")

    # Fallback: the same AG News data as CSVs from the fastai mirror.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tgz = DATA_DIR / "ag_news_csv.tgz"
    if not tgz.exists():
        urllib.request.urlretrieve(FASTAI_URL, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(DATA_DIR)
    folder = DATA_DIR / "ag_news_csv"
    train = pd.read_csv(folder / "train.csv", header=None, names=["label", "title", "text"])
    test = pd.read_csv(folder / "test.csv", header=None, names=["label", "title", "text"])
    train["label"] = train["label"] - 1  # CSV labels are 1-4, we want 0-3
    test["label"] = test["label"] - 1
    train["text"] = train["title"].fillna("") + " " + train["text"].fillna("")
    test["text"] = test["title"].fillna("") + " " + test["text"].fillna("")
    return train[["text", "label"]], test[["text", "label"]]


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_baseline(train, test):
    # Plain word TF-IDF, unigrams + bigrams, then a linear SVM.
    # Simple, fast, and usually hard to beat on news text.
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)),
            ("svc", LinearSVC(C=1.0, dual=False)),
        ]
    )
    model.fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    return model, pred


def run_bert(train, test, n_samples, epochs):
    # Fine-tune DistilBERT on a stratified subsample so it runs on CPU.
    # Use the full data by raising --bert-samples (a GPU helps a lot here).
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    import torch
    from torch.utils.data import Dataset

    sub, _ = train_test_split(
        train, train_size=n_samples, stratify=train["label"], random_state=42
    )
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    class NewsDataset(Dataset):
        def __init__(self, df):
            self.enc = tok(list(df["text"]), truncation=True, max_length=128, padding=True)
            self.labels = list(df["label"])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    args = TrainingArguments(
        output_dir=str(ARTIFACTS / "bert_tmp"),
        per_device_train_batch_size=16,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=NewsDataset(sub))
    trainer.train()
    enc = tok(list(test["text"]), truncation=True, max_length=128, padding=True, return_tensors="pt")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            batch = {k: v[i : i + 64] for k, v in enc.items()}
            preds.extend(model(**batch).logits.argmax(-1).tolist())
    model.save_pretrained(str(ARTIFACTS / "bert_model"))
    tok.save_pretrained(str(ARTIFACTS / "bert_model"))
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", action="store_true", help="also fine-tune DistilBERT")
    parser.add_argument("--bert-samples", type=int, default=4000)
    parser.add_argument("--bert-epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None, help="smoke test: use only N train rows")
    parser.add_argument("--test-limit", type=int, default=None, help="smoke test: use only N test rows")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train, test = load_ag_news()
    print(f"Train: {len(train):,} rows, test: {len(test):,} rows")
    print("Class counts (train):", train["label"].value_counts().sort_index().to_dict())
    if args.train_limit:
        train = train.sample(args.train_limit, random_state=42)
    if args.test_limit:
        test = test.sample(args.test_limit, random_state=42)

    results = {}
    model, pred = run_baseline(train, test)
    with open(ARTIFACTS / "baseline_model.pkl", "wb") as f:
        pickle.dump(model, f)
    results["baseline"] = {
        "accuracy": round(accuracy_score(test["label"], pred), 4),
        "macro_f1": round(f1_score(test["label"], pred, average="macro"), 4),
        "report": classification_report(test["label"], pred, target_names=LABELS, output_dict=True),
    }
    print("Baseline accuracy:", results["baseline"]["accuracy"])
    print("Baseline macro F1:", results["baseline"]["macro_f1"])
    print(classification_report(test["label"], pred, target_names=LABELS))
    plot_confusion(
        confusion_matrix(test["label"], pred), ARTIFACTS / "confusion_matrix.png"
    )

    if args.bert:
        bert_pred = run_bert(train, test, args.bert_samples, args.bert_epochs)
        results["distilbert"] = {
            "train_samples": args.bert_samples,
            "epochs": args.bert_epochs,
            "accuracy": round(accuracy_score(test["label"], bert_pred), 4),
            "macro_f1": round(f1_score(test["label"], bert_pred, average="macro"), 4),
        }
        print("DistilBERT accuracy:", results["distilbert"]["accuracy"])
        print("DistilBERT macro F1:", results["distilbert"]["macro_f1"])

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved metrics + plots to", ARTIFACTS)


if __name__ == "__main__":
    main()
