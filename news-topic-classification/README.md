# 📰 News Topic Classification

Classifies news articles into World, Sports, Business, or Sci/Tech using the real AG News dataset (120,000 training and 7,600 test articles) — a clean starter template for text classification, from a fast TF-IDF baseline to a transformer fine-tune.

## Dataset
**AG News** — 120,000 training articles and 7,600 test articles, evenly split across 4 topics (30,000 / 1,900 per class). Each article is a headline plus a short description.
Source: [huggingface.co/datasets/fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news). Downloaded automatically on first run via `datasets.load_dataset` — if that fails, the script falls back to the fastai CSV mirror of the same dataset. No manual download needed.

## What it does
1. **Data** — loads AG News, joins headline + description into one text field, and checks the class balance (perfectly even, so no resampling needed)
2. **Baseline** — TF-IDF (word unigrams + bigrams, 30,000 features, sublinear scaling) with a LinearSVC classifier, trained on all 120,000 training articles
3. **Evaluation** — accuracy, macro F1, a full per-class report, and a 4×4 confusion matrix on the full 7,600-article test set (see `artifacts/` and `metrics.json`)
4. **DistilBERT fine-tune (optional)** — `python news_topic_classification.py --bert` fine-tunes `distilbert-base-uncased` on a stratified subsample (4,000 articles, 1 epoch by default; raise `--bert-samples` / `--bert-epochs` on a GPU). This path needs `torch` and `transformers` — see `requirements.txt`.
5. **Saved model** — the trained baseline pipeline in `artifacts/baseline_model.pkl`, ready to load and run predictions

## Results
**TF-IDF + LinearSVC baseline: 92.16% accuracy, macro F1 0.9215** on the full test set.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| World | 0.94 | 0.91 | 0.93 |
| Sports | 0.96 | 0.98 | 0.97 |
| Business | 0.89 | 0.89 | 0.89 |
| Sci/Tech | 0.90 | 0.91 | 0.90 |

Sports is the easiest to spot (F1 0.97); Business and Sci/Tech get mixed up most (143 Business articles predicted as Sci/Tech, 118 the other way) — company and technology news share a lot of vocabulary.

## Tech
Python · scikit-learn · pandas · matplotlib · Hugging Face Datasets · transformers (optional, for the DistilBERT path)

## Run it
```bash
pip install -r requirements.txt
python news_topic_classification.py
```

For a quick smoke test on a small slice of the data:
```bash
python news_topic_classification.py --train-limit 5000 --test-limit 1000
```

To also fine-tune DistilBERT (needs torch + transformers):
```bash
python news_topic_classification.py --bert --bert-samples 4000 --bert-epochs 1
```

Plots, `metrics.json`, and `baseline_model.pkl` are written to `artifacts/`.
