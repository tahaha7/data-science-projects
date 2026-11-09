# RAG chatbot over a Wikipedia sample.
# Loads real Wikipedia articles, chunks them, embeds them with
# sentence-transformers, indexes them in FAISS, then answers questions
# by pulling back the most relevant passages. No made-up answers:
# every answer quotes the passage it came from.
#
# Run it:  python rag_chatbot.py            (builds the index, runs eval, starts chat)
# Quick test: python rag_chatbot.py --max-articles 200
# Just the index: python rag_chatbot.py --build-only

import argparse
import json
import os
import pickle
import random
import re
import sys
import time

import faiss
import numpy as np

ARTICLES_DIR = "artifacts"
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"

# A few general-knowledge questions with the article they should point to.
# Only counted when that article is actually in the loaded sample.
GENERAL_QA = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the chemical symbol for gold?", "Gold"),
    ("In what year did humans first land on the Moon?", "Apollo 11"),
]


def load_articles(max_articles, parquet_path=None):
    if parquet_path:
        # Offline path: read the first rows of a local parquet shard
        # (same schema as the HF dataset: title + text).
        import pandas as pd

        print(f"Loading first {max_articles} articles from {parquet_path}...")
        df = pd.read_parquet(parquet_path, columns=["title", "text"])
        articles = []
        for row in df.itertuples(index=False):
            title = (row.title or "").strip()
            text = (row.text or "").strip()
            if not title or len(text.split()) < 50:
                continue  # skip stubs and empty pages
            articles.append({"title": title, "text": text})
            if len(articles) >= max_articles:
                break
        print(f"Loaded {len(articles)} articles.")
        return articles

    # Stream the dataset so we only download the first few thousand
    # articles instead of the whole 6M-article dump.
    from datasets import load_dataset

    print(f"Loading first {max_articles} articles from {DATASET} ({CONFIG})...")
    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    articles = []
    for row in ds:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not title or len(text.split()) < 50:
            continue  # skip stubs and empty pages
        articles.append({"title": title, "text": text})
        if len(articles) >= max_articles:
            break
    print(f"Loaded {len(articles)} articles.")
    return articles


def chunk_text(text, chunk_words=120, overlap=30):
    # Split into word chunks with a bit of overlap so nothing
    # important gets cut off at a boundary.
    words = text.split()
    chunks = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + chunk_words])
        if len(piece.split()) >= 30:  # skip tiny leftovers
            chunks.append(piece)
        if start + chunk_words >= len(words):
            break
    return chunks


def build_chunks(articles, chunk_words=200, overlap=40, max_words=600):
    # Only the first few hundred words of each article: the lead
    # section holds the facts a chatbot gets asked about, and it keeps
    # the build quick enough to run on a laptop CPU.
    chunks = []
    for art in articles:
        text = " ".join(art["text"].split()[:max_words])
        for piece in chunk_text(text, chunk_words, overlap):
            chunks.append({"title": art["title"], "text": piece})
    return chunks


def build_index(chunks, model_name):
    # Embed every chunk, then put the vectors in a FAISS index.
    # Cosine similarity via inner product on normalised vectors.
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    t0 = time.time()
    vectors = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")
    print(f"Embedded in {time.time() - t0:.0f}s.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, model


def save_index(index, chunks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index + {len(chunks)} chunks to {out_dir}/")


def load_index(out_dir, model_name):
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(model_name)
    return index, chunks, model


def retrieve(index, chunks, model, question, top_k=5):
    q = model.encode([question], normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q, top_k)
    hits = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        hits.append({
            "title": chunks[i]["title"],
            "text": chunks[i]["text"],
            "score": float(score),
        })
    return hits


def answer(index, chunks, model, question, top_k=5):
    hits = retrieve(index, chunks, model, question, top_k)
    if not hits:
        return "I couldn't find anything about that in my articles.", hits
    best = hits[0]
    reply = best["text"]
    return reply, hits


def evaluate(index, chunks, model, out_dir, seed=7):
    # Two checks, both grounded in the articles we actually loaded:
    # 1) general questions, counted only if the expected article is here
    # 2) "tell me about X" for 10 articles picked at random from the sample
    rng = random.Random(seed)
    titles = {c["title"] for c in chunks}

    results = {}
    for k in (1, 3, 5, 10):
        results[k] = {"asked": 0, "hit": 0}

    def check(question, expected_title):
        hits = retrieve(index, chunks, model, question, top_k=10)
        got = [h["title"] for h in hits]
        for k in (1, 3, 5, 10):
            results[k]["asked"] += 1
            if expected_title in got[:k]:
                results[k]["hit"] += 1

    general_counted = 0
    for question, expected in GENERAL_QA:
        if expected in titles:
            check(question, expected)
            general_counted += 1
    print(f"General questions counted: {general_counted}/{len(GENERAL_QA)} "
          f"(rest skipped — article not in sample)")

    sampled = rng.sample(sorted(titles), min(10, len(titles)))
    for title in sampled:
        check(f"What is {title}?", title)
    print(f"Sampled-article questions: {len(sampled)}")

    print("\nRetrieval recall (expected article in top-k):")
    recall = {}
    for k in (1, 3, 5, 10):
        r = results[k]["hit"] / max(results[k]["asked"], 1)
        recall[f"recall@{k}"] = round(r, 3)
        print(f"  Recall@{k}: {r:.1%}  ({results[k]['hit']}/{results[k]['asked']})")

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump({
            "general_questions_counted": general_counted,
            "sampled_articles": sampled,
            "recall": recall,
            "n_chunks": len(chunks),
        }, f, indent=2)
    return recall


def plot_recall(recall, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [1, 3, 5, 10]
    vals = [recall[f"recall@{k}"] * 100 for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f"Top {k}" for k in ks], vals, color="#2563eb",
                  edgecolor="#1e40af")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Recall (%)")
    ax.set_title("RAG retrieval recall — expected article found in top-k")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_recall.png"), dpi=150)
    print("Saved", os.path.join(out_dir, "retrieval_recall.png"))


def chat_loop(index, chunks, model, top_k=5):
    print("\nChat ready — ask anything about the loaded articles.")
    print("Type /sources to see where an answer came from, /quit to stop.\n")
    last_hits = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            break
        if q.lower() == "/sources":
            for h in last_hits:
                print(f"  - {h['title']} (score {h['score']:.3f})")
            continue
        reply, last_hits = answer(index, chunks, model, q, top_k)
        print(f"\nBot: {reply}")
        print(f"[from: {last_hits[0]['title']}]\n")


def main():
    ap = argparse.ArgumentParser(description="RAG chatbot over Wikipedia")
    ap.add_argument("--max-articles", type=int, default=5000)
    ap.add_argument("--chunk-words", type=int, default=200)
    ap.add_argument("--chunk-overlap", type=int, default=40)
    ap.add_argument("--max-words-per-article", type=int, default=600,
                    help="only the first N words of each article are indexed")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--parquet", default=None,
                    help="local parquet shard of the dataset (offline runs)")
    ap.add_argument("--out-dir", default=ARTICLES_DIR)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-chat", action="store_true")
    args = ap.parse_args()

    index_path = os.path.join(args.out_dir, "faiss.index")
    if args.eval_only and os.path.exists(index_path):
        index, chunks, model = load_index(args.out_dir, args.model)
    else:
        articles = load_articles(args.max_articles, args.parquet)
        chunks = build_chunks(articles, args.chunk_words,
                              args.chunk_overlap, args.max_words_per_article)
        print(f"Built {len(chunks)} chunks from {len(articles)} articles.")
        index, model = build_index(chunks, args.model)
        save_index(index, chunks, args.out_dir)
        with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
            json.dump({
                "dataset": DATASET, "config": CONFIG,
                "n_articles": len(articles), "n_chunks": len(chunks),
                "embedding_model": args.model,
                "chunk_words": args.chunk_words,
                "chunk_overlap": args.chunk_overlap,
                "max_words_per_article": args.max_words_per_article,
            }, f, indent=2)

    if args.build_only:
        return

    recall = evaluate(index, chunks, model, args.out_dir)
    plot_recall(recall, args.out_dir)

    # A few demo questions so the run log shows real answers.
    print("\nDemo questions:")
    for q in ["What is the capital of France?",
              "Who wrote Romeo and Juliet?",
              "What planet is known as the Red Planet?"]:
        reply, hits = answer(index, chunks, model, q, args.top_k)
        print(f"\nQ: {q}\nA: {reply[:400]}")
        print(f"[from: {hits[0]['title']}]")

    if not args.no_chat:
        chat_loop(index, chunks, model, args.top_k)


if __name__ == "__main__":
    main()
