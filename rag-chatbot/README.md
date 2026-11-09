# 🤖 RAG Chatbot over Wikipedia

Ask questions and get answers pulled straight from real Wikipedia articles — a retrieval-augmented generation (RAG) chatbot built from scratch: chunk the text, embed it, search it with FAISS, and answer only from what was retrieved. No made-up answers — every reply quotes the passage it came from.

## Dataset
**Wikipedia (English)** — a sample of the first 5,000 articles from the `20231101.en` dump, streamed via Hugging Face Datasets so only the sample is downloaded, not the full 6M-article dump.
Source: [huggingface.co/datasets/wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia). Downloaded automatically on first run via `datasets.load_dataset` (streaming). No manual download needed.

## What it does
1. **Data** — streams the first 5,000 English Wikipedia articles, skips stubs and empty pages
2. **Chunking** — takes the first 600 words of each article (the lead section holds the facts a chatbot gets asked about, and it keeps the build quick on a laptop CPU) and splits them into ~200-word chunks with a 40-word overlap, so nothing important gets cut off at a boundary
3. **Embedding** — encodes every chunk with `all-MiniLM-L6-v2` (sentence-transformers, 384-dim vectors)
4. **Indexing** — stores the vectors in a FAISS index (inner product on normalised vectors = cosine similarity) and saves it to `artifacts/` for reuse
5. **Answering** — for each question, retrieves the top-k most similar chunks and answers with the best passage, naming its source article
6. **Evaluation** — checks whether the expected article lands in the top-k retrieved results, for general-knowledge questions and for 10 articles sampled from the corpus (see `artifacts/eval.json`)

## Results
**Retrieval recall: Recall@1 81.8% · Recall@3 90.9% · Recall@5 90.9% · Recall@10 90.9%** on an 11-question evaluation set — 10 articles sampled at random from the corpus ("What is \<title\>?") plus 1 general-knowledge question whose article is in the sample. 10 of 11 expected articles land in the top 5 retrieved chunks.

One honest limit: the first 5,000 Wikipedia articles are mostly obscure topics (the sample includes "Shawbury United F.C." and "Tuggeranong Homestead", but not "Paris", "Mars" or "William Shakespeare"), so famous-topic questions get the closest passage the corpus actually has — e.g. "What is the capital of France?" returns a passage about Bordeaux, a regional capital. The bot never hides that: every answer names its source article, so a miss is easy to spot. The 8 general questions in the eval script are only counted when their article is in the sample (1 of 8 here); the rest are skipped, not scored.

## Tech
Python · sentence-transformers · FAISS · Hugging Face Datasets · numpy · matplotlib

## Run it
```bash
pip install -r requirements.txt
python rag_chatbot.py
```

That builds the index, runs the evaluation, shows a few demo questions, then drops into an interactive chat. Type `/sources` to see where the last answer came from, `/quit` to stop.

For a quick smoke test on a small slice of the data:
```bash
python rag_chatbot.py --max-articles 200 --no-chat
```

To rebuild the index only, without eval or chat:
```bash
python rag_chatbot.py --build-only
```

To re-run the evaluation on a saved index:
```bash
python rag_chatbot.py --eval-only --no-chat
```

Offline run from a local copy of the dataset shard (same `title`/`text` schema):
```bash
python rag_chatbot.py --parquet /path/to/train-00000-of-00041.parquet
```

The FAISS index, chunk store, evaluation results and recall chart are written to `artifacts/`. The index (`faiss.index`) and chunk store (`chunks.pkl`) ship in the project ZIP — GitHub's text-only connector can't push binary files, so they live in the ZIP, not the repo.
