# CLAUDE.md

## Purpose

Sistem Temu Kembali Informasi (STKI) coursework (UTS) implementing seven Information
Retrieval methods over an Indonesian-language corpus, with the core focus on
TF-IDF and the Vector Space Model. The TF-IDF/VSM notebook computes term weights
(local TF times global IDF), builds document vectors, and ranks documents by
cosine similarity against a query. Methods cover Boolean retrieval, Jaccard
similarity, TF-IDF, VSM, extractive summarization, and MinHash/LSH.

The seventh method (in `rag/`) is a RAG (Retrieval-Augmented Generation) pipeline:
hybrid retrieval (BM25 lexical + IndoBERT dense embeddings fused with Reciprocal
Rank Fusion) feeds top-k passages to GPT (via OpenRouter) so the system *answers*
questions with `[D#]` citations instead of only ranking documents.

The corpus for methods 1-6 is three academic PDFs (`Jurnal1.pdf`, `Jurnal2.pdf`,
`Jurnal3.pdf`) on library science and information retrieval. The RAG method uses a
separate corpus of ten Indonesian tax-law PDFs (PBB + PKB) in `rag/corpus_pajak/`.

## Tech Stack

- Python 3.12, Jupyter notebooks (no standalone scripts or package).
- numpy, pandas, scikit-learn (TF-IDF/VSM utilities), matplotlib.
- PySastrawi (Indonesian stopwords and Nazief-Adriani stemming), nltk (Indonesian stopwords).
- pdfplumber (extract text from the journal PDFs).
- RAG only (`rag/`): torch + transformers + sentence-transformers (IndoBERT
  `firqaaa/indo-sentence-bert-base`), rank-bm25 (lexical scoring), openai
  (GPT via OpenRouter), python-dotenv, gradio (chat frontend). Deployed to
  Hugging Face Spaces (`raviarnan/asisten-hukum-pajak-rag`).
- Deps pinned in `requirements.txt`; environment is plain venv plus pip.

## Commands

```bash
# Setup
python3 -m venv lsi-colab/.venv
source lsi-colab/.venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run methods 1-6 (must run from lsi-colab/ so relative PDF/PNG paths resolve)
cd lsi-colab
jupyter lab        # or: jupyter notebook

# Run the RAG chat frontend (method 7) — needs OPENROUTER_API_KEY in rag/.env
echo "OPENROUTER_API_KEY=sk-or-v1-..." > rag/.env
cd rag
python app.py      # Gradio chat at http://127.0.0.1:7860
```

## Structure

- `lsi-colab/` — all notebooks, corpus, and outputs.
  - `STKI_TFIDF_VSM_Final.ipynb` — TF-IDF and VSM (the main artifact).
  - `STKI_Boolean_Inverted.ipynb`, `STKI_Jaccard.ipynb`, `STKI_TFIDF_Summarization.ipynb`, `STKI_MinHash.ipynb` — other methods.
  - `Jurnal{1,2,3}.pdf` — test corpus; `*.png` — experiment result figures.
  - `docs/` — `LAPORAN_UTS_STKI.md` (full report), `JAWABAN_SOAL_STKI.md`, `rumus.md`, `laporan_minhash_1.3.md`.
- `rag/` — method 7 (RAG), self-contained.
  - `rag_pipeline.py` — core module: page-aware chunking, hybrid retrieve (BM25 + IndoBERT dense via RRF), `jawab()` (GPT answer with citations).
  - `app.py` — Gradio chat frontend (local); includes collapsible "chunk yang dipakai sebagai konteks" panel.
  - `STKI_RAG_LLM.ipynb` — notebook walkthrough/report.
  - `corpus_pajak/` — ten tax-law PDFs; `hf_cache/` — embedding cache + model (gitignored); `.env` — `OPENROUTER_API_KEY` (gitignored).
  - `hf-space/` — staging copy deployed to Hugging Face Spaces.
  - `laporan_rag_1.0.md`, `README.md` — RAG implementation report and subsystem guide.
- `requirements.txt`, `README.md` at repo root.

## Conventions and Gotchas

- Notebooks reference PDFs and PNGs by relative path, so always launch Jupyter from inside `lsi-colab/`.
- Notebooks were authored for Colab; some start with `!pip install PySastrawi`, which is redundant once the venv is set up.
- Preprocessing pipeline is consistent across all methods: case folding, cleaning, tokenizing, stopword removal, stemming.
- `.venv/` lives at `lsi-colab/.venv/` and is gitignored.
- All prose, report text, and identifiers are in Bahasa Indonesia.
- RAG: `rag_pipeline.py` resolves paths relative to its own file (`BASE_DIR`), so it
  runs from any cwd; the encoder loads lazily, so `app.py` warms it up at startup.
  The OpenRouter key lives only in gitignored `rag/.env` (local) and as the HF Space
  secret `OPENROUTER_API_KEY` — never hardcode it. The Space is public, so visitors
  consume the owner's OpenRouter credits.
- `rag/hf-space/rag_pipeline.py` is a synced copy of `rag/rag_pipeline.py`; keep both
  in step when editing pipeline logic, then re-upload to the Space.
