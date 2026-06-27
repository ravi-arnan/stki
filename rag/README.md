# RAG — Asisten Hukum Pajak (IndoBERT + GPT)

Metode ke-7 proyek STKI. Berbeda dengan metode 1-6 yang hanya **me-ranking** dokumen,
subsistem ini **menjawab pertanyaan** dalam Bahasa Indonesia dengan menyertakan sitasi
dokumen sumber `[D#]` — sebuah pipeline **RAG (Retrieval-Augmented Generation)**.

**Demo live (cloud):** <https://huggingface.co/spaces/raviarnan/asisten-hukum-pajak-rag>

## Cara kerja singkat

```
PDF hukum pajak ──► ekstraksi per-halaman ──► chunking (≈4 kalimat / maks 160 kata)
                                                   │  + filter noise (buang tabel tarif)
                                                   ▼
                                    700-an passage berlabel {doc_id, sumber, halaman, teks}
                                                   │
         ┌─────────────────────────────────────────┴───────────────────────────────────┐
         ▼                                                                               ▼
  BM25 (leksikal)                                                        IndoBERT (dense, 768-d)
  cocok kata/singkatan/angka persis                                      cocok makna semantik
  mis. "PPnBM", "2025"                                                   mis. "pajak mobil mewah"
         └───────────────────────────► Reciprocal Rank Fusion (RRF, K=60) ◄─────────────┘
                                                   │
                                       top-5 passage paling relevan
                                                   ▼
                            GPT (gpt-4o-mini via OpenRouter) + system prompt grounding
                                                   ▼
                          Jawaban Bahasa Indonesia + sitasi [D#] + panel chunk konteks
```

Retrieval **hybrid** dipakai karena dense saja gagal pada query yang bergantung pada
token persis (mis. "PPnBM" tidak punya tetangga semantik dekat di IndoBERT umum). BM25
menangkap kecocokan leksikal itu; RRF menggabungkan kedua ranking tanpa perlu menyetel
bobot skala skor yang berbeda.

## Korpus

Sepuluh PDF hukum pajak di `corpus_pajak/` (tema PBB + Pajak Kendaraan Bermotor):

| ID | Sumber |
|----|--------|
| D1 / D2 | UU 12/1985 — PBB (D2 salinan) |
| D3 | PMK 85/2024 — Penilaian NJOP PBB-P2 |
| D4 | PMK 234/2022 — PBB pertambangan/kehutanan |
| D5 | Modul PBB Universitas Terbuka |
| D6 / D7 / D8 | Permendagri PKB & BBN-KB (2025 / 2024 / 2023) |
| D9 | PMK 8/2024 — PPN Kendaraan Listrik (EV) |
| D10 | PMK 5/2022 — PPnBM Kendaraan Bermotor |

## Isi folder

| Berkas | Keterangan |
|--------|------------|
| `rag_pipeline.py` | Modul inti: `build_korpus()`, `embed()`, `load_index()`, `retrieve()` (hybrid), `jawab()` |
| `app.py` | Frontend chat Gradio (lokal), lengkap dengan panel "chunk yang dipakai sebagai konteks" |
| `STKI_RAG_LLM.ipynb` | Notebook penjelasan/laporan (eksperimen bertahap) |
| `laporan_rag_1.0.md` | Laporan alur implementasi + justifikasi parameter |
| `corpus_pajak/` | 10 PDF korpus (tidak di-commit bila besar) |
| `hf_cache/` | Cache embedding (`rag_index_v2.npz`) + model IndoBERT — **gitignored** |
| `.env` | `OPENROUTER_API_KEY=...` — **gitignored**, tidak boleh di-commit |
| `hf-space/` | Salinan staging untuk deploy ke Hugging Face Spaces |

## Menjalankan lokal

```bash
# Dari root repo: pasang dependency (lihat requirements.txt utama)
source lsi-colab/.venv/bin/activate

# Kunci OpenRouter (tidak di-commit)
echo "OPENROUTER_API_KEY=sk-or-v1-..." > rag/.env

cd rag
python app.py        # buka http://127.0.0.1:7860
```

Saat pertama dijalankan, model IndoBERT (~500 MB) diunduh ke `rag/hf_cache/` dan
index embedding korpus dibangun lalu di-cache (`rag_index_v2.npz`). Menjalankan
berikutnya langsung memakai cache. `app.py` melakukan *warm-up* encoder saat startup
agar pesan pertama tidak lambat.

Uji cepat dari terminal tanpa frontend:

```bash
python rag_pipeline.py "Kendaraan bermotor apa saja yang dikenai PPnBM?"
```

## Deploy ke Hugging Face Spaces

`hf-space/` adalah salinan yang di-upload ke Space (`app.py`, `rag_pipeline.py`,
`requirements.txt`, `README.md` metadata, korpus, dan cache). Kunci API disimpan sebagai
**Space secret** `OPENROUTER_API_KEY`, bukan di dalam berkas.

> Saat mengubah logika di `rag_pipeline.py`, sinkronkan juga `hf-space/rag_pipeline.py`
> lalu upload ulang agar versi cloud ikut diperbarui.

## Konfigurasi (konstanta di `rag_pipeline.py`)

| Konstanta | Nilai | Arti |
|-----------|-------|------|
| `MODEL_EMBED` | `firqaaa/indo-sentence-bert-base` | Encoder IndoBERT (Sentence-BERT) |
| `MODEL_LLM` | `openai/gpt-4o-mini` | Generator jawaban (via OpenRouter) |
| `TOP_K` | 5 | Jumlah passage konteks |
| `KALIMAT_PER_CHUNK` / `MAX_KATA_CHUNK` | 4 / 160 | Ukuran chunk |
| `WORDY_MIN` | 0.60 | Ambang filter noise (rasio kata "wajar") |
| `RRF_K` | 60 | Konstanta Reciprocal Rank Fusion |

## Keterbatasan

- Jawaban dibatasi pada isi 10 dokumen; fakta di luar korpus (mis. rincian jenis
  kendaraan pada PP 73/2019) memang tidak tersedia — sistem akan menyatakannya jujur,
  bukan mengarang.
- Query komparatif lintas-tahun (mis. "beda tarif 2023 vs 2025") bergantung pada
  ketersediaan kedua dokumen di top-k.

## Keamanan

Kunci `OPENROUTER_API_KEY` hanya berada di `.env` (lokal, gitignored) dan sebagai Space
secret — **tidak pernah** di-hardcode atau di-commit. Space bersifat publik, sehingga
pengunjung memakai kredit OpenRouter pemilik; disarankan memasang batas kredit di
dashboard OpenRouter atau menjadikan Space privat.
