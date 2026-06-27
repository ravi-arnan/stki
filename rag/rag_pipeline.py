"""Pipeline RAG (IndoBERT + GPT) hukum pajak.

Modul ini berisi logika inti RAG agar dapat dipakai ulang oleh notebook
`STKI_RAG_LLM.ipynb` maupun frontend chat `app.py`:

    load_index()        -> ekstraksi PDF, chunking, embedding korpus (dengan cache)
    retrieve(query)     -> top-k passage paling relevan (Dense VSM / cosine)
    jawab(query)        -> jawaban GPT berbahasa Indonesia + referensi dokumen

Kunci API dibaca dari berkas `.env` (variabel OPENROUTER_API_KEY) dan tidak
boleh di-commit.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# Batasi thread BLAS sebelum import torch agar mesin tetap responsif (CPU tidak 100%).
_N_THREAD = "4"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _N_THREAD)

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))

# ---- Konstanta ----
KORPUS_DIR        = BASE_DIR / "corpus_pajak"
CACHE_PATH        = BASE_DIR / "hf_cache" / "rag_index_v2.npz"   # cache embedding korpus (v2: per-halaman)
MODEL_EMBED       = "firqaaa/indo-sentence-bert-base"          # IndoBERT (Sentence-BERT)
MODEL_LLM         = "openai/gpt-4o-mini"                        # GPT via OpenRouter
TOP_K             = 5
KALIMAT_PER_CHUNK = 4
MAX_KATA_CHUNK    = 160
MIN_CHARS         = 40
WORDY_MIN         = 0.60

# nama_pdf_asli: (doc_id, label sumber, nama_file tampilan)
PETA_DOKUMEN = {
    "85uu012.pdf":                        ("D1",  "UU 12/1985 - PBB",                         "D1_UU_PBB_1985.pdf"),
    "UU Nomor 12 Tahun 1985.pdf":         ("D2",  "UU 12/1985 - PBB (salinan)",               "D2_UU_PBB_1985.pdf"),
    "2024pmkeuangan085.pdf":              ("D3",  "PMK 85/2024 - Penilaian NJOP PBB-P2",      "D3_PMK_85_2024_NJOP.pdf"),
    "234~PMK.03~2022Per.pdf":             ("D4",  "PMK 234/2022 - PBB pertambangan/kehutanan","D4_PMK_234_2022_PBB.pdf"),
    "PAJA323304-M1.pdf":                  ("D5",  "Modul PBB Universitas Terbuka",            "D5_Modul_PBB_UT.pdf"),
    "Permendagri Nomor 7 Tahun 2025.pdf": ("D6",  "Permendagri 7/2025 - PKB & BBN-KB 2025",   "D6_Permendagri_7_2025_PKB.pdf"),
    "Permendagri No 8 Tahun 2024_OCR.pdf":("D7",  "Permendagri 8/2024 - PKB & BBN-KB 2024",   "D7_Permendagri_8_2024_PKB.pdf"),
    "Permendagri Nomor 6 Tahun 2023.pdf": ("D8",  "Permendagri 6/2023 - PKB & BBN-KB 2023",   "D8_Permendagri_6_2023_PKB.pdf"),
    "2024pmkeuangan008.pdf":              ("D9",  "PMK 8/2024 - PPN Kendaraan Listrik (EV)",  "D9_PMK_8_2024_EV.pdf"),
    "5~PMK.010~2022Per.pdf":              ("D10", "PMK 5/2022 - PPnBM Kendaraan Bermotor",     "D10_PMK_5_2022_PPnBM.pdf"),
}

SISTEM_PROMPT = (
    "Anda asisten hukum pajak Indonesia. Jawab pertanyaan HANYA berdasarkan KONTEKS dokumen "
    "yang diberikan. Gunakan Bahasa Indonesia yang jelas dan ringkas. Selalu sertakan sitasi "
    "dokumen sumber dalam format [D#] pada bagian fakta yang relevan. Jika informasi tidak ada "
    "di dalam konteks, katakan dengan jujur bahwa informasi tidak ditemukan pada dokumen. "
    "Jangan mengarang angka atau pasal."
)


# --------------------------------------------------------------------------- #
# Ekstraksi & chunking
# --------------------------------------------------------------------------- #
def ekstrak_teks(path: Path) -> str:
    import pdfplumber
    bagian = []
    with pdfplumber.open(path) as pdf:
        for halaman in pdf.pages:
            bagian.append(halaman.extract_text() or "")
    return "\n".join(bagian)


def bersihkan(teks: str) -> str:
    teks = re.sub(r"-\s*\n\s*", "", teks)
    teks = re.sub(r"\s*\n\s*", " ", teks)
    teks = re.sub(r"\s{2,}", " ", teks)
    return teks.strip()


def pisah_kalimat(teks: str):
    potongan = re.split(r"(?<=[.;])\s+(?=[A-Z(0-9])", teks)
    return [p.strip() for p in potongan if p.strip()]


def _pecah_kata(teks: str, maks=MAX_KATA_CHUNK):
    kata = teks.split()
    return [" ".join(kata[i:i + maks]) for i in range(0, len(kata), maks)]


def chunk_dokumen(teks: str):
    kalimat = pisah_kalimat(teks)
    chunks, buf, buf_kata = [], [], 0

    def flush():
        if buf:
            c = " ".join(buf).strip()
            if len(c) >= MIN_CHARS:
                chunks.append(c)

    for kal in kalimat:
        nk = len(kal.split())
        if nk > MAX_KATA_CHUNK:
            flush(); buf, buf_kata = [], 0
            chunks.extend(s for s in _pecah_kata(kal) if len(s) >= MIN_CHARS)
            continue
        if len(buf) >= KALIMAT_PER_CHUNK or buf_kata + nk > MAX_KATA_CHUNK:
            flush(); buf, buf_kata = [], 0
        buf.append(kal); buf_kata += nk
    flush()
    return chunks


def rasio_kata(teks: str) -> float:
    tok = teks.split()
    if not tok:
        return 0.0
    wordy = [t for t in tok if sum(c.isalpha() for c in t) >= 0.6 * len(t) and len(t) >= 3]
    return len(wordy) / len(tok)


def build_korpus():
    """Ekstrak per-halaman + chunking + filter noise -> list dict passage berlabel.

    Chunking dilakukan per halaman PDF agar tiap passage menyimpan nomor halaman
    asalnya (untuk provenance "hal. X" pada sitasi).
    """
    import pdfplumber
    korpus = []
    urut = {}   # penghitung chunk per dokumen
    for nama_pdf, (doc_id, sumber, nama_file) in PETA_DOKUMEN.items():
        with pdfplumber.open(KORPUS_DIR / nama_pdf) as pdf:
            for halaman, page in enumerate(pdf.pages, start=1):
                teks = bersihkan(page.extract_text() or "")
                if not teks:
                    continue
                for passage in chunk_dokumen(teks):
                    if rasio_kata(passage) < WORDY_MIN:
                        continue
                    j = urut.get(doc_id, 0)
                    urut[doc_id] = j + 1
                    korpus.append({
                        "chunk_id": f"{doc_id}#{j}",
                        "doc_id": doc_id,
                        "sumber": sumber,
                        "nama_file": nama_file,
                        "halaman": halaman,
                        "teks": passage,
                    })
    return korpus


# --------------------------------------------------------------------------- #
# Embedding (IndoBERT Sentence-BERT) - lazy load
# --------------------------------------------------------------------------- #
_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        import torch
        torch.set_num_threads(int(_N_THREAD))
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(MODEL_EMBED, device="cpu")
    return _encoder


def embed(texts, batch_size=16):
    if isinstance(texts, str):
        texts = [texts]
    return get_encoder().encode(texts, batch_size=batch_size,
                                normalize_embeddings=True, show_progress_bar=False)


# --------------------------------------------------------------------------- #
# Index (korpus + matriks embedding) dengan cache disk
# --------------------------------------------------------------------------- #
_korpus = None
_matriks = None


def load_index(rebuild: bool = False, verbose: bool = True):
    """Muat index dari cache; bila belum ada / rebuild, bangun lalu simpan."""
    global _korpus, _matriks
    if _korpus is not None and not rebuild:
        return _korpus, _matriks

    if CACHE_PATH.exists() and not rebuild:
        if verbose:
            print(f"Memuat index dari cache: {CACHE_PATH.name}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        _korpus = list(data["korpus"])
        _matriks = data["matriks"]
        if verbose:
            print(f"Index siap: {len(_korpus)} passage.")
        return _korpus, _matriks

    if verbose:
        print("Membangun index (ekstraksi + embedding, ~2-4 menit di CPU)...")
    _korpus = build_korpus()
    _matriks = embed([c["teks"] for c in _korpus])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, korpus=np.array(_korpus, dtype=object), matriks=_matriks)
    if verbose:
        print(f"Index dibangun & disimpan: {len(_korpus)} passage -> {CACHE_PATH.name}")
    return _korpus, _matriks


# --------------------------------------------------------------------------- #
# Retrieval hybrid (BM25 leksikal + IndoBERT dense), digabung via RRF
# --------------------------------------------------------------------------- #
RRF_K = 60   # konstanta Reciprocal Rank Fusion (semakin besar = perbedaan rank makin halus)

_bm25 = None


def _tokenize(teks: str):
    """Tokenisasi sederhana untuk BM25 (huruf+angka, lowercase).

    Menyatukan singkatan seperti 'PPnBM' -> 'ppnbm' sehingga cocok dengan query.
    Tanda hubung dalam-kata juga dibuang ('BBN-KB' -> 'bbnkb', 'PBB-P2' -> 'pbbp2')
    agar query bertanda hubung cocok dengan istilah yang di PDF tertulis menyatu.
    """
    teks = re.sub(r"(?<=\w)-(?=\w)", "", teks.lower())
    return re.findall(r"\w+", teks)


def get_bm25():
    """Bangun (sekali) indeks BM25 dari label sumber + teks korpus.

    Label `sumber` (mis. "Permendagri 8/2024 - PKB & BBN-KB 2024") ikut diindeks
    agar tahun & jenis dokumen kanonik jadi sinyal leksikal yang kuat. Ini
    membedakan dokumen yang nyaris identik tapi beda tahun (D6/D7/D8): query
    "PKB 2024" cocok ke label D7 (ada 'pkb' DAN '2024') ketimbang D9 (PMK 8/2024,
    ada '2024' tapi tanpa 'pkb').
    """
    global _bm25
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        korpus, _ = load_index(verbose=False)
        _bm25 = BM25Okapi([_tokenize(p["sumber"] + " " + p["teks"]) for p in korpus])
    return _bm25


def retrieve(query: str, top_k: int = TOP_K):
    """Retrieval hybrid: gabungkan ranking dense (cosine IndoBERT) & leksikal (BM25).

    Dense menangkap makna semantik; BM25 menangkap kata/singkatan/angka secara persis
    (mis. 'PPnBM', '2025'). Keduanya digabung dengan Reciprocal Rank Fusion (RRF).
    Skor yang ditampilkan tetap cosine similarity (mudah ditafsirkan).
    """
    korpus, matriks = load_index(verbose=False)

    # 1) skor dense (cosine)
    q = embed(query)[0]
    cos = matriks @ q / (np.linalg.norm(matriks, axis=1) * np.linalg.norm(q) + 1e-9)

    # 2) skor leksikal (BM25)
    lex = np.asarray(get_bm25().get_scores(_tokenize(query)), dtype=float)

    # 3) Reciprocal Rank Fusion: setiap dokumen dapat 1/(K+rank) dari tiap daftar
    rank_cos = np.empty(len(cos), dtype=int)
    rank_cos[np.argsort(-cos)] = np.arange(len(cos))
    rank_lex = np.empty(len(lex), dtype=int)
    rank_lex[np.argsort(-lex)] = np.arange(len(lex))
    rrf = 1.0 / (RRF_K + rank_cos) + 1.0 / (RRF_K + rank_lex)

    urut = np.argsort(-rrf)[:top_k]
    hasil = []
    for rank, idx in enumerate(urut, start=1):
        item = dict(korpus[idx])
        item["skor"] = float(cos[idx])           # tampilkan cosine (interpretable)
        item["skor_bm25"] = float(lex[idx])
        item["rank"] = rank
        hasil.append(item)
    return hasil


_client = None


def get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(BASE_DIR / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY tidak ditemukan. Buat berkas lsi-colab/.env berisi:\n"
                "OPENROUTER_API_KEY=sk-or-v1-..."
            )
        _client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _client


def _bangun_konteks(passages):
    blok = [f"[{p['doc_id']}] (sumber: {p['sumber']})\n{p['teks']}" for p in passages]
    return "\n\n".join(blok)


def jawab(query: str, top_k: int = TOP_K) -> dict:
    """Pipeline RAG lengkap: retrieve -> konteks -> GPT -> jawaban + referensi."""
    passages = retrieve(query, top_k)
    pesan = [
        {"role": "system", "content": SISTEM_PROMPT},
        {"role": "user",
         "content": f"KONTEKS DOKUMEN:\n{_bangun_konteks(passages)}\n\n"
                    f"PERTANYAAN: {query}\n\nJAWABAN:"},
    ]
    resp = get_client().chat.completions.create(
        model=MODEL_LLM, messages=pesan, temperature=0.2, max_tokens=500)
    return {
        "query": query,
        "jawaban": resp.choices[0].message.content.strip(),
        "referensi": passages,
    }


if __name__ == "__main__":
    # Uji cepat dari terminal: python rag_pipeline.py "pertanyaan..."
    import sys
    load_index()
    q = " ".join(sys.argv[1:]) or "Apa objek yang dikenakan Pajak Bumi dan Bangunan?"
    hasil = jawab(q)
    print("\nPERTANYAAN:", hasil["query"])
    print("\nJAWABAN:\n", hasil["jawaban"])
    print("\nREFERENSI:")
    for i, p in enumerate(hasil["referensi"], start=1):
        print(f"  [{i}] {p['nama_file']} - hal. {p['halaman']} (skor {p['skor']:.3f})")
