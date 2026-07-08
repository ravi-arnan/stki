# BAB IV
# PEMBAHASAN DAN UJI COBA

## 4.1 Uji Coba Program

Pada uji coba ini dibangun sebuah program Retrieval-Augmented Generation (RAG) untuk
domain **hukum pajak Indonesia** menggunakan bahasa pemrograman Python. Berbeda dengan
implementasi RAG sederhana yang memakai satu contoh basis pengetahuan buatan, sistem ini
menggunakan korpus nyata berupa **sepuluh dokumen hukum pajak (D1–D10)** berformat PDF —
mencakup Undang-Undang, Peraturan Menteri Keuangan (PMK), Peraturan Menteri Dalam Negeri
(Permendagri), dan satu modul akademik — seputar Pajak Bumi dan Bangunan (PBB) serta Pajak
Kendaraan Bermotor (PKB).

Retrieval yang dipakai bukan hanya leksikal atau semantik semata, melainkan **hybrid
retrieval**: skor leksikal **BM25** digabung dengan skor kemiripan semantik dari model
**IndoBERT (Sentence-BERT)** melalui **Reciprocal Rank Fusion (RRF)**. Fase generation
memakai model **GPT (`gpt-4o-mini`)** yang diakses melalui **OpenRouter**. Berikut kode
program beserta penjelasannya, dibagi menjadi empat tahap pipeline.

### Kode Program 4.1 — Persiapan Korpus dan Chunking

```python
KORPUS_DIR        = BASE_DIR / "corpus_pajak"
MODEL_EMBED       = "firqaaa/indo-sentence-bert-base"   # IndoBERT (Sentence-BERT)
KALIMAT_PER_CHUNK = 4
MAX_KATA_CHUNK    = 160
WORDY_MIN         = 0.60

# Setiap PDF dipetakan ke doc_id + label sumber (dipakai LLM untuk sitasi [D#])
PETA_DOKUMEN = {
    "85uu012.pdf":              ("D1", "UU 12/1985 - PBB",                    "D1_UU_PBB_1985.pdf"),
    "2024pmkeuangan085.pdf":    ("D3", "PMK 85/2024 - Penilaian NJOP PBB-P2", "D3_PMK_85_2024_NJOP.pdf"),
    "Permendagri No 7/2025":    ("D6", "Permendagri 7/2025 - PKB & BBN-KB",   "D6_Permendagri_7_2025_PKB.pdf"),
    # ... total sepuluh dokumen D1–D10
}

def build_korpus():
    korpus = []
    for nama_pdf, (doc_id, sumber, nama_file) in PETA_DOKUMEN.items():
        with pdfplumber.open(KORPUS_DIR / nama_pdf) as pdf:
            for halaman, page in enumerate(pdf.pages, start=1):
                teks = bersihkan(page.extract_text() or "")
                for passage in chunk_dokumen(teks):        # 4 kalimat / 160 kata per chunk
                    if rasio_kata(passage) < WORDY_MIN:    # buang baris tabel/angka
                        continue
                    korpus.append({
                        "chunk_id": f"{doc_id}#{...}", "doc_id": doc_id,
                        "sumber": sumber, "halaman": halaman, "teks": passage,
                    })
    return korpus
```
**Kode Program 4.1 Persiapan Korpus dan Chunking**

Kode ini adalah tahap **Indexing (Data Ingestion)**, yakni mempersiapkan basis pengetahuan
sebelum sistem RAG dijalankan. Setiap berkas PDF terlebih dulu dipetakan ke sebuah ID
dokumen (`D1`–`D10`) beserta label sumber yang manusiawi (misalnya `D3` → "PMK 85/2024 -
Penilaian NJOP PBB-P2"). Label sumber inilah yang nantinya dipakai oleh LLM untuk menuliskan
sitasi `[D#]`, sehingga pengguna dapat menelusuri kembali jawaban ke peraturan aslinya.

Karena dokumen hukum berukuran besar dan tujuannya menjawab pertanyaan spesifik (bukan
membandingkan dokumen secara utuh), teks tidak disimpan bulat-bulat melainkan dipecah
menjadi potongan kecil (*chunk/passage*). Chunking dilakukan **per halaman** — agar setiap
passage tetap menyimpan nomor halaman asalnya untuk keperluan provenance — dengan dua batas:
maksimal `KALIMAT_PER_CHUNK = 4` kalimat dan `MAX_KATA_CHUNK = 160` kata (agar tidak
melampaui batas token model, 512 token). Sebagai tahap terakhir, passage yang **didominasi
angka atau kode** (baris tabel tarif, nomor induk, kode sampel) dibuang melalui filter
`rasio_kata >= 0.60` — minimal 60% token harus berupa kata huruf. Tanpa filter ini, ribuan
baris tabel tarif Permendagri akan mendominasi dan mencemari hasil retrieval. Setelah proses
ini, korpus berisi **700 passage** informatif yang siap diindeks.

### Kode Program 4.2 — Pengindeksan Embedding IndoBERT (dengan Cache)

```python
def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_EMBED, device="cpu")   # IndoBERT, lazy-load

def embed(texts, batch_size=16):
    return get_encoder().encode(texts, batch_size=batch_size,
                                normalize_embeddings=True)   # vektor dinormalisasi (|v|=1)

def load_index(rebuild=False):
    if CACHE_PATH.exists() and not rebuild:                  # muat dari cache disk
        data = np.load(CACHE_PATH, allow_pickle=True)
        return list(data["korpus"]), data["matriks"]

    korpus  = build_korpus()
    matriks = embed([c["teks"] for c in korpus])             # (700, 768)
    np.savez(CACHE_PATH, korpus=np.array(korpus, dtype=object), matriks=matriks)
    return korpus, matriks
```
**Kode Program 4.2 Pengindeksan Embedding IndoBERT**

Blok kode ini mengubah tiap passage teks menjadi **vektor padat (dense embedding)**
berdimensi 768 menggunakan encoder `firqaaa/indo-sentence-bert-base`, yaitu IndoBERT yang
di-*fine-tune* dengan objektif Sentence-BERT. Pemilihan model ini disengaja: IndoBERT mentah
hanya dilatih *masked language modeling* sehingga embedding-nya kurang diskriminatif untuk
pencarian, sedangkan versi Sentence-BERT dilatih agar kalimat yang bermakna mirip berdekatan
di ruang vektor. Berbeda dengan TF-IDF/VSM, teks di sini **tidak** melalui stemming maupun
stopword removal, karena transformer memiliki *tokenizer subword* sendiri dan menghapus
imbuhan justru merusak makna yang ditangkap model.

Seluruh passage di-*encode* dengan `normalize_embeddings=True` sehingga setiap vektor
berpanjang 1 dan *cosine similarity* setara dengan *dot product*. Karena proses encoding di
CPU memakan waktu (±2–4 menit), hasilnya berupa matriks `(700, 768)` disimpan ke berkas cache
`rag_index_v2.npz`; pada eksekusi berikutnya index dimuat langsung dari cache sehingga sistem
siap dalam hitungan detik.

### Kode Program 4.3 — Fase Retrieval (Hybrid: BM25 + IndoBERT via RRF)

```python
def retrieve(query, top_k=5):
    korpus, matriks = load_index()

    # 1) skor semantik (dense) — cosine similarity IndoBERT
    q   = embed(query)[0]
    cos = matriks @ q / (np.linalg.norm(matriks, axis=1) * np.linalg.norm(q) + 1e-9)

    # 2) skor leksikal — BM25 (label sumber + teks diindeks agar tahun/jenis jadi sinyal kuat)
    lex = np.asarray(get_bm25().get_scores(_tokenize(query)), dtype=float)

    # 3) Reciprocal Rank Fusion: tiap dokumen dapat 1/(K+rank) dari kedua daftar
    rank_cos = np.empty(len(cos), int); rank_cos[np.argsort(-cos)] = np.arange(len(cos))
    rank_lex = np.empty(len(lex), int); rank_lex[np.argsort(-lex)] = np.arange(len(lex))
    rrf = 1.0/(RRF_K + rank_cos) + 1.0/(RRF_K + rank_lex)    # RRF_K = 60

    urut = np.argsort(-rrf)[:top_k]                          # ambil Top-K passage
    return [ {**korpus[i], "skor": float(cos[i])} for i in urut ]
```
**Kode Program 4.3 Fase Retrieval (Hybrid BM25 + IndoBERT)**

Blok kode ini merupakan **fase Retrieval** yang mengekstraksi passage paling relevan sebelum
diproses LLM. Berbeda dengan pendekatan yang hanya mengandalkan TF-IDF/cosine, sistem ini
menggabungkan **dua sudut pandang**. Pertama, skor **dense (semantik)** — query di-embed
dengan encoder yang sama, lalu dihitung *cosine similarity*-nya terhadap seluruh matriks
korpus; ini menangkap makna dan menangani sinonim/parafrase. Kedua, skor **leksikal (BM25)** —
menangkap kata kunci, singkatan, dan angka secara **persis** (misalnya "PPnBM", "2025"), yang
kerap luput oleh model semantik. Perhatikan bahwa label sumber (mis. "Permendagri 8/2024 - PKB
& BBN-KB") ikut diindeks ke BM25, agar tahun dan jenis dokumen kanonik menjadi sinyal leksikal
yang kuat untuk membedakan dokumen yang nyaris identik namun beda tahun.

Kedua daftar peringkat digabung dengan **Reciprocal Rank Fusion (RRF)**: setiap passage
memperoleh skor `1/(K + rank)` dari masing-masing daftar (dengan `K = 60`), lalu dijumlahkan.
RRF menggabungkan **peringkat**, bukan skor mentah, sehingga tidak perlu menormalkan skala
cosine (0–1) terhadap skala BM25 yang tak terbatas. Sebanyak `TOP_K = 5` passage teratas
diambil sebagai konteks; skor yang ditampilkan tetap *cosine similarity* karena lebih mudah
ditafsirkan.

### Kode Program 4.4 — Fase Generation (Prompt Ketat + GPT via OpenRouter)

```python
SISTEM_PROMPT = (
    "Anda asisten hukum pajak Indonesia. Jawab pertanyaan HANYA berdasarkan KONTEKS dokumen "
    "yang diberikan. Gunakan Bahasa Indonesia yang jelas dan ringkas. Selalu sertakan sitasi "
    "dokumen sumber dalam format [D#]. Jika informasi tidak ada di dalam konteks, katakan "
    "dengan jujur bahwa informasi tidak ditemukan. Jangan mengarang angka atau pasal."
)

def jawab(query, top_k=5):
    passages = retrieve(query, top_k)                        # fase retrieval
    pesan = [
        {"role": "system", "content": SISTEM_PROMPT},
        {"role": "user",   "content": f"KONTEKS DOKUMEN:\n{_bangun_konteks(passages)}\n\n"
                                       f"PERTANYAAN: {query}\n\nJAWABAN:"},
    ]
    resp = get_client().chat.completions.create(             # OpenRouter (OpenAI-compatible)
        model="openai/gpt-4o-mini", messages=pesan, temperature=0.2, max_tokens=500)
    return {"query": query, "jawaban": resp.choices[0].message.content.strip(),
            "referensi": passages}
```
**Kode Program 4.4 Fase Generation (GPT via OpenRouter)**

Blok kode terakhir ini adalah **fase Generation** yang menyusun jawaban akhir. Prosesnya
dimulai dengan teknik **Prompt Engineering**: kelima passage teratas dirangkai menjadi blok
KONTEKS (masing-masing diberi penanda `[D#]` dan label sumbernya), lalu digabung dengan
pertanyaan pengguna. Sebuah *system prompt* yang **ketat** memaksa model untuk menjawab hanya
berdasarkan konteks, menggunakan Bahasa Indonesia, menyertakan sitasi `[D#]`, dan berterus
terang bila informasi tidak ada di konteks. Instruksi terakhir inilah yang menekan
**halusinasi**.

Prompt gabungan dikirim ke model `gpt-4o-mini` melalui OpenRouter (API OpenAI-compatible).
Kunci API dibaca dari berkas `.env` (`OPENROUTER_API_KEY`) yang tidak di-commit demi keamanan.
Suhu (*temperature*) diset rendah (0,2) agar jawaban faktual dan stabil. Keluaran fungsi
`jawab()` mengembalikan tiga bagian: pertanyaan, jawaban LLM, dan daftar dokumen referensi
(label sumber, nomor halaman, dan skor kemiripan) — sehingga setiap klaim dapat ditelusuri
kembali ke passage sumbernya.

## 4.2 Pembahasan Hasil dan Analisis Uji Coba

Berdasarkan pengujian sistem RAG di atas, program menghasilkan keluaran yang dibagi menjadi
dua fase utama: fase **Retrieval** (pencarian passage) dan fase **Generation** (pembuatan
jawaban). Sebagai contoh diuji pertanyaan **"Apa objek yang dikenakan Pajak Bumi dan
Bangunan?"**. Berikut analisis dari masing-masing fase berdasarkan output yang dihasilkan.

```text
User Query: 'Apa objek yang dikenakan Pajak Bumi dan Bangunan?'

=== RETRIEVAL RESULT (TOP-5, hybrid BM25 + IndoBERT via RRF) ===
rank 1 | D5 hal.16 | cos=0.7022 | bm25=14.90 | Modul PBB Universitas Terbuka
rank 2 | D5 hal.10 | cos=0.6876 | bm25=13.08 | Modul PBB Universitas Terbuka
rank 3 | D2 hal.11 | cos=0.6503 | bm25=14.26 | UU 12/1985 - PBB (salinan)
rank 4 | D3 hal.05 | cos=0.6871 | bm25=12.67 | PMK 85/2024 - Penilaian NJOP PBB-P2
rank 5 | D5 hal.10 | cos=0.7217 | bm25=11.90 | Modul PBB Universitas Terbuka
```
**Gambar 4.1 Hasil Retrieval**

Gambar 4.1 menampilkan keluaran fase pencarian. Sistem berhasil menyaring lima passage
teratas dari **700 passage** di dalam korpus. Kelima passage yang terpilih seluruhnya
berkaitan langsung dengan objek/subjek PBB dan berasal dari dokumen yang tepat: Modul PBB
(D5), salinan UU 12/1985 (D2), dan PMK 85/2024 (D3). Skor *cosine similarity* berada pada
rentang tinggi (0,65–0,72), menandakan kedekatan semantik yang kuat antara query dan passage.

```text
Mengirim konteks (Top-5) ke gpt-4o-mini via OpenRouter...

=== JAWABAN AKHIR RAG ===
Objek yang dikenakan Pajak Bumi dan Bangunan (PBB) adalah bumi dan/atau bangunan yang
dimiliki oleh orang pribadi atau badan yang secara nyata mempunyai hak atas bumi, memperoleh
manfaat atas bumi, memiliki bangunan, menguasai bangunan, atau memperoleh manfaat atas
bangunan. Selain itu, semua objek pajak yang digunakan oleh Negara untuk penyelenggaraan
pemerintahan juga dikenakan pajak, termasuk yang dimiliki oleh Pemerintah Pusat dan
Pemerintah Daerah [D2][D5]. Objek PBB juga dapat terdiri dari objek pajak umum dan objek
pajak khusus [D3].
```
**Gambar 4.2 Output Jawaban**

Gambar 4.2 menunjukkan jawaban akhir dari fase generation setelah konteks Top-5 dikirim ke
model `gpt-4o-mini`. Model berhasil merangkum kelima passage menjadi jawaban yang koheren,
faktual, dan berbahasa Indonesia — sekaligus **menyertakan sitasi** `[D2]`, `[D5]`, dan `[D3]`
pada bagian fakta yang relevan. Hal ini membuktikan efektivitas *strict prompt engineering*:
model tidak mengarang, melainkan mengekstraksi dan menyintesis fakta hanya dari konteks yang
diberikan, lengkap dengan jejak sumbernya.

### 4.2.1 Analisis Fase Retrieval (Pencarian Konteks)

Pada proses retrieval, sistem berhasil menyaring lima passage paling relevan dari 700 passage
korpus. Beberapa hal penting dapat dianalisis:

1. **Keunggulan retrieval hybrid.** Passage peringkat pertama (D5, hal. 16) tidak hanya
   unggul secara semantik (cosine 0,7022) tetapi juga secara leksikal (BM25 tertinggi, 14,90).
   Passage D2 (peringkat 3) memiliki cosine relatif lebih rendah (0,6503) namun terangkat oleh
   skor BM25 yang tinggi (14,26) karena mengandung frasa "obyek pajak dikenakan pajak" secara
   harfiah. Inilah nilai tambah RRF: passage yang kuat di salah satu sinyal tetap terangkat,
   sehingga sistem tidak buta terhadap kecocokan kata kunci maupun kecocokan makna.

2. **Diversitas sumber.** Konteks terpilih tidak menumpuk pada satu dokumen, melainkan menyebar
   ke tiga dokumen berbeda (D5, D2, D3). Ini memberi LLM landasan fakta yang lebih lengkap —
   definisi objek/subjek dari modul dan UU, sekaligus rincian jenis objek dari PMK — sehingga
   jawaban yang dihasilkan lebih komprehensif.

3. **Provenance per halaman.** Karena chunking dilakukan per halaman, tiap passage membawa
   nomor halaman asalnya (mis. "D3 hal. 5"), sehingga jawaban benar-benar dapat ditelusuri
   sampai ke halaman peraturan sumbernya.

Hasil ini membuktikan bahwa kombinasi pencarian **leksikal (BM25)** dan **semantik (IndoBERT)**
yang difusikan dengan **RRF** memberikan konteks yang presisi. LLM pada tahap berikutnya hanya
menerima "bahan baku" yang benar-benar relevan.

### 4.2.2 Analisis Fase Generation (Pemrosesan LLM dan Prompt Engineering)

Setelah kelima passage diserahkan ke model `gpt-4o-mini`, sistem menghasilkan jawaban yang
akurat dan tertelusur. Analisis terhadap keluaran ini menunjukkan tiga faktor utama:

1. **Sintesis multi-dokumen.** Model tidak sekadar menyalin satu passage, melainkan
   **menggabungkan** informasi lintas dokumen: definisi hak atas bumi/bangunan (D5), ketentuan
   bahwa objek milik Negara pun dikenakan pajak (D2), serta pembagian objek umum/khusus (D3).
   Kemampuan menyintesis inilah nilai tambah utama LLM dibanding metode retrieval murni
   (TF-IDF, VSM, Boolean, Jaccard, MinHash) yang hanya mengembalikan daftar passage terurut
   tanpa merangkumnya menjadi jawaban.

2. **Kepatuhan pada sitasi dan grounding.** Sesuai instruksi *system prompt*, model menempelkan
   penanda `[D2]`, `[D5]`, dan `[D3]` tepat pada fakta yang bersangkutan. Setiap klaim jawaban
   dapat diverifikasi kembali ke passage sumbernya — inilah wujud **transparansi/traceability**
   yang menjadi keunggulan arsitektur RAG.

3. **Mitigasi halusinasi.** Aturan ketat "jawab HANYA berdasarkan konteks" dan "jangan mengarang
   angka atau pasal", dikombinasikan dengan *temperature* rendah (0,2), memaksa model membatasi
   jawaban pada fakta yang tersedia. Model tidak menambahkan pasal atau angka yang tidak ada di
   konteks; bila informasi memang tidak ditemukan, model diinstruksikan menjawab jujur "tidak
   ditemukan pada dokumen" alih-alih mengarang.

Secara keseluruhan, uji coba ini mendemonstrasikan bahwa arsitektur RAG yang dibangun — dengan
retrieval hybrid (BM25 + IndoBERT via RRF) dan generation ber-*grounding* (GPT dengan prompt
ketat) — mampu menjawab pertanyaan hukum pajak secara **presisi, faktual, dan tertelusur**,
sekaligus memitigasi risiko halusinasi yang menjadi kelemahan utama LLM murni.
