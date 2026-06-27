# 7.3 Alur Implementasi RAG (Retrieval-Augmented Generation)

Implementasi RAG dijalankan dalam Jupyter Notebook `STKI_RAG_LLM.ipynb`. Berbeda dengan metode
sebelumnya (TF-IDF, VSM, Boolean, Jaccard, MinHash) yang hanya mampu **me-ranking** dokumen
terhadap sebuah query, RAG menambahkan sebuah *Large Language Model* (LLM) di ujung pipeline
sehingga sistem dapat **menjawab pertanyaan dalam bahasa alami** beserta **referensi dokumen
sumber**. Alur kerja dibagi menjadi tiga tahap berurutan: persiapan korpus dan chunking
(Subbab 7.3.1), pengindeksan embedding dengan IndoBERT (Subbab 7.3.2), serta retrieval dan
generation dengan LLM (Subbab 7.3.3).

Korpus penelitian bertema **hukum pajak** dan terdiri atas sepuluh dokumen (D1–D10) berformat PDF,
mencakup Pajak Bumi dan Bangunan (PBB) serta Pajak Kendaraan Bermotor (PKB). Sumbernya adalah
Undang-Undang, Peraturan Menteri Keuangan (PMK), Peraturan Menteri Dalam Negeri (Permendagri), dan
satu modul akademik Universitas Terbuka. Berbeda dengan MinHash yang memperlakukan tiap jurnal
sebagai satu dokumen utuh, di sini tiap PDF dipecah menjadi banyak *passage* karena dokumen hukum
berukuran besar dan tujuan akhirnya adalah menjawab pertanyaan spesifik, bukan membandingkan
dokumen secara keseluruhan.

## 7.3.1 Persiapan Korpus dan Chunking

Tahap ini menerima sepuluh berkas PDF dan menghasilkan daftar `korpus` berisi *passage* berlabel,
masing-masing dengan metadata `chunk_id`, `doc_id`, `sumber`, dan `teks`. Empat sub-langkah
dijalankan berurutan.

### 7.3.1.1 Pemetaan Dokumen

Setiap berkas PDF dipetakan ke sebuah ID dokumen (`D1`–`D10`) dan label sumber yang manusiawi
(mis. `D3` → "PMK 85/2024 - Penilaian NJOP PBB-P2"). Label sumber inilah yang nantinya dipakai
LLM untuk menuliskan sitasi, sehingga pengguna dapat menelusuri jawaban kembali ke peraturan
aslinya.

### 7.3.1.2 Ekstraksi Teks PDF

Tiap berkas dibaca dengan library `pdfplumber`, lalu seluruh halaman digabung menjadi satu string.
Teks dinormalisasi: kata yang terpenggal tanda hubung di akhir baris disambung kembali, karakter
newline diubah menjadi spasi, dan spasi ganda dirapatkan. Seluruh sepuluh dokumen sudah diverifikasi
dapat diekstrak teksnya (bukan PDF hasil pindai gambar).

### 7.3.1.3 Chunking menjadi Passage

Teks dipecah menjadi *passage* dengan dua batas. Pertama, passage dibentuk dari maksimal
`KALIMAT_PER_CHUNK = 4` kalimat (pemisahan kalimat memakai RegEx pada penanda akhir `.` atau `;`).
Kedua, panjang passage dibatasi `MAX_KATA_CHUNK = 160` kata agar tidak melampaui batas token model
(512 token); kalimat raksasa — seperti baris tabel tarif PKB yang tidak memiliki tanda baca akhir —
dipecah paksa per jendela kata. Passage berukuran bervariasi inilah yang memenuhi tuntutan tugas
akan "dokumen yang banyak dan berbeda panjang".

### 7.3.1.4 Filter Noise

Passage yang **didominasi angka atau kode** (baris tabel tarif, nomor induk pegawai, kode sampel)
dibuang melalui filter `rasio_kata >= WORDY_MIN` dengan `WORDY_MIN = 0.60`, yakni minimal 60% token
harus berupa kata huruf. Tanpa filter ini, ribuan baris tabel tarif Permendagri mendominasi korpus
dan mencemari hasil retrieval. Setelah filter, korpus berisi sekitar **617 passage** informatif.

## 7.3.2 Pengindeksan Embedding dengan IndoBERT

Tahap ini menerima daftar `korpus` dan menghasilkan matriks embedding `matriks_korpus` berukuran
`(jumlah_passage, 768)`.

### 7.3.2.1 Pemilihan Model Encoder

Encoder yang dipakai adalah `firqaaa/indo-sentence-bert-base`, yaitu **IndoBERT
(`indobenchmark/indobert-base-p1`) yang di-fine-tune dengan objektif Sentence-BERT**. Pemilihan ini
disengaja: IndoBERT mentah hanya dilatih *masked language modeling* sehingga embedding hasil
*mean pooling*-nya kurang diskriminatif untuk pencarian semantik. Pada percobaan awal, query tentang
Pajak Kendaraan Bermotor justru tertarik ke dokumen Pajak Bumi dan Bangunan. Versi Sentence-BERT
dilatih khusus agar kalimat yang bermakna mirip berdekatan di ruang vektor, sehingga passage yang
benar-benar relevan naik ke peringkat atas.

### 7.3.2.2 Catatan Preprocessing

Berbeda dengan TF-IDF/VSM, teks **tidak** melewati stemming maupun stopword removal sebelum
di-embed. Transformer memiliki *tokenizer subword* sendiri dan memahami konteks kalimat utuh;
menghapus imbuhan justru merusak makna yang ditangkap model. Passage dikirim ke encoder dalam
bentuk teks asli.

### 7.3.2.3 Komputasi Embedding

Seluruh passage di-*encode* dengan `normalize_embeddings=True` sehingga setiap vektor berpanjang 1
dan *cosine similarity* setara dengan *dot product*. Demi menjaga mesin tetap responsif, jumlah
*thread* PyTorch dibatasi (`torch.set_num_threads(4)` dan variabel lingkungan `OMP/MKL`). Proses ini
memakan sekitar 2–4 menit di CPU dan hanya dilakukan sekali; matriks hasilnya disimpan di memori.

## 7.3.3 Retrieval dan Generation dengan LLM

Tahap ini menerima sebuah query bahasa alami dan menghasilkan jawaban beserta daftar dokumen
referensi.

### 7.3.3.1 Retrieval (Dense Vector Space Model)

Query di-embed dengan encoder yang sama, lalu *cosine similarity* dihitung terhadap seluruh
`matriks_korpus`. Sebanyak `TOP_K = 5` passage dengan skor tertinggi diambil sebagai konteks. Secara
konsep ini identik dengan VSM klasik (cosine similarity), bedanya vektor di sini adalah embedding
semantik IndoBERT, bukan bobot TF-IDF.

### 7.3.3.2 Konstruksi Prompt

Kelima passage teratas dirangkai menjadi blok KONTEKS, masing-masing diberi penanda `[D#]` dan label
sumbernya. Sebuah *system prompt* menginstruksikan LLM untuk: menjawab **hanya** berdasarkan konteks,
menggunakan Bahasa Indonesia, menyertakan sitasi `[D#]`, dan menyatakan dengan jujur bila informasi
tidak ada di konteks. Instruksi terakhir berperan menekan **halusinasi**.

### 7.3.3.3 Pemanggilan LLM

Jawaban dihasilkan oleh model **`gpt-4o-mini`** yang diakses melalui **OpenRouter** (API
OpenAI-compatible). Klien dibangun dengan SDK `openai` memakai `base_url`
`https://openrouter.ai/api/v1`. Kunci API dibaca dari berkas `.env` (variabel
`OPENROUTER_API_KEY`) yang tidak di-commit ke repositori demi keamanan. Suhu (*temperature*) diset
rendah (0,2) agar jawaban faktual dan stabil.

### 7.3.3.4 Output dan Sitasi

Fungsi `jawab(query)` menampilkan tiga bagian: pertanyaan, jawaban LLM, dan daftar dokumen referensi
(chunk_id, label sumber, skor kemiripan, serta cuplikan teks). Dengan demikian setiap klaim pada
jawaban dapat ditelusuri kembali ke passage dan peraturan sumbernya.

Output akhir Subbab 7.3 adalah sebuah pipeline tanya-jawab utuh: untuk pertanyaan seperti
"Bagaimana cara penilaian NJOP untuk PBB-P2?", sistem mengembalikan jawaban terstruktur yang
mengutip Pasal 3 dan Pasal 15 PMK 85/2024 `[D3]`. Sebagai pembanding, retrieval/VSM saja hanya
mengembalikan daftar passage terurut tanpa merangkumnya menjadi jawaban — inilah nilai tambah utama
penerapan LLM di dalam sistem temu kembali informasi.

## 7.3.4 Keterbatasan

Kualitas jawaban bergantung pada kualitas chunking dan retrieval: bila passage relevan tidak masuk
`TOP_K`, jawaban bisa kurang lengkap atau dijawab "tidak ditemukan" meski informasinya sebenarnya
ada di korpus. Batas token encoder (512) memaksa pemotongan passage panjang, dan tabel tarif yang
dibuang oleh filter noise membuat pertanyaan yang menuntut angka tarif spesifik sulit dijawab.
Selain itu, pemanggilan LLM bersifat berbayar dan memerlukan koneksi internet.

### 7.3.4.1 Studi Kasus: Kegagalan pada Query Komparatif

Pertanyaan **"Bagaimana dasar pengenaan PKB tahun 2025 dibanding tahun 2023?"** dijawab "tidak
ditemukan", padahal jawabannya ada di D6 (Permendagri 7/2025) dan D8 (Permendagri 6/2023).
Penelusuran retrieval menunjukkan polanya jelas:

| Query | Top-3 dokumen | Skor |
|-------|---------------|------|
| `dasar pengenaan PKB` | D6, D7, D8 | 0.63 |
| `dasar pengenaan pajak kendaraan bermotor` | D6, D10, D6 | 0.74 |
| `dasar pengenaan PKB tahun 2025` | **D5, D3**, D7 | 0.51 |
| `dasar pengenaan PKB tahun 2025 dibanding tahun 2023` | **D5, D6, D5** | 0.52 |

Akar penyebabnya **bukan** bug, bukan filter noise, dan bukan konten yang hilang — query tanpa
token tahun justru menemukan D6/D7/D8 dengan tepat. Penyebabnya:

1. **Token tahun menyetir embedding.** Begitu "tahun 2025/2023" ditambahkan, vektor query tertarik
   ke D5 (Modul PBB) yang padat angka tahun dan narasi sejarah ("Tahun 1985", "2005", "SEJARAH
   PERKEMBANGAN PBB"). Sentence-BERT memampatkan seluruh kalimat menjadi **satu vektor**, sehingga
   token yang "berisik" secara permukaan dapat mendominasi maksud sebenarnya.
2. **Pertanyaan komparatif/multi-hop.** "2025 vs 2023" menuntut informasi dari **dua dokumen**
   sekaligus; dense retrieval satu-vektor tidak dapat menggabungkan kedua sisi dalam sekali pencarian.
3. **Singkatan.** "PKB" sedikit lebih lemah daripada frasa penuh, meski pengaruhnya kecil.

### 7.3.4.2 Opsi Perbaikan (Pekerjaan Lanjutan)

Keterbatasan di atas dapat diatasi dengan teknik IR berikut (belum diimplementasikan, dicatat
sebagai opsi):

- **Hybrid retrieval (BM25 + dense).** Menggabungkan skor leksikal **BM25** — yang menangkap kata,
  singkatan, dan angka tahun secara **persis** ("PKB", "2025", "2023") — dengan skor cosine IndoBERT
  yang semantik. Pendekatan ini menyambung langsung ke materi TF-IDF/pembobotan term pada Bab 3–4
  dan merupakan solusi paling tepat untuk kasus ini.
- **Query expansion.** Memperluas singkatan sebelum embedding (mis. "PKB" → "Pajak Kendaraan
  Bermotor"), serta opsional menurunkan bobot token tahun yang berdiri sendiri.
- **Filter metadata tahun.** Karena tiap Permendagri memetakan ke satu tahun (D6=2025, D8=2023),
  tahun pada query dapat dipakai untuk menyaring kandidat dokumen sebelum retrieval semantik.
- **Multi-query / decomposition.** Memecah pertanyaan komparatif menjadi dua sub-pertanyaan
  (satu untuk 2025, satu untuk 2023), me-retrieve masing-masing, lalu menggabungkan konteksnya
  sebelum dikirim ke LLM.
