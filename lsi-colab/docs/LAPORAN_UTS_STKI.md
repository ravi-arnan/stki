# LAPORAN UTS — SISTEM TEMU KEMBALI INFORMASI (STKI)

**Topik:** Boolean Retrieval, Jaccard Similarity, TF-IDF, Vector Space Model, Text Summarization, dan MinHash

**Anggota Kelompok:**
- Ravi Arnan Irianto (2305551076)
- Richad Krishnadana Primantara (2305551151)
- Putu Satria Arya Putra (2305551122)

---

## DAFTAR ISI

- Pendahuluan: Preprocessing Bahasa Indonesia (NLTK & Sastrawi)
- Bab 1 — Boolean Retrieval: Indices & Inverted Index
- Bab 2 — Jaccard Similarity
- Bab 3 — TF-IDF (Term Frequency–Inverse Document Frequency)
- Bab 4 — Vector Space Model (VSM)
- Bab 5 — Extractive Text Summarization berbasis TF-IDF
- Bab 6 — MinHash & Locality-Sensitive Hashing (LSH)
- Bab 7 — RAG: Retrieval-Augmented Generation (IndoBERT + LLM)
- Penutup

---

# PENDAHULUAN — Preprocessing Bahasa Indonesia (NLTK & Sastrawi)

Seluruh metode dalam laporan ini menggunakan pipeline preprocessing yang sama untuk Bahasa Indonesia. Pipeline lima tahap dijalankan berurutan: **case folding → cleaning → tokenizing → stopword removal → stemming**.

## Stopword Removal — NLTK & Sastrawi

**Stopword** adalah kata-kata umum yang muncul sangat sering namun tidak membawa makna pembeda (contoh: *yang, di, dan, ini, untuk, dari, pada*). Stopword dibuang karena (1) memperbesar ukuran indeks tanpa menambah informasi diskriminatif, dan (2) menurunkan kualitas pembobotan TF-IDF (term umum mendapat skor tinggi yang menyesatkan).

**NLTK (`nltk.corpus.stopwords`)** menyediakan daftar **757 stopword Bahasa Indonesia** dalam bentuk `set` Python. Cara kerja: setiap token dicocokkan keanggotaannya dalam set; bila token ada di set, token dibuang (`O(1)` lookup).

**Sastrawi (`StopWordRemoverFactory`)** adalah library khusus Bahasa Indonesia. Cara kerja: `stopword_remover.remove(text)` menerima string, melakukan tokenisasi internal, lalu mengembalikan string tanpa stopword. Daftar stopword Sastrawi berbeda (lebih ringkas, ~126 entri) dan dirancang khusus untuk morfologi Bahasa Indonesia.

**Contoh:**
```
Input  : "Mahasiswa belajar sistem temu kembali informasi"
Output : "mahasiswa belajar sistem temu kembali informasi"
(kata "yang", "di", "dan" — bila ada — sudah terbuang)
```

## Stemming — Sastrawi

**Stemming** mereduksi kata berimbuhan menjadi kata dasar. Tujuannya adalah menyamakan token *mahasiswa*, *kemahasiswaan*, *bermahasiswa* menjadi satu term `mahasiswa`, sehingga query *mahasiswa* dapat menemukan dokumen yang menulis *kemahasiswaan*.

**Sastrawi `StemmerFactory`** mengimplementasikan algoritma **Nazief & Adriani** (1996) — algoritma stemming Bahasa Indonesia berbasis kamus. Cara kerja:
1. Cari kata di kamus → bila ditemukan, kembalikan apa adanya.
2. Hapus inflectional suffix (`-lah`, `-kah`, `-tah`, `-pun`, `-ku`, `-mu`, `-nya`).
3. Hapus derivational suffix (`-i`, `-an`, `-kan`).
4. Hapus prefix (`me-`, `ber-`, `pe-`, `ter-`, `di-`, `ke-`, `se-`).
5. Setelah tiap penghapusan, cek kembali ke kamus.

**Contoh:**
```
mahasiswa     -> mahasiswa     (sudah kata dasar)
mengakses     -> akses         (prefix me- + suffix -i)
mengelola     -> kelola        (prefix me-)
menggunakan   -> guna          (prefix meng- + suffix -kan)
perpustakaan  -> pustaka       (prefix per- + suffix -an)
```

## Pipeline Lengkap

```python
def preprocess(text):
    text = text.lower()                         # 1. case folding
    text = re.sub(r'[^a-z\s]', ' ', text)       # 2. cleaning (buang angka & tanda baca)
    tokens = text.split()                       # 3. tokenizing
    tokens = [t for t in tokens if t not in stop_id]   # 4. stopword removal
    tokens = [stemmer.stem(t) for t in tokens]  # 5. stemming
    return tokens
```

**Contoh Output:**
```
Input  : "Mahasiswa belajar sistem temu kembali informasi."
Tahap 1 (case folding) : "mahasiswa belajar sistem temu kembali informasi."
Tahap 2 (cleaning)     : "mahasiswa belajar sistem temu kembali informasi"
Tahap 3 (tokenizing)   : ['mahasiswa', 'belajar', 'sistem', 'temu', 'kembali', 'informasi']
Tahap 4 (stopword)     : ['mahasiswa', 'belajar', 'sistem', 'temu', 'kembali', 'informasi']
Tahap 5 (stemming)     : ['mahasiswa', 'ajar', 'sistem', 'temu', 'informasi']
```

Kata `kembali` adalah stopword dalam daftar NLTK Indonesia sehingga terbuang di tahap 4. Kata `belajar` distemming menjadi `ajar` di tahap 5.

---

# BAB 1 — BOOLEAN RETRIEVAL: INDICES & INVERTED INDEX

## 1.1 Teori Fundamental

**Boolean Retrieval** adalah model temu kembali paling klasik. Dokumen direpresentasikan sebagai himpunan term (set), dan query dinyatakan sebagai ekspresi Boolean menggunakan operator **AND**, **OR**, dan **NOT**. Hasil pencarian bersifat biner: dokumen *relevan* atau *tidak relevan* — tanpa ranking.

**Inverted Index** adalah struktur data inti yang memetakan setiap term ke daftar dokumen yang memuatnya (disebut *posting list*). Inverted index mempercepat pencarian dibanding *forward index* (per dokumen → list term) karena query Boolean berubah menjadi operasi himpunan sederhana pada posting list.

**Rumus:**

$$\text{index}[t] = \{ d \mid t \in d \}$$

**Operasi Boolean atas posting list:**
- $A \text{ AND } B \;=\; A \cap B$ (irisan)
- $A \text{ OR } B \;=\; A \cup B$ (gabungan)
- $\text{NOT } A \;=\; U \setminus A$ (komplemen terhadap koleksi)

## 1.2 Contoh Sederhana

**Koleksi Dokumen:**

| ID | Teks |
|----|------|
| D1 | Mahasiswa belajar sistem temu kembali informasi. |
| D2 | Sistem informasi mengelola data mahasiswa di kampus. |
| D3 | Temu kembali dokumen menggunakan inverted index. |
| D4 | Mahasiswa mengakses dokumen melalui sistem temu kembali. |

**Query Uji:**
1. `mahasiswa AND sistem`
2. `informasi OR dokumen`
3. `mahasiswa AND NOT dokumen`
4. `(sistem OR dokumen) AND mahasiswa`

## 1.3 Hasil Output Python

**Hasil Preprocessing (Tokenisasi + Stopword + Stemming):**
```
D1: ['mahasiswa', 'ajar', 'sistem', 'temu', 'informasi']
D2: ['sistem', 'informasi', 'kelola', 'data', 'mahasiswa', 'kampus']
D3: ['temu', 'dokumen', 'inverted', 'index']
D4: ['mahasiswa', 'akses', 'dokumen', 'sistem', 'temu']
```

**Inverted Index:**
```
ajar         -> ['D1']
akses        -> ['D4']
data         -> ['D2']
dokumen      -> ['D3', 'D4']
index        -> ['D3']
informasi    -> ['D1', 'D2']
inverted     -> ['D3']
kampus       -> ['D2']
kelola       -> ['D2']
mahasiswa    -> ['D1', 'D2', 'D4']
sistem       -> ['D1', 'D2', 'D4']
temu         -> ['D1', 'D3', 'D4']
```

**Eksekusi Query:**

| Query | Hasil |
|-------|-------|
| `mahasiswa AND sistem` | D1, D2, D4 |
| `informasi OR dokumen` | D1, D2, D3, D4 |
| `mahasiswa AND NOT dokumen` | D1, D2 |
| `(sistem OR dokumen) AND mahasiswa` | D1, D2, D4 |

## 1.4 Pembahasan

- *posting list* untuk term `mahasiswa` adalah `{D1, D2, D4}` dan untuk `sistem` adalah `{D1, D2, D4}`. Irisan keduanya menghasilkan `{D1, D2, D4}` — tepat sesuai output.
- Query `mahasiswa AND NOT dokumen` membutuhkan komplemen: `NOT dokumen = {D1, D2, D3, D4} \ {D3, D4} = {D1, D2}`, lalu di-AND dengan `mahasiswa` = `{D1, D2}`.
- Boolean retrieval **tidak memberi ranking** — semua dokumen yang memenuhi query dianggap sama relevannya. Untuk kebutuhan ranking, gunakan Jaccard (Bab 2), TF-IDF (Bab 3), atau VSM (Bab 4).

---

# BAB 2 — JACCARD SIMILARITY

## 2.1 Teori Fundamental

**Jaccard Similarity** mengukur kemiripan antara dua himpunan berdasarkan rasio elemen yang sama (*irisan*) terhadap total elemen unik (*gabungan*). Jaccard murni berbasis himpunan: hanya melihat **apakah** sebuah term muncul, bukan **berapa kali** muncul.

**Rumus:**

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

- Nilai berada pada rentang $[0, 1]$.
- $J = 1$ → himpunan term identik.
- $J = 0$ → tidak ada term yang sama.

**Karakteristik:**
- Cocok untuk *near-duplicate detection* (deteksi dokumen serupa).
- Tidak sensitif terhadap frekuensi → kurang baik untuk ranking yang memperhitungkan kepentingan term.
- Dasar konseptual untuk MinHash (Bab 6).

## 2.2 Contoh Sederhana

**Koleksi & Query:**

| ID | Teks |
|----|------|
| D1 | Mahasiswa belajar sistem temu kembali informasi. |
| D2 | Sistem informasi mengelola data mahasiswa di kampus. |
| D3 | Temu kembali dokumen menggunakan inverted index. |
| D4 | Mahasiswa mengakses dokumen melalui sistem temu kembali. |

**Query:** `sistem temu kembali mahasiswa`

## 2.3 Hasil Output Python

**Hasil Preprocessing (sebagai himpunan):**
```
Query  : {'mahasiswa', 'sistem', 'temu'}
D1     : {'ajar', 'informasi', 'mahasiswa', 'sistem', 'temu'}
D2     : {'data', 'informasi', 'kampus', 'kelola', 'mahasiswa', 'sistem'}
D3     : {'dokumen', 'index', 'inverted', 'temu'}
D4     : {'akses', 'dokumen', 'mahasiswa', 'sistem', 'temu'}
```

**Perhitungan Jaccard Query terhadap Setiap Dokumen:**

| Doc | Irisan (Q ∩ D) | Gabungan (Q ∪ D) | $|Q \cap D|$ | $|Q \cup D|$ | Jaccard |
|-----|----------------|-------------------|--------------|--------------|---------|
| D1 | {mahasiswa, sistem, temu} | {ajar, informasi, mahasiswa, sistem, temu} | 3 | 5 | **0.6000** |
| D2 | {mahasiswa, sistem} | {data, informasi, kampus, kelola, mahasiswa, sistem, temu} | 2 | 7 | 0.2857 |
| D3 | {temu} | {dokumen, index, inverted, mahasiswa, sistem, temu} | 1 | 6 | 0.1667 |
| D4 | {mahasiswa, sistem, temu} | {akses, dokumen, mahasiswa, sistem, temu} | 3 | 5 | **0.6000** |

**Ranking Akhir:**
```
1. D1  -> Jaccard = 0.6000
2. D4  -> Jaccard = 0.6000
3. D2  -> Jaccard = 0.2857
4. D3  -> Jaccard = 0.1667
```

## 2.4 Pembahasan

- D1 dan D4 mendapat skor tertinggi (0.6000) karena keduanya memuat **ketiga term query** (`mahasiswa`, `sistem`, `temu`) dengan ukuran himpunan dokumen relatif kecil (5 elemen).
- D2 lebih rendah meski memuat dua term query, karena himpunan D2 lebih besar (7 elemen) sehingga gabungan membesar dan rasio menurun.
- D3 paling rendah karena hanya berbagi satu term (`temu`) dengan query.
- Jaccard tidak membedakan dokumen yang sama-sama memuat semua term query — D1 dan D4 dianggap setara meski isi sebenarnya berbeda. Kelemahan ini diatasi oleh TF-IDF dan VSM.

---

# BAB 3 — TF-IDF (TERM FREQUENCY–INVERSE DOCUMENT FREQUENCY)

## 3.1 Teori Fundamental

**TF-IDF** adalah skema pembobotan term yang mengukur **seberapa penting** suatu term dalam suatu dokumen relatif terhadap koleksi seluruh dokumen. Ide kuncinya:
- Term yang **sering muncul** di sebuah dokumen → penting bagi dokumen itu (komponen TF).
- Term yang **jarang muncul** di seluruh koleksi → bersifat diskriminatif (komponen IDF).
- Bobot final = TF × IDF: tinggi untuk term yang sering di satu dokumen tetapi langka di koleksi.

**Rumus:**

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}$$

$$\text{IDF}(t) = \log_{10} \left( \frac{N}{\text{DF}(t)} \right)$$

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Keterangan:
- $f_{t,d}$ = frekuensi kemunculan term $t$ dalam dokumen $d$
- $N$ = jumlah total dokumen
- $\text{DF}(t)$ = jumlah dokumen yang memuat term $t$
- Basis logaritma 10 dipakai karena umum digunakan di literatur dan menghasilkan rentang nilai yang mudah dibaca (Septiani & Isabela, 2022).

**Mengapa basis $\log_{10}$?** Penelitian Dwi Septiani & Ica Isabela (SINTESIA, 2022) — yang menjadi sumber korpus untuk Bab 3 dan 4 — menggunakan formula $\text{IDF} = \log_{10}(N/\text{DF})$. Logaritma berfungsi mengompresi rentang nilai sehingga term yang muncul di hampir semua dokumen mendapat IDF mendekati 0, sedangkan term unik mendapat IDF tinggi.

## 3.2 Contoh Sederhana

**Korpus** (4 paragraf jurnal "Analisis TF-IDF dalam Temu Kembali Informasi pada Dokumen Teks", Septiani & Isabela, SINTESIA Vol. 01 No. 2, Maret 2022):

| Doc | Bagian Jurnal |
|-----|---------------|
| D1 | Pendahuluan |
| D2 | Metode Penelitian |
| D3 | Hasil dan Pembahasan |
| D4 | Penutup (Kesimpulan & Saran) |

**Query:** `sistem temu kembali informasi metode term frequency inverse document frequency`

## 3.3 Hasil Output Python

**Statistik Korpus setelah Preprocessing:**
```
Jumlah dokumen           : 4
Jumlah vocabulary        : 124 term unik
Term query (preprocessed): ['sistem', 'temu', 'informasi', 'metode',
                            'term', 'frequency', 'inverse', 'document', 'frequency']
```

**Top-5 Bobot TF-IDF per Dokumen (Keyword Utama):**

```
D1 — Pendahuluan
  1. cocok                  TF-IDF: 0.017202
  2. model                  TF-IDF: 0.012901
  3. query                  TF-IDF: 0.012901
  4. algoritma              TF-IDF: 0.008601
  5. based                  TF-IDF: 0.008601

D2 — Metode Penelitian
  1. bagi                   TF-IDF: 0.017202
  2. nilai                  TF-IDF: 0.017202
  3. total                  TF-IDF: 0.017202
  4. hitung                 TF-IDF: 0.012901
  5. koleksi                TF-IDF: 0.012901

D3 — Hasil dan Pembahasan
  1. frekuensi              TF-IDF: 0.014507
  2. indeks                 TF-IDF: 0.014507
  3. milik                  TF-IDF: 0.014507
  4. koleksi                TF-IDF: 0.010881
  5. akses                  TF-IDF: 0.007254

D4 — Penutup
  1. akurat                 TF-IDF: 0.007526
  2. analisis               TF-IDF: 0.007526
  3. big                    TF-IDF: 0.007526
  4. butuh                  TF-IDF: 0.007526
  5. cakup                  TF-IDF: 0.007526
```

**Ranking Dokumen berdasarkan Skor TF-IDF terhadap Query:**

| Ranking | Dokumen | Bagian | Skor TF-IDF |
|--------:|:-------:|--------|------------:|
| 1 | D4 | Penutup | 0.023426 |
| 2 | D1 | Pendahuluan | 0.021418 |
| 3 | D2 | Metode Penelitian | 0.021418 |
| 4 | D3 | Hasil dan Pembahasan | 0.007526 |

## 3.4 Pembahasan

- Term `cocok`, `model`, `query` menjadi keyword utama D1 (Pendahuluan) karena membahas representasi dokumen dan algoritma pencocokan.
- D2 (Metode) didominasi term komputasi seperti `bagi`, `nilai`, `total`, `hitung` — sesuai isi yang menjabarkan rumus.
- Skor TF-IDF query terhadap D4 tertinggi karena D4 (Penutup) menyebut hampir semua kata query sekaligus dalam paragraf yang relatif pendek → TF-nya lebih tinggi.
- Skor TF-IDF dihitung dengan **menjumlahkan** bobot tiap term query pada dokumen — tidak dinormalisasi. Akibatnya dokumen panjang dengan banyak kemunculan term query bisa mendapat skor besar meski tidak fokus pada query. Untuk mengatasi ini, gunakan VSM (Bab 4).

---

# BAB 4 — VECTOR SPACE MODEL (VSM)

## 4.1 Teori Fundamental

**Vector Space Model (VSM)** merepresentasikan setiap dokumen dan query sebagai **vektor** dalam ruang berdimensi `|V|` (jumlah term dalam vocabulary). Komponen vektor adalah bobot TF-IDF. Kemiripan antara query dan dokumen diukur menggunakan **cosine similarity** — sudut antar vektor.

**Rumus Cosine Similarity:**

$$\cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \cdot \|\vec{d}\|} = \frac{\sum_{t} w_{t,q} \cdot w_{t,d}}{\sqrt{\sum_{t} w_{t,q}^2} \cdot \sqrt{\sum_{t} w_{t,d}^2}}$$

- $\vec{q}, \vec{d}$ = vektor TF-IDF query dan dokumen.
- $\vec{q} \cdot \vec{d}$ = perkalian dot (jumlah perkalian komponen).
- $\|\vec{v}\|$ = norma Euclidean (akar dari jumlah kuadrat komponen).
- Nilai berada pada rentang $[0, 1]$ (untuk bobot non-negatif).

**Keunggulan VSM dibanding TF-IDF Sum:**
- VSM **menormalisasi panjang dokumen** melalui pembagian dengan norma → dokumen panjang tidak otomatis mendapat skor tinggi.
- Pengukuran sudut → dua dokumen dengan komposisi term proporsional dianggap mirip walau panjangnya berbeda.

**Konstanta dalam VSM** — tidak ada konstanta numerik bebas (seperti $k=0.7$). Semua nilai diturunkan dari rumus matematis.

## 4.2 Contoh Sederhana

Korpus, vocabulary, dan query identik dengan Bab 3. Vektor query dan dokumen disusun dari **bobot TF-IDF** untuk masing-masing dari 124 term vocabulary.

## 4.3 Hasil Output Python

**Ranking Dokumen berdasarkan VSM (Cosine Similarity):**

| Ranking | Dokumen | Bagian | Skor VSM |
|--------:|:-------:|--------|---------:|
| 1 | D4 | Penutup | 0.183949 |
| 2 | D2 | Metode Penelitian | 0.155703 |
| 3 | D1 | Pendahuluan | 0.152036 |
| 4 | D3 | Hasil dan Pembahasan | 0.054441 |

**Perbandingan TF-IDF vs VSM:**

| Dokumen | Bagian | Skor TF-IDF | Rank TF-IDF | Skor VSM | Rank VSM |
|:-------:|--------|------------:|------------:|---------:|---------:|
| D4 | Penutup | 0.023426 | **1** | 0.183949 | **1** |
| D1 | Pendahuluan | 0.021418 | 2 | 0.152036 | 3 |
| D2 | Metode Penelitian | 0.021418 | 3 | 0.155703 | 2 |
| D3 | Hasil dan Pembahasan | 0.007526 | 4 | 0.054441 | 4 |

## 4.4 Pembahasan

- D4 menempati peringkat 1 baik di TF-IDF maupun VSM → sangat relevan terhadap query.
- D3 konsisten di peringkat 4 → paling tidak relevan.
- **Perbedaan terjadi pada peringkat 2 dan 3:** TF-IDF menempatkan D1 di atas D2, sedangkan VSM membalik urutan menjadi D2 di atas D1. Penjelasan: D2 (Metode) lebih panjang dan memuat banyak term query secara berulang. TF-IDF Sum yang **tidak dinormalisasi panjang** sedikit dirugikan, sementara cosine VSM memperhitungkan sudut → D2 ternyata "lebih sejajar" dengan vektor query.
- Skor VSM lebih besar (rentang ~0.05–0.18) dibanding TF-IDF Sum (~0.007–0.023) karena pembagian dengan norma menghasilkan rasio, bukan akumulasi.

---

# BAB 5 — EXTRACTIVE TEXT SUMMARIZATION BERBASIS TF-IDF

## 5.1 Teori Fundamental

**Extractive Summarization** menghasilkan ringkasan dengan **memilih** kalimat-kalimat penting dari dokumen asli (bukan menghasilkan kalimat baru seperti *abstractive*). Pendekatan berbasis TF-IDF:
1. Hitung bobot TF-IDF semua term di dokumen.
2. Untuk setiap kalimat, hitung skor sebagai **rata-rata bobot TF-IDF token** dalam kalimat tersebut.
3. Pilih kalimat dengan skor tertinggi sebagai ringkasan.

**Rumus Skor Kalimat:**

$$\text{score}(s) = \frac{1}{|s|} \sum_{t \in s} \text{TF}(t, s) \cdot \text{IDF}(t)$$

- $|s|$ = jumlah token kalimat $s$.
- $\text{TF}(t, s)$ = frekuensi term $t$ dalam kalimat $s$ dibagi panjang $s$.
- $\text{IDF}(t)$ = IDF term $t$ dari koleksi seluruh paragraf.

**Mengapa rata-rata, bukan jumlah?** Pembagian dengan $|s|$ menghindari bias terhadap kalimat panjang. Tanpa normalisasi, kalimat 50 kata otomatis berskor lebih tinggi dari kalimat 10 kata, padahal belum tentu lebih informatif.

## 5.2 Contoh — Jurnal Sederhana

**Studi kasus:** Jurnal "Perkembangan dan Peran OPAC pada Aplikasi CIP (Cerah Informasi Pustaka) untuk Temu Kembali Informasi di Perpustakaan Universitas Tridinanti Palembang" oleh **Betari Ayu Elsadantia** (2023). Dipilih karena strukturnya sederhana (Pendahuluan → Hasil & Pembahasan → Kesimpulan) tanpa banyak sub-bab.

**Pipeline Implementasi:**
1. **Ekstraksi PDF → raw text** menggunakan `pdfplumber`. Posisi horizontal `x0` digunakan untuk mendeteksi indentasi → awal paragraf baru.
2. **RegEx memotong bab**: ambil isi `PENDAHULUAN ... KESIMPULAN`. Abstrak, Kesimpulan, dan Daftar Pustaka otomatis terbuang.
3. **Split paragraf** dengan `\n\n` → setiap paragraf menjadi **sub-dokumen** (D1, D2, ...).
4. Preprocessing 5 tahap (sama seperti Pendahuluan).
5. Hitung TF-IDF manual.
6. Pilih kalimat skor tertinggi tiap paragraf.
7. Gabungkan kalimat-kalimat terpilih menjadi ringkasan akhir.

## 5.3 Hasil Output Python

**Statistik:**
```
Jumlah karakter raw PDF      : 22.811
Jumlah karakter setelah RegEx: 15.844
Jumlah paragraf (sub-dokumen): 26
Vocabulary unik              : 393 term
```

**Top-5 Keyword TF-IDF per Paragraf** (cuplikan 4 paragraf pertama):
```
D1 (Paragraf 1):  teknologi, kembang, perusaan, awal, dahulu
D2 (Paragraf 2):  pustaka, pemustaka, opac, pegawai, layan
D3 (Paragraf 3):  metodologi, kualitatif, kuantitatif, riset, observasi
D4 (Paragraf 4):  cip, opac, pustaka, aplikasi, pemustaka
```

**Ranking 5 Paragraf Teratas terhadap Query** `perpustakaan opac sistem temu kembali informasi aplikasi cip`:

| Rank TF-IDF | Doc | Skor TF-IDF | | Rank VSM | Doc | Skor VSM |
|:-----------:|:---:|:-----------:|---|:--------:|:---:|:--------:|
| 1 | D4 | 0.127299 | | 1 | D4 | 0.424317 |
| 2 | D25 | 0.121268 | | 2 | D25 | 0.322466 |
| 3 | D18 | 0.102487 | | 3 | D24 | 0.207621 |
| 4 | D26 | 0.102377 | | 4 | D26 | 0.176480 |
| 5 | D20 | 0.082108 | | 5 | D20 | 0.120623 |

**Ringkasan Akhir** (kalimat skor TF-IDF tertinggi tiap paragraf, digabung):

> "Awal dari perkembangan teknologi yang sederhana ini perusahaan-perusaan dan lembaga-lembaga informasi mengembangkan teknologi yang dimilikinya. ... Aplikasi CIP (Cerah Informasi Pustaka) ini sangat efektif dan memiliki banyak peran bagi pemustaka dan pustakawan yaitu untuk mempermudah dalam temu kembali suatu informasi, mempermudah dalam mengklasifikasi bahan pustaka."

## 5.4 Pembahasan

- Paragraf D4 mendominasi karena memuat term `cip`, `opac`, `pustaka`, `aplikasi` — semua adalah **inti topik jurnal** yang muncul di query.
- D25 dan D26 adalah paragraf akhir bab Hasil & Pembahasan yang merangkum kesimpulan analisis — wajar berskor tinggi.
- **Ranking TF-IDF dan VSM tidak selalu sama** (mis. D18 vs D24 di peringkat 3). VSM memperhitungkan distribusi seluruh term dokumen, sedangkan TF-IDF Sum hanya menjumlah bobot term query — sehingga dokumen yang term query-nya pekat tetapi pendek bisa mengungguli dokumen besar di TF-IDF, namun kalah di VSM.
- Pemilihan kalimat **per paragraf** (bukan global) menjamin **distribusi topik** ringkasan tetap proporsional dengan struktur dokumen — tiap paragraf diwakili tepat satu kalimat.

---

# BAB 6 — MINHASH & LOCALITY-SENSITIVE HASHING (LSH)

## 6.1 Teori Fundamental

**MinHash** adalah teknik probabilistik untuk **mengestimasi Jaccard similarity** antar himpunan dengan biaya komputasi rendah. Idenya: dua himpunan diubah menjadi *signature* berukuran tetap (vektor pendek), lalu kemiripan diestimasi dari proporsi posisi signature yang sama.

**4 Rumus Inti MinHash:**

**Rumus 1 — Jaccard Similarity (ground truth):**
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Rumus 2 — Hash Function Universal:**
$$h(x) = (a \cdot x + b) \bmod p$$

- $a, b$ = konstanta acak (berbeda untuk tiap fungsi hash).
- $p$ = bilangan prima besar (di implementasi: $p = 4{.}294{.}967{.}311 \approx 2^{32}$).
- $x$ = indeks integer dari shingle dalam universe.

**Rumus 3 — MinHash Signature:**
$$\text{minhash}_k(S) = \min_{x \in S} h_k(x)$$

Untuk setiap fungsi hash $h_k$, ambil nilai **minimum** dari hasil hash atas seluruh elemen $S$. Hasilnya: vektor signature panjang $K$.

**Rumus 4 — Estimasi Jaccard via Signature:**
$$\hat{J}(A, B) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\bigl[\text{minhash}_k(A) = \text{minhash}_k(B)\bigr]$$

Probabilitas dua signature pada posisi yang sama secara matematis **sama dengan** Jaccard similarity sesungguhnya. Semakin besar $K$, error $\to 0$.

## 6.2 Konstanta yang Dipakai & Justifikasinya

Implementasi MinHash menggunakan beberapa konstanta. Sesuai arahan dosen ("kenapa nilai itu dipakai"), berikut justifikasinya:

| Konstanta | Nilai | Justifikasi |
|-----------|------:|-------------|
| `K_SHINGLE` | 2 (bigram) | Bigram lazim di literatur shingling (Rajaraman & Ullman, *Mining of Massive Datasets*). $k=1$ terlalu kasar (semua dokumen mirip), $k \geq 3$ membuat shingle terlalu sparse pada dokumen pendek. |
| `N_HASH` ($K$) | 100 | Trade-off akurasi vs biaya. Dengan $K=100$, standard error estimasi $\approx 1/\sqrt{K} = 0{.}1$ — cukup untuk mendeteksi pasangan dengan Jaccard ≥ 0.3. |
| `PRIME` ($p$) | 4.294.967.311 | Bilangan prima pertama di atas $2^{32}$. Memastikan distribusi hash uniform di seluruh range 32-bit, mendekati asumsi "hash universal". |
| Bands `B` | 10 | Konfigurasi LSH klasik yang membagi 100 hash menjadi 10 band × 10 row. |
| Rows `R` | 10 | Diturunkan dari $N_{HASH}/B = 10$. |
| Threshold LSH | $\approx 0{.}7943$ | Dihitung dari rumus $t \approx (1/B)^{1/R} = (1/10)^{1/10} = 0{.}7943$. **Inilah konstanta ~0.7 yang disebut dalam diskusi.** |

**Mengapa threshold ≈ 0.7943 (dekat 0.7)?**
Threshold LSH menentukan ambang Jaccard di mana pasangan dokumen menjadi "kandidat mirip" dengan probabilitas tinggi. Nilai 0.7–0.8 dipilih karena:
1. **Kasus penggunaan**: deteksi *near-duplicate* (plagiarisme, duplikasi konten) — pasangan dengan Jaccard ≥ 0.7 jelas sangat mirip secara isi.
2. **Trade-off false positive vs false negative**: threshold lebih rendah (mis. 0.3) menangkap pasangan agak mirip tapi memunculkan banyak *false positive*; threshold tinggi (≥0.8) berisiko melewatkan pasangan mirip yang sebenarnya (*false negative*).
3. **Konvensi literatur**: Rajaraman & Ullman (*Mining of Massive Datasets*, Bab 3.4) merekomendasikan threshold di kisaran 0.7–0.85 untuk *near-duplicate detection* sebagai *sweet spot* praktis.
4. **Perhitungan otomatis**: $t = (1/B)^{1/R}$ memberi nilai 0.7943 untuk $B=R=10$ — ini bukan nilai bebas, melainkan turunan matematis dari konfigurasi band.

## 6.3 Contoh

**Korpus**: 3 jurnal bertema perpustakaan & temu kembali informasi.

| Doc | Sumber | Judul Ringkas |
|-----|--------|---------------|
| D1 | Jurnal1.pdf | Perkembangan & Peran OPAC pada Aplikasi CIP — Tridinanti Palembang (Elsadantia, 2023) |
| D2 | Jurnal2.pdf | Pemanfaatan STKI menggunakan OPAC — Poltekkes Sorong (Nanlohy dkk., 2023) |
| D3 | Jurnal3.pdf | Strategi Pustakawan dalam Shelving Koleksi — SMAN 2 Trenggalek (Fatoni & Handayani, FIHRIS 2024) |

**Pipeline**: ekstraksi PDF → potong bab `PENDAHULUAN ... KESIMPULAN` → preprocessing 5 tahap → bigram shingling → MinHash signature ($K=100$) → estimasi Jaccard → LSH banding.

## 6.4 Hasil Output Python

**Statistik Korpus:**
```
                Berkas       Karakter       Bigram unik
D1   Jurnal1.pdf      17.401            925
D2   Jurnal2.pdf       8.445            409
D3   Jurnal3.pdf      12.044            642
Universe shingle (gabungan unik): 1.849
```

**Matriks Jaccard Eksak (Ground Truth, Rumus 1):**
```
        D1      D2      D3
D1  1.0000  0.0504  0.0336
D2  0.0504  1.0000  0.0264
D3  0.0336  0.0264  1.0000
```

**Matriks Estimasi MinHash (N_HASH=100, Rumus 4):**
```
      D1    D2    D3
D1  1.00  0.05  0.01
D2  0.05  1.00  0.03
D3  0.01  0.03  1.00
```

**Perbandingan Eksak vs Estimasi:**

| Pasangan | Jaccard Eksak | Estimasi MinHash | Error Absolut |
|----------|--------------:|-----------------:|--------------:|
| D1 – D2 | 0.0504 | 0.0500 | 0.0004 |
| D1 – D3 | 0.0336 | 0.0100 | 0.0236 |
| D2 – D3 | 0.0264 | 0.0300 | 0.0036 |

```
Rata-rata Error Absolut : 0.0092
Error Maksimum          : 0.0236
```

**Analisis Akurasi terhadap Jumlah Hash $K$:**

| $K$ (N_Hash) | Pasangan | Eksak | Estimasi | Error |
|-------------:|----------|------:|---------:|------:|
| 10 | D1–D2 | 0.0504 | 0.1000 | 0.0496 |
| 50 | D1–D2 | 0.0504 | 0.0400 | 0.0104 |
| 100 | D1–D2 | 0.0504 | 0.0500 | 0.0004 |
| 200 | D1–D2 | 0.0504 | 0.0500 | 0.0004 |
| 500 | D1–D2 | 0.0504 | 0.0480 | 0.0024 |

**Hasil LSH Banding:**
```
Konfigurasi: B=10 band × R=10 baris/band
Threshold  : 0.7943
Kandidat pasangan terdeteksi: 0
```

## 6.5 Pembahasan

- Matriks Jaccard eksak menunjukkan ketiga jurnal berisi **isi yang berbeda**. Pasangan paling mirip adalah D1–D2 dengan Jaccard hanya 0.0504 (~5%), karena keduanya membahas OPAC tetapi pada institusi berbeda.
- Estimasi MinHash dengan $K=100$ sangat akurat: rata-rata error 0.0092. Hal ini membuktikan **Rumus 4** secara empiris.
- Hubungan $K$ vs error: error berkurang $\propto 1/\sqrt{K}$. Pada $K=100$ error sudah cukup kecil; menambah $K$ ke 500 tidak memberi peningkatan signifikan (*diminishing returns*).
- LSH dengan threshold 0.7943 **tidak mendeteksi** pasangan kandidat — konsisten dengan kenyataan bahwa Jaccard tertinggi hanya 0.0504, jauh di bawah threshold. LSH bekerja sebagaimana seharusnya: tidak menghasilkan *false positive*.
- Untuk korpus dengan kemiripan rendah seperti ini, threshold LSH bisa diturunkan dengan konfigurasi $B$ dan $R$ yang berbeda. Misal $B=20, R=5$ memberi threshold $\approx 0{.}5493$; $B=50, R=2$ memberi threshold $\approx 0{.}1414$.

---

# BAB 7 — RAG: RETRIEVAL-AUGMENTED GENERATION (INDOBERT + LLM)

## 7.1 Teori Fundamental

Keenam metode sebelumnya hanya mampu **me-ranking** dokumen terhadap query; pengguna tetap harus membaca sendiri dokumen teratas untuk menemukan jawaban. **RAG (Retrieval-Augmented Generation)** menambahkan sebuah *Large Language Model* (LLM) di ujung pipeline sehingga sistem dapat **menjawab pertanyaan dalam bahasa alami** beserta **referensi dokumen sumber**.

RAG bekerja dalam dua fase. Fase **retrieval** mencari potongan dokumen (*passage*) paling relevan terhadap query — di sini relevansi diukur dengan cosine similarity, sama seperti VSM, tetapi vektornya adalah **embedding semantik** dari transformer **IndoBERT**, bukan bobot TF-IDF. Fase **generation** mengirim passage teratas sebagai konteks ke LLM, yang lalu menyusun jawaban berdasarkan konteks tersebut dan menuliskan sitasi `[D#]`.

Perbedaan kunci dengan VSM/TF-IDF terletak pada representasi. TF-IDF bersifat *sparse* dan mencocokkan kata secara literal, sedangkan embedding IndoBERT bersifat *dense* dan menangkap makna — sehingga query "mobil listrik" dapat menemukan dokumen yang menulis "kendaraan bermotor listrik berbasis baterai" meski tidak ada kata yang sama persis.

## 7.2 Konstanta yang Dipakai & Justifikasinya

- **Encoder `firqaaa/indo-sentence-bert-base`** — IndoBERT yang di-fine-tune Sentence-BERT. IndoBERT mentah (hanya *masked language modeling*) menghasilkan embedding yang kurang diskriminatif untuk pencarian; versi Sentence-BERT membuat passage relevan naik ke peringkat atas.
- **`MODEL_LLM = gpt-4o-mini`** (OpenRouter) — model GPT yang murah dan cukup untuk QA berbasis konteks.
- **`TOP_K = 5`** — jumlah passage konteks; cukup untuk merangkum jawaban tanpa membebani prompt.
- **`KALIMAT_PER_CHUNK = 4`, `MAX_KATA_CHUNK = 160`** — ukuran passage; batas kata menjaga agar tidak melampaui limit 512 token encoder.
- **`WORDY_MIN = 0.60`** — filter noise; passage yang <60% kata-huruf (baris tabel tarif, nomor) dibuang.

## 7.3 Contoh & Alur Implementasi

Detail alur langkah demi langkah dipaparkan terpisah pada dokumen [`laporan_rag_1.0.md`](../../rag/laporan_rag_1.0.md). Korpus bertema **hukum pajak** terdiri atas sepuluh PDF (D1–D10): Undang-Undang, PMK, Permendagri, dan modul akademik mengenai PBB serta Pajak Kendaraan Bermotor. Tiap PDF diekstrak dengan `pdfplumber`, dipecah menjadi passage berukuran bervariasi, difilter dari noise, lalu di-embed menjadi vektor 768 dimensi (~617 passage informatif).

## 7.4 Hasil Output Python

- Pertanyaan **"Bagaimana cara penilaian NJOP untuk PBB-P2?"** dijawab terstruktur: tiga metode penilaian (perbandingan harga, nilai perolehan baru, nilai jual pengganti) serta penilaian massal/individual, dengan sitasi Pasal 3 dan Pasal 15 PMK 85/2024 `[D3]`.
- Pertanyaan **"Apa objek yang dikenakan PBB?"** dijawab "bumi dan/atau bangunan yang dimiliki, dikuasai, dan/atau dimanfaatkan" dengan sitasi `[D3]`.
- Pertanyaan **"Apa yang dimaksud Kendaraan Bermotor Listrik Berbasis Baterai?"** dijawab dari definisi PMK 8/2024 `[D9]`; retrieval memunculkan D9, D7, dan D8 yang sama-sama mendefinisikan KBL.
- Sebagai pembanding, untuk query yang sama, retrieval/VSM saja hanya mengembalikan **daftar passage terurut** tanpa merangkumnya — kontras inilah yang menunjukkan nilai tambah LLM.

## 7.5 Pembahasan

- **Dari meranking ke menjawab.** Inilah lompatan utama: RAG mengubah keluaran sistem dari daftar dokumen menjadi jawaban siap-pakai yang tetap dapat ditelusuri lewat sitasi.
- **Sentence-BERT krusial.** Percobaan awal dengan IndoBERT mentah + mean pooling membuat query PKB tertarik ke dokumen PBB; versi Sentence-BERT memperbaikinya.
- **Grounding menekan halusinasi.** LLM diinstruksikan menjawab hanya dari konteks dan menyatakan jujur bila informasi tidak ada, sehingga jawaban tidak mengarang pasal atau angka.
- **Keterbatasan pada query komparatif.** Pertanyaan "dasar pengenaan PKB tahun 2025 dibanding 2023" dijawab "tidak ditemukan" walau datanya ada. Penyebabnya: token tahun ("2025"/"2023") menyetir embedding ke D5 (Modul PBB) yang padat angka tahun, dan pertanyaan komparatif menuntut dua dokumen sekaligus yang sulit ditangani dense retrieval satu-vektor. Query tanpa token tahun (`dasar pengenaan PKB`) justru menemukan D6/D7/D8 dengan tepat — jadi ini keterbatasan retrieval, bukan bug. Rincian di [`laporan_rag_1.0.md`](../../rag/laporan_rag_1.0.md) Subbab 7.3.4.
- **Opsi perbaikan (lanjutan).** Hybrid **BM25 + dense** (menangkap singkatan/tahun secara persis sekaligus makna — menyambung ke Bab 3–4), *query expansion* singkatan, filter metadata tahun, atau dekomposisi pertanyaan komparatif menjadi sub-query.
- **Keterbatasan lain.** Bila passage relevan tidak masuk `TOP_K`, jawaban bisa kurang lengkap; tabel tarif yang difilter membuat pertanyaan menuntut angka tarif spesifik sulit dijawab; serta pemanggilan LLM bersifat berbayar dan butuh internet.

---

# PENUTUP

Laporan ini telah memaparkan tujuh metode dalam Sistem Temu Kembali Informasi:

1. **Boolean Retrieval** — model klasik berbasis operasi himpunan pada inverted index. Cepat dan sederhana, tetapi tidak memberi ranking.
2. **Jaccard Similarity** — ukuran kemiripan berbasis himpunan, mengabaikan frekuensi. Cocok untuk *near-duplicate detection*.
3. **TF-IDF** — skema pembobotan term yang menggabungkan frekuensi lokal (TF) dan kelangkaan global (IDF).
4. **VSM (Vector Space Model)** — representasi vektor dokumen dengan cosine similarity; menormalisasi panjang dokumen.
5. **Extractive Summarization berbasis TF-IDF** — pemilihan kalimat skor tertinggi per paragraf untuk membentuk ringkasan otomatis.
6. **MinHash & LSH** — estimasi Jaccard secara probabilistik dengan signature ringkas, dilengkapi LSH untuk *similarity search* skala besar.
7. **RAG (IndoBERT + LLM)** — retrieval semantik dengan embedding IndoBERT lalu jawaban kontekstual dari GPT beserta sitasi; satu-satunya metode yang **menjawab** pertanyaan, bukan sekadar meranking.

Metode 1–6 menggunakan **pipeline preprocessing 5 tahap** untuk Bahasa Indonesia: case folding → cleaning → tokenizing → stopword removal (NLTK / Sastrawi) → stemming (Sastrawi/Nazief-Adriani). Metode RAG sengaja **tidak** memakai stemming/stopword removal karena transformer bekerja paling baik pada teks asli.

## Daftar Pustaka

1. Septiani, D., & Isabela, I. (2022). Analisis Term Frequency Inverse Document Frequency (TF-IDF) dalam Temu Kembali Informasi pada Dokumen Teks. *SINTESIA: Jurnal Sistem dan Teknologi Informasi Indonesia*, Vol. 01 No. 2, Maret 2022.
2. Elsadantia, B. A. (2023). Perkembangan dan Peran OPAC pada Aplikasi CIP (Cerah Informasi Pustaka) untuk Temu Kembali Informasi di Perpustakaan Universitas Tridinanti Palembang.
3. Nanlohy, L., Londa, J. W., & Runtuwene, A. (2023). Pemanfaatan Sistem Simpan Temu Kembali Informasi menggunakan Online Public Access Catalog (OPAC) di Perpustakaan Politeknik Kesehatan Kemenkes Sorong. *Jurnal Acta Diurna Komunikasi*, Vol. 5 No. 2.
4. Fatoni, M. R., & Handayani, N. S. (2024). Strategi Pustakawan dalam Pelaksanaan Shelving Koleksi sebagai Upaya Temu Kembali Informasi di SMAN 2 Trenggalek. *FIHRIS: Jurnal Ilmu Perpustakaan dan Informasi*, Vol. 19 No. 1.
5. Rajaraman, A., & Ullman, J. D. (2011). *Mining of Massive Datasets*. Cambridge University Press, Bab 3 (Finding Similar Items).
6. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
7. Nazief, B., & Adriani, M. (1996). Confix-Stripping: Approach to Stemming Algorithm for Bahasa Indonesia. *Internal Publication*, Universitas Indonesia.
