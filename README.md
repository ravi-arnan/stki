# Sistem Temu Kembali Informasi (STKI)

Implementasi enam metode dasar Information Retrieval untuk korpus berbahasa Indonesia, dengan fokus pada TF-IDF dan Vector Space Model. Setiap metode disertai dasar teori, rumus, dan eksperimen pada jurnal nyata. Proyek ini dikerjakan sebagai tugas UTS mata kuliah STKI.

## Daftar Isi

- [Metode](#metode)
- [Pipeline Preprocessing](#pipeline-preprocessing)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Dependency](#dependency)
- [Korpus Uji](#korpus-uji)
- [Dokumentasi](#dokumentasi)
- [Tim](#tim)
- [Referensi](#referensi)

## Metode

| No | Metode | Inti | Notebook |
|----|--------|------|----------|
| 1 | Boolean Retrieval | Inverted index dengan operasi himpunan (AND, OR, NOT) | `STKI_Boolean_Inverted.ipynb` |
| 2 | Jaccard Similarity | Kemiripan berbasis himpunan term | `STKI_Jaccard.ipynb` |
| 3 | TF-IDF | Pembobotan term (frekuensi lokal dikali kelangkaan global) | `STKI_TFIDF_VSM_Final.ipynb` |
| 4 | Vector Space Model (VSM) | Representasi vektor dengan cosine similarity | `STKI_TFIDF_VSM_Final.ipynb` |
| 5 | Extractive Summarization | Pemilihan kalimat dengan skor TF-IDF tertinggi | `STKI_TFIDF_Summarization.ipynb` |
| 6 | MinHash dan LSH | Estimasi Jaccard probabilistik untuk deteksi dokumen mirip | `STKI_MinHash.ipynb` |

## Pipeline Preprocessing

Seluruh metode menggunakan pipeline lima tahap yang konsisten untuk Bahasa Indonesia:

```
case folding -> cleaning -> tokenizing -> stopword removal -> stemming
```

Stopword removal memakai NLTK (757 stopword Bahasa Indonesia) dan Sastrawi (sekitar 126 entri). Stemming memakai Sastrawi dengan algoritma Nazief dan Adriani (confix-stripping berbasis kamus).

## Struktur Proyek

```
stki-tfidf-vsm/
├── README.md
├── requirements.txt
└── lsi-colab/
    ├── STKI_Boolean_Inverted.ipynb              Bab 1: Boolean Retrieval
    ├── STKI_Jaccard.ipynb                        Bab 2: Jaccard Similarity
    ├── STKI_TFIDF_VSM_Final.ipynb               Bab 3 dan 4: TF-IDF dan VSM
    ├── STKI_TFIDF_Summarization.ipynb            Bab 5: Extractive Summarization
    ├── STKI_TFIDF_Summarization_Penjelasan.ipynb
    ├── STKI_MinHash.ipynb                        Bab 6: MinHash dan LSH
    ├── Jurnal1.pdf, Jurnal2.pdf, Jurnal3.pdf     Korpus uji
    ├── *.png                                      Gambar hasil eksperimen
    └── docs/
        ├── LAPORAN_UTS_STKI.md                   Laporan lengkap
        ├── JAWABAN_SOAL_STKI.md                  Jawaban soal analisis
        ├── laporan_minhash_1.3.md
        └── rumus.md                              Rangkuman rumus MinHash
```

## Prasyarat

- Python 3.12
- pip dan virtualenv

## Instalasi

```bash
# Buat dan aktifkan virtual environment
python3 -m venv lsi-colab/.venv
source lsi-colab/.venv/bin/activate        # Windows: lsi-colab\.venv\Scripts\activate

# Install dependency
pip install -r requirements.txt

# Unduh data NLTK (stopword Bahasa Indonesia)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Cara Menjalankan

Notebook merujuk file PDF dan PNG dengan path relatif, sehingga harus dijalankan dari dalam direktori `lsi-colab/`.

```bash
cd lsi-colab
jupyter lab        # atau: jupyter notebook
```

## Dependency

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.12 | Runtime |
| numpy | 2.4 | Komputasi vektor dan matriks |
| pandas | 3.0 | Tabulasi hasil |
| scikit-learn | 1.8 | Utilitas TF-IDF dan VSM |
| PySastrawi | 1.2 | Stopword dan stemming Bahasa Indonesia |
| nltk | terbaru | Stopword Bahasa Indonesia |
| pdfplumber | 0.11 | Ekstraksi teks PDF jurnal |
| matplotlib | 3.10 | Visualisasi akurasi dan ranking |
| jupyterlab | 4.5 | Lingkungan notebook |

## Korpus Uji

Tiga jurnal bertema perpustakaan dan temu kembali informasi:

1. `Jurnal1.pdf`: OPAC pada Aplikasi CIP, Tridinanti Palembang (Elsadantia, 2023)
2. `Jurnal2.pdf`: Pemanfaatan STKI via OPAC, Poltekkes Sorong (Nanlohy dkk., 2023)
3. `Jurnal3.pdf`: Strategi Shelving Koleksi, SMAN 2 Trenggalek (Fatoni dan Handayani, 2024)

## Dokumentasi

Penjelasan teori lengkap, rumus, dan analisis hasil tiap metode tersedia di [`lsi-colab/docs/LAPORAN_UTS_STKI.md`](lsi-colab/docs/LAPORAN_UTS_STKI.md).

## Tim

| Nama | NIM |
|------|-----|
| Ravi Arnan Irianto | 2305551076 |
| Richad Krishnadana Primantara | 2305551151 |
| Putu Satria Arya Putra | 2305551122 |

## Referensi

1. Manning, C. D., Raghavan, P., dan Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
2. Rajaraman, A., dan Ullman, J. D. (2011). *Mining of Massive Datasets*, Bab 3 (Finding Similar Items). Cambridge University Press.
3. Nazief, B., dan Adriani, M. (1996). *Confix-Stripping: Approach to Stemming Algorithm for Bahasa Indonesia*. Universitas Indonesia.
4. Septiani, D., dan Isabela, I. (2022). Analisis TF-IDF dalam Temu Kembali Informasi pada Dokumen Teks. *SINTESIA*, Vol. 01 No. 2.
