---
title: Asisten Hukum Pajak RAG
emoji: 📑
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
short_description: RAG IndoBERT + GPT untuk tanya-jawab hukum pajak
---

# Asisten Hukum Pajak — RAG (IndoBERT + GPT)

Sistem Temu Kembali Informasi berbasis **Retrieval-Augmented Generation**. Korpus 10 dokumen
hukum pajak (UU, PMK, Permendagri, Modul UT) bertema Pajak Bumi & Bangunan dan Pajak Kendaraan
Bermotor di-embed dengan **IndoBERT (Sentence-BERT)**; passage paling relevan diambil via cosine
similarity, lalu **GPT (`gpt-4o-mini` via OpenRouter)** menyusun jawaban berbahasa Indonesia
beserta sitasi dokumen `[D#]`.

## Konfigurasi (wajib)

Tambahkan **secret** di Space ini: `Settings → Variables and secrets`:

- `OPENROUTER_API_KEY` = kunci OpenRouter (`sk-or-v1-...`)

> Saat cold start, model IndoBERT (~500 MB) diunduh dan index korpus dibangun (~2–4 menit di CPU),
> lalu di-cache selama Space aktif.

Bagian dari proyek STKI: <https://github.com/ravi-arnan/stki-tfidf-vsm>
