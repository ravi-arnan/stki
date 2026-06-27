"""Frontend chat RAG hukum pajak (Gradio).

Menjalankan:
    cd lsi-colab
    python app.py

Lalu buka URL localhost yang tertera (default http://127.0.0.1:7860).
Butuh berkas .env berisi OPENROUTER_API_KEY. Saat pertama dijalankan, model
IndoBERT (~500MB) diunduh dan index korpus dibangun (~2-4 menit), lalu di-cache.
"""
import os

# Nonaktifkan telemetri Gradio (menghindari hang saat tanpa internet / headless).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import rag_pipeline as rag

JUDUL = "Asisten Hukum Pajak (RAG: IndoBERT + GPT)"
DESKRIPSI = (
    "Tanya seputar Pajak Bumi & Bangunan (PBB) dan Pajak Kendaraan Bermotor (PKB). "
    "Jawaban disusun oleh GPT **hanya** berdasarkan 10 dokumen hukum pajak (UU/PMK/Permendagri), "
    "lengkap dengan referensi dokumen `[D#]`."
)

CONTOH = [
    "Apa objek yang dikenakan Pajak Bumi dan Bangunan?",
    "Bagaimana cara penilaian NJOP untuk PBB-P2?",
    "Apa yang dimaksud dengan Kendaraan Bermotor Listrik Berbasis Baterai?",
    "Kendaraan bermotor apa saja yang dikenai PPnBM?",
]


def _panel_konteks(referensi):
    """Panel lipat (HTML <details>) berisi chunk lengkap yang dipakai sebagai konteks."""
    blok = []
    for i, p in enumerate(referensi, start=1):
        blok.append(
            f"**[{i}] {p['nama_file']} — hal. {p['halaman']} (skor: {p['skor']:.3f})**\n\n"
            f"{p['teks']}"
        )
    isi = "\n\n---\n\n".join(blok)
    return (
        "<details>\n"
        "<summary><b>Lihat chunk yang dipakai sebagai konteks</b></summary>\n\n"
        f"{isi}\n\n"
        "</details>"
    )


def respond(message, history):
    """Fungsi chat: terima pertanyaan -> jalankan RAG -> jawaban + panel konteks."""
    try:
        hasil = rag.jawab(message)
    except Exception as e:  # tampilkan error secara ramah, jangan crash UI
        return f"Terjadi kesalahan: {type(e).__name__}: {e}"

    return f"{hasil['jawaban']}\n\n{_panel_konteks(hasil['referensi'])}"


demo = gr.ChatInterface(
    fn=respond,
    title=JUDUL,
    description=DESKRIPSI,
    examples=CONTOH,
)


if __name__ == "__main__":
    print("Menyiapkan index korpus (sekali di awal)...")
    rag.load_index()                     # bangun/muat index sebelum server jalan
    print("Memuat & memanaskan model encoder (agar query pertama tidak lambat)...")
    rag.embed("pemanasan model")         # paksa load encoder SEKARANG, bukan saat query pertama
    print("Index & model siap. Menjalankan server Gradio...")
    demo.launch()
