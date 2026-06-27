"""Frontend chat RAG hukum pajak (Gradio) - versi Hugging Face Spaces.

Di Hugging Face Spaces, kunci API dibaca dari SECRET `OPENROUTER_API_KEY`
(Settings -> Variables and secrets). Tidak ada berkas .env yang di-commit.
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import rag_pipeline as rag

JUDUL = "Asisten Hukum Pajak (RAG: IndoBERT + GPT)"
DESKRIPSI = (
    "Tanya seputar Pajak Bumi & Bangunan (PBB) dan Pajak Kendaraan Bermotor (PKB). "
    "Jawaban disusun oleh GPT **hanya** berdasarkan 10 dokumen hukum pajak (UU/PMK/Permendagri), "
    "lengkap dengan referensi dokumen `[D#]`. Retrieval memakai embedding **IndoBERT** "
    "(Sentence-BERT) + cosine similarity."
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
    try:
        hasil = rag.jawab(message)
    except Exception as e:
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
    rag.load_index()
    print("Memuat & memanaskan model encoder...")
    rag.embed("pemanasan model")
    print("Index & model siap. Menjalankan server Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
