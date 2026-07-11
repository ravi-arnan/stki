"""Launcher varian: jalankan RAG dengan KALIMAT_PER_CHUNK & port tertentu.

Dipakai untuk membandingkan ukuran chunk secara berdampingan (screenshot).
    KALIMAT=2 PORT=7860 python serve_variant.py
    KALIMAT=6 PORT=7861 python serve_variant.py

Tiap varian memakai cache embedding sendiri (rag_index_k{N}.npz) agar tidak
saling menimpa. Judul halaman menampilkan konfigurasinya.
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

N = int(os.environ.get("KALIMAT", "6"))
PORT = int(os.environ.get("PORT", "7860"))
CACHE_OVERRIDE = os.environ.get("CACHE_FILE")  # opsional: reuse cache yang sudah ada

import gradio as gr
import rag_pipeline as rag

# Override konfigurasi chunk + cache khusus varian ini.
rag.KALIMAT_PER_CHUNK = N
rag.CACHE_PATH = rag.BASE_DIR / "hf_cache" / (CACHE_OVERRIDE or f"rag_index_k{N}.npz")

JUDUL = f"RAG Pajak — {N} KALIMAT / CHUNK  (port {PORT})"
DESKRIPSI = (
    f"**Konfigurasi: {N} kalimat per chunk.** "
    "Tanya seputar PBB & PKB; jawaban GPT berdasarkan 10 dokumen hukum pajak, "
    "dengan referensi `[D#]`. Buka panel konteks untuk melihat chunk + skor cosine."
)
CONTOH = [
    "Apa objek yang dikenakan Pajak Bumi dan Bangunan?",
    "Bagaimana cara penilaian NJOP untuk PBB-P2?",
    "Apa yang dimaksud dengan Kendaraan Bermotor Listrik Berbasis Baterai?",
    "Kendaraan bermotor apa saja yang dikenai PPnBM?",
]


def _panel_konteks(referensi):
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


demo = gr.ChatInterface(fn=respond, title=JUDUL, description=DESKRIPSI, examples=CONTOH)

if __name__ == "__main__":
    print(f"[varian {N} kalimat] menyiapkan index -> {rag.CACHE_PATH.name}")
    rag.load_index()
    rag.embed("pemanasan model")
    print(f"[varian {N} kalimat] siap. Server di port {PORT}")
    demo.launch(server_port=PORT)
