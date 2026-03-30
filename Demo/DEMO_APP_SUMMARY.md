# Demo App Summary (UI-First)

## Goal
Menyediakan satu dashboard visual untuk membandingkan 4 skenario LLIE + YOLO pada ExDark, dengan prioritas awal pada tampilan dan alur presentasi.

## Scope Saat Ini
- UI leaderboard metrik utama
- UI image comparison panel per skenario
- UI section detail per skenario (trend chart + compute summary)
- Semua data masih mock untuk validasi layout

## Scope Berikutnya (Backend)
- Load otomatis artifact per skenario dari folder output
- Sinkronisasi metrik detection, NR-IQA, latency, FLOPs
- Render gambar real: original, enhanced, prediction, ground truth
- Export tabel perbandingan ke CSV

## Struktur App
- Demo/app.py: halaman utama Streamlit
- Demo/requirements.txt: dependency minimal UI
- Demo/README.md: instruksi jalankan

## UX Flow
1. Pilih skenario utama di sidebar
2. Lihat KPI cards dan leaderboard
3. Lihat image comparison grid
4. Lihat detail chart skenario

## Catatan
Backend sengaja belum diaktifkan agar validasi UI bisa cepat tanpa menunggu pipeline data final.
