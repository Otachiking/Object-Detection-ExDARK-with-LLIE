"""
Script to automatically zip the Kaggle scenarios output folder for easy downloading.

Usage inside Kaggle Notebook:
    !python src/zip_kaggle_scenarios.py
"""

import shutil
import os

def zip_scenarios(output_dir="/kaggle/working/scenarios", zip_name="/kaggle/working/Scenarios_Output"):
    """Zips the scenarios directory."""
    if not os.path.exists(output_dir):
        print(f"❌ Error: Directory '{output_dir}' not found.")
        print("   Pastikan training atau evaluasi sudah selesai dan memproduksi output di folder tersebut.")
        return

    print(f"📦 Mengarsipkan folder '{output_dir}' menjadi '{zip_name}.zip' ...")
    
    try:
        shutil.make_archive(zip_name, 'zip', output_dir)
        
        # Calculate size
        zip_path = f"{zip_name}.zip"
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"✅ Berhasil! File ZIP tersimpan di: {zip_path} ({size_mb:.2f} MB)")
        print("   Sekarang kamu bisa download file ini langsung dari sidebar Kaggle (di folder /working).")
        
    except Exception as e:
        print(f"❌ Gagal membuat arsip zip: {e}")

if __name__ == "__main__":
    zip_scenarios()
