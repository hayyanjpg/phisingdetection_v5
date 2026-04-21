# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================
# Skrip ini digunakan untuk menjalankan seluruh tahapan penelitian Anda secara
# berurutan (end-to-end) hanya dengan 1 kali klik. Jika suatu saat dataset Anda
# di-update, Anda cukup menjalankan file ini.
# ==============================================================================

import os
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Urutan modul yang harus dijalankan
SCRIPTS = [
    "01_preprocessing.py",
    "02_balancing.py",
    "03_feature_engineering.py",
    "04_split_scaling.py",
    "05_model_training.py",
    "06_model_evaluation.py"
]

def main():
    print("=" * 70)
    print("  MEMULAI EKSEKUSI PIPELINE MACHINE LEARNING...")
    print("=" * 70)
    
    total_start_time = time.time()

    for script_name in SCRIPTS:
        script_path = os.path.join(BASE_DIR, script_name)
        
        print(f"\n[{script_name}] Memulai eksekusi...")
        
        # Menjalankan modul menggunakan subprocess
        result = subprocess.run(["python", script_path], cwd=BASE_DIR, capture_output=False)
        
        if result.returncode != 0:
            print(f"\n[!] TERJADI KESALAHAN PADA MODUL: {script_name}")
            print(f"    Pipeline dihentikan secara paksa.")
            exit(1)
            
        print(f"[{script_name}] Selesai.")

    total_duration = time.time() - total_start_time

    print("\n" + "=" * 70)
    print(f"  SELURUH TAHAPAN PIPELINE BERHASIL DIJALANKAN 100%!")
    print(f"   Waktu Total : {total_duration/60:.2f} menit")
    print("=" * 70)
    print("   [v] Modul Siap Digunakan:")
    print("      > Untuk Inference   : jalankan `import 07_inference`")
    print("      > Untuk Aplikasi Web: jalankan `streamlit run app.py`")
    print("\n   [v] Rekapitulasi Lokasi Ekspor Tambahan (Sesuai Rencana):")
    print("      - Dataset Transisi  : DATASET/dataset_fitur_kustom.csv")
    print("      - Grafik Balancing  : OUTPUT/grafik_balancing.png")
    print("      - Grafik Kustom     : OUTPUT/distribusi_fitur_kustom.png")
    print("      - Grafik Split Data : OUTPUT/grafik_splitting.png")
    print("      - Grafik Komparasi  : OUTPUT/grafik_komparasi_algoritma.png")
    print("      - Grafik Conf Mtrx  : OUTPUT/grafik_confusion_matrix_svm_tuned.png")
    print("      - Semua File Model  : OUTPUT/*_baseline.pkl & *_tuned.pkl")

if __name__ == "__main__":
    main()
