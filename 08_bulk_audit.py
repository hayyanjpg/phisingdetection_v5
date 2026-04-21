# ==============================================================================
# MODUL 08: BULK AUDIT (Evaluasi Penuh Dataset)
# ==============================================================================
# Deskripsi:
#   Modul ini melakukan prediksi massal pada seluruh dataset (dataset_raw_seimbang.csv)
#   untuk mengaudit performa model secara menyeluruh.
#
#   Proses:
#     1. Memuat dataset mentah yang belum di-scale.
#     2. Menjalankan penambahan fitur kustom (http_in_domain) dari Modul 03.
#     3. Memuat best_model.pkl dan scaler.pkl.
#     4. Melakukan prediksi secara vektorisasi (sangat cepat).
#     5. Menyimpan hasil perbandingan (Ground Truth vs Prediksi) ke CSV.
#     6. Menampilkan ringkasan error dan top 10 contoh prediksi terburuk
#        (model sangat yakin tapi salah).
# ==============================================================================

import pandas as pd
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Konfigurasi Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "DATASET", "url_features_extracted1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pkl")
FEAT_NAMES_PATH = os.path.join(OUTPUT_DIR, "feature_names.pkl")
OUTPUT_AUDIT_PATH = os.path.join(OUTPUT_DIR, "hasil_audit_total.csv")

# Import Modul 03 untuk Feature Engineering (menambah http_in_domain dll)
import importlib.util
spec3 = importlib.util.spec_from_file_location(
    "feature_engineering", os.path.join(BASE_DIR, "03_feature_engineering.py"))
mod3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(mod3)

def run_bulk_audit():
    print("=" * 70)
    print("MODUL 08: BULK AUDIT MODEL CLASSIFICATION")
    print("=" * 70)

    # 1. Cek Ketersediaan File
    files_needed = [DATASET_PATH, SCALER_PATH, MODEL_PATH, FEAT_NAMES_PATH]
    for b_file in files_needed:
        if not os.path.exists(b_file):
            print(f"[!] Error: File tidak ditemukan: {b_file}")
            print("    Pastikan Anda telah menjalankan pipeline pelatihan (Tahap 1-5).")
            return

    # 2. Muat Data
    print(f"\n[1/5] Memuat dataset dari: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Pre-cleaning darurat agar kompatibel jika menggunakan dataset mentah sebelum tahap 01
    jumlah_sebelum = len(df)
    df = df.dropna()
    if 'ClassLabel' in df.columns:
        df['ClassLabel'] = df['ClassLabel'].astype(int)
    
    if len(df) < jumlah_sebelum:
        print(f"      [!] Ditemukan missing values. Membuang {jumlah_sebelum - len(df)} baris data kotor.")
        
    print(f"      Terdapat {len(df):,} baris data bersih siap proses.")

    # 3. Feature Engineering Tambahan
    # Menggunakan fungsi spesifik untuk menambahkan fitur kustom tanpa harus print full ringkasan 
    # atau generate gambar ulang agar konsol lebih bersih.
    print(f"\n[2/5] Menambahkan fitur kustom (Feature Engineering)...")
    df = mod3.tambah_fitur_kustom(df)

    # 4. Muat Pipeline ML
    print(f"\n[3/5] Memuat model machine learning dan artefak...")
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_NAMES_PATH)

    # 5. Persiapan Matriks Fitur dan Target
    X_raw = df[feature_names].copy()
    y_true = df['ClassLabel'].values # 0=Phishing, 1=Legitimate

    # 6. Scaling & Prediksi (Batch Vectorized)
    print(f"\n[4/5] Memproses scaling dan prediksi (Batch Vectorized)...")
    X_scaled = scaler.transform(X_raw)
    y_pred = model.predict(X_scaled)
    probabilitas = model.predict_proba(X_scaled) 

    # Ekstrak Confidence Score
    # Probabilitas kelas yang diprediksi (menjadi persentase)
    confidence_score = [prob[pred] * 100 for prob, pred in zip(probabilitas, y_pred)]

    # Konversi label numerik ke teks (Opsional, tapi diminta di Output)
    label_map = {0: "Phishing", 1: "Legitimate"}
    
    # 7. Susun DataFrame Output
    print(f"\n[5/5] Menyusun hasil dan mengekspor ke: {OUTPUT_AUDIT_PATH}")
    df_audit = pd.DataFrame({
        'URL': df['URL'],
        'Label_Asli_L': y_true,
        'Label_Asli': [label_map[y] for y in y_true],
        'Label_Prediksi': [label_map[y] for y in y_pred],
        'Confidence_Score': confidence_score,
        'Status_Prediksi': ["Benar" if true == pred else "Salah" for true, pred in zip(y_true, y_pred)]
    })

    # Export
    df_audit.to_csv(OUTPUT_AUDIT_PATH, index=False)
    print(f"      Berhasil dieskspor!")

    # 8. Analisis Summary (Ringkasan Error)
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL AUDIT")
    print("=" * 70)
    
    salah_tebak = df_audit[df_audit['Status_Prediksi'] == "Salah"]
    
    false_positives = salah_tebak[(salah_tebak['Label_Asli_L'] == 1) & (salah_tebak['Label_Prediksi'] == "Phishing")]
    false_negatives = salah_tebak[(salah_tebak['Label_Asli_L'] == 0) & (salah_tebak['Label_Prediksi'] == "Legitimate")]

    total_salah = len(salah_tebak)
    akurasi = 100 - ((total_salah / len(df_audit)) * 100)

    print(f"Total Sampel          : {len(df_audit):,}")
    print(f"Akurasi Keseluruhan    : {akurasi:.2f}%")
    print(f"Total Salah Tebak      : {total_salah:,} sampel")
    print(f"  - False Positives (Legitimate ditebak Phishing) : {len(false_positives):,}")
    print(f"  - False Negatives (Phishing ditebak Legitimate) : {len(false_negatives):,}")

    print("\n" + "-" * 70)
    print("10 CONTOH URL PALING MEMBINGUNGKAN MODEL")
    print(" (Prediksi Salah TAPI dengan Confidence Tinggi)")
    print("-" * 70)
    
    # Kita cari prediksi salah dengan confidence score paling tinggi (sangat yakin tapi salah)
    # Ini yang paling berbahaya/membingungkan
    top_bingung = salah_tebak.sort_values(by='Confidence_Score', ascending=False).head(10)

    for i, (_, row) in enumerate(top_bingung.iterrows(), 1):
        # Truncate URL agar rapi di terminal
        url_short = row['URL'] if len(str(row['URL'])) <= 60 else str(row['URL'])[:57] + "..."
        print(f"{i:>2}. URL: {url_short}")
        print(f"    Asli: {row['Label_Asli']} | Tebak: {row['Label_Prediksi']} | Keyakinan: {row['Confidence_Score']:.2f}%")

    buat_visualisasi_audit_doughnut(df_audit['Status_Prediksi'], df_audit['Label_Asli_L'], df_audit['Label_Prediksi'])


def buat_visualisasi_audit_doughnut(status_prediksi, y_true, y_pred_labels):
    """
    Membuat grafik Doughnut Chart untuk visualisasi komparasi TP, TN, FP, dan FN
    untuk disajikan pada Bab 4 evaluasi massal (Bulk Audit).
    Versi yang Diperbarui: Tampilan yang lebih elegan, rapih, resolusi tinggi, 
    dan tata letak anotasi yang lebih baik.
    """
    print("\n[VISUALISASI] Membuat grafik Doughnut Chart hasil audit massal yang lebih rapi...")
    
    path_gambar = os.path.join(OUTPUT_DIR, "audit_massal_doughnut.png")
    
    # Hitung matriks TP, TN, FP, FN secara manual
    tp_phishing = sum((y_true == 0) & (y_pred_labels == "Phishing"))
    tn_legit = sum((y_true == 1) & (y_pred_labels == "Legitimate"))
    total_benar = tp_phishing + tn_legit
    
    fp_pred_phishing = sum((y_true == 1) & (y_pred_labels == "Phishing"))
    fn_pred_legit = sum((y_true == 0) & (y_pred_labels == "Legitimate"))
    
    total_semua = total_benar + fp_pred_phishing + fn_pred_legit
    akurasi = (total_benar / total_semua) * 100
    
    labels_lengkap = [
        f'Benar (TP & TN)\n{total_benar:,} sampel',
        f'False Positives (FP)\n{fp_pred_phishing:,} sampel',
        f'False Negatives (FN)\n{fn_pred_legit:,} sampel'
    ]
    
    sizes = [total_benar, fp_pred_phishing, fn_pred_legit]
    
    # Warna bernuansa profesional (Hijau untuk benar, Merah bata untuk FP, Kuning jingga untuk FN)
    colors = ['#2E8B57', '#E63946', '#F4A261']
    explode = (0, 0.15, 0.15) # Pemisahan sedikit agar slice yang error (meski kecil) bisa ter-notice
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    def func_pct(pct, allvals):
        if pct > 1.0: 
            return f"{pct:.2f}%"
        return ""

    wedges, texts, autotexts = ax.pie(
        sizes, 
        autopct=lambda pct: func_pct(pct, sizes),
        startangle=140, 
        colors=colors, 
        explode=explode,
        pctdistance=0.80, 
        textprops=dict(color="white", fontsize=13, fontweight='bold'),
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )

    # Anotasi eksternal cerdas untuk persentase < 1% (karena label autopct disembunyikan agar tidak tumpang tindih)
    for i, p in enumerate(wedges):
        pct = sizes[i] / sum(sizes) * 100
        if pct <= 1.0 and sizes[i] > 0:
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            
            # Dinamis offset agar teks FP dan FN yang bersebelahan tidak ketumpuk
            y_text = 1.5 * y
            if i == 1:
                y_text += 0.45 # Angkat label FP ke atas
            elif i == 2:
                y_text -= 0.45 # Turunkan label FN ke bawah

            ax.annotate(
                f"{labels_lengkap[i]}\n({pct:.3f}%)", 
                xy=(x, y), xytext=(1.4 * np.sign(x), y_text),
                horizontalalignment=horizontalalignment, 
                fontsize=11, fontweight='bold', color=colors[i],
                arrowprops=dict(arrowstyle="->", connectionstyle=connectionstyle, color='#888888', lw=1.5)
            )

    ax.set_title("Audit Prediksi Seluruh Dataset Mentah", fontsize=16, fontweight='bold', pad=20)
    
    # Bagian Doughnut (Lingkaran Tengah)
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Teks informasi pada tengah Doughnut
    ax.text(0, 0.15, f"Total Sampel\n{total_semua:,}", size=13, weight='bold', color='#333333', ha='center', va='center')
    ax.text(0, -0.15, f"Akurasi\n{akurasi:.2f}%", size=18, weight='bold', color='#2E8B57', ha='center', va='center')
    
    # Legend (Opsional, tapi membantu) ditaruh di luar agar tidak sumpek
    ax.legend(wedges, labels_lengkap, title="Detail Kategori Hasil Prediksi", 
              loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=11, title_fontsize=12)
    
    ax.axis('equal') 
    fig.tight_layout()
    # Export dengan DPI tinggi (300) agar tajam
    fig.savefig(path_gambar, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  [v] Visualisasi berhasil diperbarui dan disimpan ke: {path_gambar}")

if __name__ == "__main__":
    run_bulk_audit()
