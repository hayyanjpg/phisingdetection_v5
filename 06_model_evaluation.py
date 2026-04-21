# ==============================================================================
# MODUL 06: EVALUASI MODEL (Bab 3.10)
# ==============================================================================
# Deskripsi:
#   Modul ini khusus untuk membandingkan dan mengevaluasi model yang telah
#   dilatih sebelumnya di Modul 05. 
#   Proses yang dilakukan:
#     1. Memuat seluruh model (baseline dan tuned) untuk ketiga algoritma:
#        Logistic Regression (LR), Random Forest (RF), dan SVM.
#     2. Melakukan prediksi pada data uji (X_test) yang tersimpan di OUTPUT.
#     3. Menghitung metrik performa: Accuracy, Precision, Recall, F1-Score.
#     4. Menampilkan tabel perbandingan yang komprehensif antara model
#        sebelum tuning (baseline) dan sesudah tuning.
#     5. Menentukan model paling stabil/terbaik secara keseluruhan dan 
#        mencetak Classification Report & Confusion Matrix-nya.
# ==============================================================================

import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.inspection import permutation_importance

# --- Konfigurasi Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")

# Definisi nama algoritma
ALGORITHMS = ['lr', 'rf', 'svm']


def muat_data_uji():
    """
    Memuat data uji yang telah melalui proses preprocessing, balancing,
    feature engineering, split, dan scaling (format CSV).

    Returns:
        tuple: (X_test, y_test)
    """
    path_test = os.path.join(OUTPUT_DIR, "X_test_scaled.csv")
    
    if not os.path.exists(path_test):
        raise FileNotFoundError(f"File data uji tidak ditemukan di rute: {path_test}")
    
    df_test = pd.read_csv(path_test)
    y_test = df_test['ClassLabel']
    X_test = df_test.drop('ClassLabel', axis=1)

    print(f"  [v] Data uji berhasil dimuat: {X_test.shape[0]:,} sampel")
    return X_test, y_test


def evaluasi_model_tersimpan(X_test, y_test):
    """
    Melakukan proses load dan prediksi untuk seluruh model baseline dan tuned,
    kemudian menghitung nilai metrik klasifikasinya.

    Returns:
        dict: Hasil evaluasi keseluruhan per model (nama model -> metrics).
    """
    hasil_evaluasi = {}
    
    print("\n[EVALUASI PADA DATA UJI]")
    for algo in ALGORITHMS:
        for tipe in ['baseline', 'tuned']:
            nama_model = f"{algo}_{tipe}"
            path_model = os.path.join(OUTPUT_DIR, f"{nama_model}.pkl")
            
            if not os.path.exists(path_model):
                print(f"  [?] Terlewati: Model {nama_model}.pkl tidak ditemukan.")
                continue

            model = joblib.load(path_model)
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            hasil_evaluasi[nama_model] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'y_pred': y_pred
            }
            print(f"  [v] Ter-evaluasi: {nama_model.upper():<12}")
            
    return hasil_evaluasi


def tampilkan_tabel_perbandingan(hasil_evaluasi):
    """
    Menampilkan tabel perbandingan yang membedakan hasil baseline 
    dengan hasil setelah parameter tuning.
    """
    print(f"\n{'-'*95}")
    print("  TABEL PERBANDINGAN PERFORMA SEBELUM DAN SESUDAH TUNING Pada Data Uji (Termasuk TP, TN, FP, FN)")
    print(f"{'-'*95}")
    print(f"  {'Model':<15} {'Tipe':<10} | {'Accuracy':>8} {'Precision':>9} "
          f"{'Recall':>9} {'F1-Score':>9} | {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'-'*93}")

    for algo in ALGORITHMS:
        for tipe in ['baseline', 'tuned']:
            nama_model = f"{algo}_{tipe}"
            if nama_model in hasil_evaluasi:
                metrik = hasil_evaluasi[nama_model]
                print(f"  {algo.upper():<15} {tipe:<10} | "
                      f"{metrik['Accuracy']:>8.4f}  {metrik['Precision']:>8.4f}  "
                      f"{metrik['Recall']:>8.4f}  {metrik['F1-Score']:>8.4f} | "
                      f"{metrik['TP']:>6} {metrik['TN']:>6} {metrik['FP']:>6} {metrik['FN']:>6}")
        # Tambahkan jarak spasial tiap algoritma
        print(f"  {'-'*93}")

def buat_visualisasi_komparasi_baseline(hasil_evaluasi):
    """
    Membuat grafik bar komparasi 4 metrik khusus untuk 3 model baseline.
    """
    print("\n[VISUALISASI] Membuat grafik komparasi khusus model Baseline...")
    if not hasil_evaluasi:
        return
    
    path_gambar = os.path.join(OUTPUT_DIR, "perbandingan_algoritma_baseline.png")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    kunci_algo = ['lr_baseline', 'rf_baseline', 'svm_baseline']
    label_algo = ['LR Baseline', 'RF Baseline', 'SVM Baseline']
    
    skor_matrix = []
    min_score = 100.0
    for kunci in kunci_algo:
        metrik_skor = []
        if kunci in hasil_evaluasi:
            for m in metrics:
                s = hasil_evaluasi[kunci][m] * 100
                metrik_skor.append(s)
                if s < min_score and s > 0:
                    min_score = s
        else:
            metrik_skor = [0, 0, 0, 0]
        skor_matrix.append(metrik_skor)
        
    skor_matrix = np.array(skor_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.2
    
    cmap_base = plt.get_cmap('Set2')
    colors = [cmap_base(0), cmap_base(1), cmap_base(2)]
    
    rects = []
    offsets = [-1, 0, 1]
    for i in range(3):
        r = ax.bar(x + (offsets[i] * width), skor_matrix[i], width, label=label_algo[i], color=colors[i], edgecolor='white')
        rects.append(r)

    ax.set_ylabel('Persentase Skor (%)', fontsize=12, fontweight='bold')
    ax.set_title('Komparasi Metrik Evaluasi Model Baseline', fontsize=14, fontweight='bold', y=1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    y_min_target = 90.0 if min_score > 90.0 else max(80.0, np.floor(min_score - 2))
    ax.set_ylim(y_min_target, 100.3)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for r_group in rects:
        for rect in r_group:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    fig.savefig(path_gambar, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  [v] Visualisasi Baseline disimpan ke: {path_gambar}")

def buat_visualisasi_komparasi_algoritma(hasil_evaluasi):
    """
    Membuat Grouped Bar Chart untuk membandingkan 4 Metrik (Accuracy, Precision, Recall, F1)
    dari 6 model (3 algoritma x baseline & tuned).
    """
    print("\n[VISUALISASI] Membuat grafik komparasi 4 Metrik algoritma (Base & Tuned)...")
    if not hasil_evaluasi:
        return
    
    path_gambar = os.path.join(OUTPUT_DIR, "perbandingan_algoritma_detail_semua.png")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    kunci_algo = ['lr_baseline', 'lr_tuned', 'rf_baseline', 'rf_tuned', 'svm_baseline', 'svm_tuned']
    label_algo = ['LR Base', 'LR Tuned', 'RF Base', 'RF Tuned', 'SVM Base', 'SVM Tuned']
    
    # Kumpulkan skor
    skor_matrix = []
    min_score = 100.0
    for kunci in kunci_algo:
        metrik_skor = []
        if kunci in hasil_evaluasi:
            for m in metrics:
                s = hasil_evaluasi[kunci][m] * 100
                metrik_skor.append(s)
                if s < min_score and s > 0:
                    min_score = s
        else:
            metrik_skor = [0, 0, 0, 0]
        skor_matrix.append(metrik_skor)
        
    skor_matrix = np.array(skor_matrix) # Bentuk: 6 model x 4 metrik
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.12 # Lebar tiap bar
    
    cmap_base = plt.get_cmap('Paired')
    colors = [cmap_base(0), cmap_base(1), cmap_base(2), cmap_base(3), cmap_base(4), cmap_base(5)]
    
    rects = []
    # Menggeser posisi bar: kita punya 6 bar per kelompok
    # offset multiplier = -2.5, -1.5, -0.5, 0.5, 1.5, 2.5
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    for i in range(6):
        r = ax.bar(x + (offsets[i] * width), skor_matrix[i], width, label=label_algo[i], color=colors[i], edgecolor='white')
        rects.append(r)

    ax.set_ylabel('Persentase Skor (%)', fontsize=12, fontweight='bold')
    ax.set_title('Komparasi Performa Metrik 6 Model Klasifikasi (Baseline vs Tuned)', fontsize=14, fontweight='bold', y=1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    # Set y-axis limits (zoom in if min_score is high)
    y_min_target = 90.0 if min_score > 90.0 else max(80.0, np.floor(min_score - 2))
    ax.set_ylim(y_min_target, 100.3)
    
    # Legend di luar chart
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Tambahkan nilai di atas bar jika ada space
    for r_group in rects:
        for rect in r_group:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    fig.savefig(path_gambar, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  [v] Visualisasi komparasi (4 metrik, 6 model) disimpan ke: {path_gambar}")

def buat_visualisasi_komparasi_error(hasil_evaluasi):
    """
    Membuat chart terpisah khusus mendedikasikan komparasi error rate & success rate:
    (True Positive, True Negative, False Positive, False Negative).
    """
    print("\n[VISUALISASI] Membuat grafik komparasi TP, TN, FP, FN untuk semua model...")
    if not hasil_evaluasi:
        return
        
    path_gambar = os.path.join(OUTPUT_DIR, "perbandingan_tp_tn_fp_fn.png")
    
    # Kita buat 2 grafik bersisian: satu untuk TP & TN (yang tinggi), satu untuk FP & FN (yang rendah)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics_success = ['TP', 'TN']
    metrics_error = ['FP', 'FN']
    
    kunci_algo = ['lr_baseline', 'lr_tuned', 'rf_baseline', 'rf_tuned', 'svm_baseline', 'svm_tuned']
    label_algo = ['LR Base', 'LR Tuned', 'RF Base', 'RF Tuned', 'SVM Base', 'SVM Tuned']
    
    # Kumpulkan skor success & error
    skor_success = []
    skor_error = []
    for kunci in kunci_algo:
        if kunci in hasil_evaluasi:
            skor_success.append([hasil_evaluasi[kunci]['TP'], hasil_evaluasi[kunci]['TN']])
            skor_error.append([hasil_evaluasi[kunci]['FP'], hasil_evaluasi[kunci]['FN']])
        else:
            skor_success.append([0, 0])
            skor_error.append([0, 0])
            
    skor_success = np.array(skor_success).T # Bentuk: 2 metrik x 6 model
    skor_error = np.array(skor_error).T # Bentuk: 2 metrik x 6 model
    
    x1 = np.arange(len(kunci_algo))
    width = 0.35
    
    # === Subplot 1: TP vs TN ===
    ax1.bar(x1 - width/2, skor_success[0], width, label='True Positive (TP)', color='cornflowerblue')
    ax1.bar(x1 + width/2, skor_success[1], width, label='True Negative (TN)', color='lightgreen')
    
    ax1.set_title('Komparasi Metrik Sukses (TP & TN)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Jumlah Sampel', fontsize=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(label_algo, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # === Subplot 2: FP vs FN ===
    ax2.bar(x1 - width/2, skor_error[0], width, label='False Positive (FP)', color='salmon')
    ax2.bar(x1 + width/2, skor_error[1], width, label='False Negative (FN)', color='red')
    
    ax2.set_title('Komparasi Metrik Error (FP & FN)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Jumlah Sampel', fontsize=12)
    ax2.set_xticks(x1)
    ax2.set_xticklabels(label_algo, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Tambahkan angka di atas bar error karena angkanya kecil dan krusial 
    for i in range(len(kunci_algo)):
        # FP
        if skor_error[0][i] > 0 or True:
            ax2.annotate(f'{skor_error[0][i]}', xy=(x1[i] - width/2, skor_error[0][i]),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        # FN
        if skor_error[1][i] > 0 or True:
            ax2.annotate(f'{skor_error[1][i]}', xy=(x1[i] + width/2, skor_error[1][i]),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
                         
    # Label nilai metrik sukses (opsional krn tinggi)
    for i in range(len(kunci_algo)):
        ax1.annotate(f'{skor_success[0][i]}', xy=(x1[i] - width/2, skor_success[0][i]),
                     xytext=(0, -15), textcoords="offset points", ha='center', va='top', color='white', fontweight='bold', rotation=90)
        ax1.annotate(f'{skor_success[1][i]}', xy=(x1[i] + width/2, skor_success[1][i]),
                     xytext=(0, -15), textcoords="offset points", ha='center', va='top', color='black', fontweight='bold', rotation=90)

    fig.tight_layout()
    fig.savefig(path_gambar, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  [v] Visualisasi komparasi (TP/TN/FP/FN) 6 model disimpan ke: {path_gambar}")


def tampilkan_model_terbaik(hasil_evaluasi, X_test, y_test):
    """
    Menampilkan dan mencetak detail lengkap model tebaik.
    """
    if not hasil_evaluasi:
        return
        
    model_terbaik = max(hasil_evaluasi, key=lambda k: hasil_evaluasi[k]['F1-Score'])
    f1_terbaik = hasil_evaluasi[model_terbaik]['F1-Score']
    
    print(f"\n  >>> MODEL DENGAN F1-SCORE TERTINGGI: {model_terbaik.upper()} "
          f"(F1: {f1_terbaik:.4f}) <<<")

    # Ambil nilai prediksi & hitung classification_report+confusion_matrix
    y_pred = hasil_evaluasi[model_terbaik]['y_pred']
    
    print(f"\n{'='*70}")
    print(f"DETAIL STATISTIK: {model_terbaik.upper()}")
    print(f"{'='*70}")

    # Classification Report
    print(f"\n  Classification Report:")
    print("-" * 70)
    target_names = ['Phishing (0)', 'Legitimate (1)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'-'*40}")
    print(f"  {'':>20} {'Pred Phishing':>14} {'Pred Legit':>12}")
    print(f"  {'Actual Phishing':<20} {cm[0][0]:>14,} {cm[0][1]:>12,}")
    print(f"  {'Actual Legitimate':<20} {cm[1][0]:>14,} {cm[1][1]:>12,}")
    print(f"  {'-'*40}")

    # Visualisasi Matriks Kebingungan (Confusion Matrix)
    print("\n[VISUALISASI] Membuat grafik Heatmap Confusion Matrix...")
    path_gambar = os.path.join(OUTPUT_DIR, f"grafik_confusion_matrix_{model_terbaik}.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=True,
                xticklabels=['Phishing', 'Legitimate'], 
                yticklabels=['Phishing', 'Legitimate'], ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_terbaik.upper()}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path_gambar, dpi=150, facecolor='white')
    plt.close(fig)
    print(f"  [v] Visualisasi disimpan ke: {path_gambar}")

    # Copy / Duplikat model terbaik sebagai best_model.pkl agar gampang dipanggil di app
    path_terbaik = os.path.join(OUTPUT_DIR, f"{model_terbaik}.pkl")
    path_sym = os.path.join(OUTPUT_DIR, "best_model.pkl")
    
    if os.path.exists(path_terbaik):
        best_mdl = joblib.load(path_terbaik)
        joblib.dump(best_mdl, path_sym)
        print(f"\n  [v] Model terbaik di-copy dan disimpan sebagai: {path_sym}")
        
    buat_visualisasi_feature_importance(best_mdl, model_terbaik, X_test, y_test)


def buat_visualisasi_feature_importance(model, model_name, X_test, y_test):
    """
    Menghitung Feature Importance menggunakan Permutation Importance 
    (karena SVM dengan RBF Kernel tidak memiliki atribut .coef_).
    """
    print("\n[VISUALISASI] Menghitung Permutation Feature Importance (memerlukan waktu)...")
    path_gambar = os.path.join(OUTPUT_DIR, f"feature_importance_{model_name}.png")
    
    # Gunakan subset kecil jika data terlalu besar agar cepat (misal ambil max 2000 baris)
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    y_sample = y_test.loc[X_sample.index]
    
    result = permutation_importance(model, X_sample, y_sample, n_repeats=10, random_state=42, n_jobs=1)
    
    # Urutkan berdasarkan pentingannya
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('coolwarm')
    
    # Plotting bar horizontal
    ax.barh(np.array(X_test.columns)[sorted_idx], result.importances_mean[sorted_idx], 
            xerr=result.importances_std[sorted_idx], color=cmap(0.9), edgecolor='white')
    
    ax.set_xlabel("Mean Accuracy Decrease", fontsize=12, fontweight='bold')
    ax.set_title(f"Permutation Feature Importance - {model_name.upper()}", fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(path_gambar, dpi=150, facecolor='white')
    plt.close(fig)
    print(f"  [v] Visualisasi Feature Importance disimpan ke: {path_gambar}")


def jalankan_modul_evaluasi():
    """ Mengorkestrasi evaluasi (Dipanggil independen) """
    print("=" * 70)
    print("TAHAP 6: EVALUASI MODEL & KOMPARASI")
    print("=" * 70)
    
    X_test, y_test = muat_data_uji()
    hasil_evaluasi = evaluasi_model_tersimpan(X_test, y_test)
    
    if hasil_evaluasi:
        tampilkan_tabel_perbandingan(hasil_evaluasi)
        buat_visualisasi_komparasi_baseline(hasil_evaluasi)
        buat_visualisasi_komparasi_algoritma(hasil_evaluasi)
        buat_visualisasi_komparasi_error(hasil_evaluasi)
        tampilkan_model_terbaik(hasil_evaluasi, X_test, y_test)
    else:
        print("\n  [!] GAGAL: Belum ada model yang dilatih. Jalankan Modul 05 terlebih dahulu.")


# --- Eksekusi Langsung ---
if __name__ == "__main__":
    jalankan_modul_evaluasi()
