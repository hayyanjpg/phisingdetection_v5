# ==============================================================================
# MODUL 05: PELATIHAN & HYPERPARAMETER TUNING (Bab 3.8 - 3.9)
# ==============================================================================
# Deskripsi:
#   Modul ini khusus untuk melakukan pelatihan model dan hyperparameter tuning.
#   Proses yang dilakukan:
#     1. Pelatihan Baseline: Melatih 3 algoritma (LR, RF, SVM) menggunakan
#        parameter default pada seluruh data latih.
#     2. Menyimpan model baseline ke dalam file .pkl di folder OUTPUT/.
#     3. Hyperparameter Tuning: Melakukan pencarian parameter terbaik dengan
#        GridSearchCV (dan 5-Fold CV) pada seluruh data latih.
#     4. Menyimpan model terbaik (tuned) ke dalam file .pkl di folder OUTPUT/.
#   
#   Dengan memisahkan tahap ini dari evaluasi, kita dapat menyimpan semua 
#   model (sebelum dan sesudah tuning) untuk dibandingkan nanti.
# ==============================================================================

import pandas as pd
import joblib
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# --- Konfigurasi Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")

# --- Konfigurasi Parameter ---
CV_FOLDS = 5            # Jumlah fold untuk Cross-Validation saat tuning
RANDOM_STATE = 42       # Seed untuk reproduksibilitas
N_JOBS = 1             # Gunakan 1 core CPU agar tidak OOM di memori


# ==============================================================================
# BAGIAN 1: DEFINISI MODEL & PARAMETER GRID
# ==============================================================================

def definisi_model_baseline():
    """
    Mendefinisikan tiga model klasifikasi dengan parameter default
    untuk evaluasi baseline.

    Returns:
        dict: Dictionary berisi nama model dan objek model.
    """
    models = {
        'lr': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000     # Ditingkatkan agar konvergen pada dataset besar
        ),
        'rf': RandomForestClassifier(
            random_state=RANDOM_STATE
        ),
        'svm': SVC(
            random_state=RANDOM_STATE,
            kernel='rbf',
            probability=True  # Aktifkan untuk mendapatkan confidence score
        )
    }
    return models


def definisi_parameter_grid():
    """
    Mendefinisikan ruang pencarian hyperparameter untuk setiap algoritma
    yang akan digunakan pada proses Grid Search.

    Returns:
        dict: Dictionary berisi nama model dan parameter grid-nya.
    """
    param_grids = {
        'lr': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000]
        },
        'rf': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'svm': {
            # Grid SVM dibatasi untuk efisiensi waktu komputasi
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf'],
            'probability': [True]
        }
    }
    return param_grids


# ==============================================================================
# BAGIAN 2: PELATIHAN MODEL BASELINE
# ==============================================================================

def latih_dan_simpan_baseline(X_train, y_train):
    """
    Melatih model dengan parameter default pada seluruh data latih,
    lalu menyimpan model tersebut.

    Parameter:
        X_train (np.ndarray): Matriks fitur data latih.
        y_train (pd.Series): Vektor label data latih.
    """
    print("=" * 70)
    print("TAHAP 5: PELATIHAN & HYPERPARAMETER TUNING")
    print("=" * 70)
    print("\n[PROSES 1] PELATIHAN MODEL BASELINE (Parameter Default)")
    
    models = definisi_model_baseline()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for nama, model in models.items():
        print(f"  Melatih model {nama.upper()} (baseline)...", end=" ")
        start_time = time.time()
        
        # Fit model pada seluruh data latih
        model.fit(X_train, y_train)
        
        durasi = time.time() - start_time
        print(f"Selesai! ({durasi:.2f} detik)")
        
        # Simpan model
        path_simpan = os.path.join(OUTPUT_DIR, f"{nama}_baseline.pkl")
        joblib.dump(model, path_simpan)
        print(f"  [v] Disimpan ke: {path_simpan}")


# ==============================================================================
# BAGIAN 3: HYPERPARAMETER TUNING (Grid Search)
# ==============================================================================

def latih_tuning_dan_simpan(X_train, y_train):
    """
    Melakukan Hyperparameter Tuning menggunakan GridSearchCV,
    lalu menyimpan model terbaik hasil pencarian.

    Parameter:
        X_train (np.ndarray): Matriks fitur data latih.
        y_train (pd.Series): Vektor label data latih.
    """
    print(f"\n{'='*70}")
    print(f"[PROSES 2] HYPERPARAMETER TUNING (Grid Search {CV_FOLDS}-Fold CV)")
    print(f"{'='*70}")

    models = definisi_model_baseline()
    param_grids = definisi_parameter_grid()
    
    for nama in models:
        model = models[nama]
        grid = param_grids[nama]
        
        print(f"\n  Memulai Grid Search untuk {nama.upper()}...")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=CV_FOLDS,
            scoring='f1',
            n_jobs=N_JOBS,
            verbose=1,
            refit=True      # Model akan di-refit pada keseluruhan data latih dengan best params
        )
        
        grid_search.fit(X_train, y_train)
        
        durasi = time.time() - start_time
        print(f"  Selesai! Waktu Tuning: {durasi:.2f} detik")
        print(f"  Best F1-Score (CV) : {grid_search.best_score_:.4f}")
        print(f"  Best Parameters    : {grid_search.best_params_}")
        
        # Ekstrak model terbaik dan simpan
        best_model = grid_search.best_estimator_
        path_simpan = os.path.join(OUTPUT_DIR, f"{nama}_tuned.pkl")
        joblib.dump(best_model, path_simpan)
        print(f"  [v] Model Tuned Disimpan ke: {path_simpan}")


def jalankan_modul_training(X_train, y_train):
    """
    Fungsi utama untuk orkestrasi pelatihan baseline dan tuning.
    """
    waktu_total_mulai = time.time()
    
    latih_dan_simpan_baseline(X_train, y_train)
    latih_tuning_dan_simpan(X_train, y_train)

    durasi_total = time.time() - waktu_total_mulai
    print(f"\n[v] Tahap 5 Selesai! Semua model berhasil dilatih dan disimpan.")
    print(f"    Waktu total proses: {durasi_total/60:.2f} menit.\n")


# --- Eksekusi Langsung ---
if __name__ == "__main__":
    import importlib.util

    # Menjalankan modul sebelumnya secara berurutan
    # 01. Preprocessing
    spec1 = importlib.util.spec_from_file_location("preprocessing", os.path.join(BASE_DIR, "01_preprocessing.py"))
    mod1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(mod1)
    df_bersih = mod1.jalankan_preprocessing()

    # 02. Balancing
    spec2 = importlib.util.spec_from_file_location("balancing", os.path.join(BASE_DIR, "02_balancing.py"))
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    df_seimbang = mod2.jalankan_balancing(df_bersih)

    # 03. Feature Engineering
    spec3 = importlib.util.spec_from_file_location("feature_engineering", os.path.join(BASE_DIR, "03_feature_engineering.py"))
    mod3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(mod3)
    df_final = mod3.jalankan_feature_engineering(df_seimbang)

    # 04. Split & Scaling
    spec4 = importlib.util.spec_from_file_location("split_scaling", os.path.join(BASE_DIR, "04_split_scaling.py"))
    mod4 = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(mod4)
    X_train, X_test, y_train, y_test, feat_names = mod4.jalankan_split_scaling(df_final)

    # 05. Pelatihan dan Tuning Model (HANYA X_train dan y_train)
    jalankan_modul_training(X_train, y_train)
