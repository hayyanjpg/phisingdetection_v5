import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "DATASET", "url_features_extracted1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")

def generate_preprocessing_chart():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    
    # 1. NA Values
    jumlah_na = df.isnull().sum().sum()
    df = df.dropna()
    
    # 2. Offline URL
    df['URL'] = df['URL'].astype(str).str.strip().str.lower()
    mask_offline = df['URL'] == 'offline'
    jumlah_offline = mask_offline.sum()
    df = df[~mask_offline]
    
    # 3. Anomaly subdomain = -1
    if 'subdomain_count' in df.columns:
        mask_subdomain = df['subdomain_count'] == -1
        jumlah_subdomain = mask_subdomain.sum()
        df = df[~mask_subdomain]
    else:
        jumlah_subdomain = 0
        
    # 4. Duplicate
    jumlah_duplikat = df.duplicated().sum()
    df = df.drop_duplicates()
    
    reasons = ['Missing Values (NaN)', "URL 'offline'", 'Subdomain Anomaly (-1)', 'Data Duplikat']
    counts = [jumlah_na, jumlah_offline, jumlah_subdomain, jumlah_duplikat]
    
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(reasons, counts, color=colors, edgecolor='black', zorder=2)
    plt.title('Jumlah Data Dihapus Saat Pra-pemrosesan Berdasarkan Alasan', fontsize=14, fontweight='bold')
    plt.ylabel('Jumlah Baris Dihapus', fontsize=12, fontweight='bold')
    plt.xlabel('Alasan Penghapusan', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.01), 
                     f'{int(yval):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "grafik_preprocessing_deletion.png")
    plt.savefig(out_path, dpi=150, facecolor='white')
    plt.close()
    print(f"[v] Grafik penghapusan preprocessing disimpan di: {out_path}")

if __name__ == "__main__":
    generate_preprocessing_chart()
