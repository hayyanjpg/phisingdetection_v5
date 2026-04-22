# ==============================================================================
# MODUL 07: INFERENCE (EKSTRAKSI FITUR REAL-TIME)
# ==============================================================================
# Deskripsi:
#   Modul ini merupakan modul mandiri (standalone) yang berfungsi untuk
#   melakukan prediksi klasifikasi Phishing atau Legitimate pada URL mentah
#   (string) yang dimasukkan pengguna. 
#
#   Proses Utama:
#     1. Memvalidasi format URL.
#     2. Mengekstrak 17 fitur secara real-time (16 leksikal bawaan + 1 kustom).
#     3. Memuat Scaler (scaler.pkl) dan menstandarkan fitur.
#     4. Memuat Model Terbaik (best_model.pkl) dan mengembalikan prediksi
#        serta confidence score (probabilitas).
# ==============================================================================

import os
import re
import math
import joblib
import pandas as pd
from urllib.parse import urlparse
import tldextract

# --- Konfigurasi Path (Adaptif untuk Lokal maupun Cloud) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")

def dapatkan_path_artefak(nama_file):
    # Cek di folder OUTPUT dulu (lokal)
    path_output = os.path.join(OUTPUT_DIR, nama_file)
    if os.path.exists(path_output):
        return path_output
    # Jika tidak ada, cek di direktori utama (Hugging Face/Cloud)
    path_root = os.path.join(BASE_DIR, nama_file)
    return path_root

SCALER_PATH = dapatkan_path_artefak("scaler.pkl")
FEAT_NAMES_PATH = dapatkan_path_artefak("feature_names.pkl")

# Khusus model, cek best_model.pkl atau svm_tuned.pkl
MODEL_PATH = dapatkan_path_artefak("best_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = dapatkan_path_artefak("svm_tuned.pkl")

# Daftar TLD (Top-Level Domain) populer (sebagai proksi tld_popularity)
POPULAR_TLDS = ['.com', '.org', '.net', '.edu', '.gov', '.uk', '.io', '.co', '.us', '.info']

# Daftar Ekstensi File Mencurigakan (yang sering digunakan phishing)
SUSPICIOUS_EXTS = ['.exe', '.jar', '.bat', '.bin', '.js', '.vbs', '.apk', '.rar', '.zip', '.tar', '.gz', '.sh']


class FeatureExtractor:
    """Kelas untuk mengekstrak 17 fitur dari sebuah string URL mentah."""
    
    def __init__(self, url):
        self.raw_url = str(url).strip().lower()
        
        # Ekstrak komponen utama menggunakan urllib
        # Tambahkan prefix http:// jika tidak ada agar parsing benar
        if not self.raw_url.startswith(('http://', 'https://')):
            self.parsed_url = urlparse('http://' + self.raw_url)
        else:
            self.parsed_url = urlparse(self.raw_url)
            
        self.domain = self.parsed_url.netloc.split(':')[0]  # Buang port jika ada
        self.path = self.parsed_url.path
        self.query = self.parsed_url.query
        
        # Ekstraksi TLD khusus untuk menyesuaikan dengan logika dataset standar
        self.tld_ext = tldextract.extract(self.raw_url)

    # --------------------------------------------------------------------------
    # FUNGSI EKSTRAKSI 16 FITUR BAWAAN
    # --------------------------------------------------------------------------
    def get_url_length(self):
        return len(self.raw_url)

    def get_has_ip_address(self):
        # Deteksi IPv4 di dalam domain (contoh: 192.168.1.1)
        ipv4_pattern = re.compile(
            r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        )
        return 1 if ipv4_pattern.match(self.domain) else 0

    def get_dot_count(self):
        return self.raw_url.count('.')

    def get_https_flag(self):
        return 1 if self.raw_url.startswith('https://') else 0

    def get_url_entropy(self):
        """Menghitung Shannon Entropy dari string URL"""
        string_data = list(self.raw_url)
        alphabets = list(set(string_data))
        entropy = 0
        for alpha in alphabets:
            p = string_data.count(alpha) / len(string_data)
            entropy -= p * math.log(p, 2)
        return entropy

    def get_token_count(self):
        # Berdasarkan analisis dataset raw, token_count dihasiikan dari
        # pemisahan karakter hierarki (./?=&) dan bukan sembarang non-alfanumerik.
        import re
        tokens = re.split(r'[./?=&]', self.raw_url)
        return len(tokens)

    def get_subdomain_count(self):
        """Menyesuaikan hitungan subdomain standar seperti pada dataset."""
        if not self.tld_ext.subdomain:
            return 0
        # Jika ada sub.sub.domain, maka split titik akan memberikan jumlahnya
        return len(self.tld_ext.subdomain.split('.'))

    def get_query_param_count(self):
        # Format dataset mentah (raw_seimbang) TERNYATA selalu menggunakan 
        # base value = 1 untuk perhitungan query parameter (tidak pernah 0).
        # Jika nilai dilempar 0, Scaler akan menganggapnya sebagai deviasi 
        # ekstrem (-8.4 Standard Deviation) yang merusak prediksi SVM !
        return self.query.count('&') + 1

    def get_tld_length(self):
        # Gunakan tldextract untuk mendapat ukuran TLD/suffix yang akurat
        return len(self.tld_ext.suffix)

    def get_path_length(self):
        return len(self.path)

    def get_has_hyphen_in_domain(self):
        return 1 if '-' in self.domain else 0

    def get_number_of_digits(self):
        return sum(c.isdigit() for c in self.raw_url)

    def get_tld_popularity(self):
        # Mengecek apakah akhir domain memiliki TLD populer
        for tld in POPULAR_TLDS:
            if self.domain.endswith(tld):
                return 1
        return 0

    def get_suspicious_file_extension(self):
        for ext in SUSPICIOUS_EXTS:
            if self.path.endswith(ext):
                return 1
        return 0

    def get_domain_name_length(self):
        # Dataset mentah menganggap 'domain' hanya nama domain intinya, bukan seluruh host
        return len(self.tld_ext.domain)

    def get_percentage_numeric_chars(self):
        length = len(self.raw_url)
        if length == 0: return 0.0
        return (self.get_number_of_digits() / length) * 100

    # --------------------------------------------------------------------------
    # FUNGSI EKSTRAKSI 1 FITUR KUSTOM (Dari Bab 3.6)
    # --------------------------------------------------------------------------

    def get_http_in_domain(self):
        if 'http' in self.domain or 'https' in self.domain:
            return 1
        return 0

    # --------------------------------------------------------------------------
    # BUILD PIPELINE
    # --------------------------------------------------------------------------
    def extract_all(self):
        """Menggabungkan seluruh proses dalam Dictionary"""
        return {
            'url_length': self.get_url_length(),
            'has_ip_address': self.get_has_ip_address(),
            'dot_count': self.get_dot_count(),
            'https_flag': self.get_https_flag(),
            'url_entropy': self.get_url_entropy(),
            'token_count': self.get_token_count(),
            'subdomain_count': self.get_subdomain_count(),
            'query_param_count': self.get_query_param_count(),
            'tld_length': self.get_tld_length(),
            'path_length': self.get_path_length(),
            'has_hyphen_in_domain': self.get_has_hyphen_in_domain(),
            'number_of_digits': self.get_number_of_digits(),
            'tld_popularity': self.get_tld_popularity(),
            'suspicious_file_extension': self.get_suspicious_file_extension(),
            'domain_name_length': self.get_domain_name_length(),
            'percentage_numeric_chars': self.get_percentage_numeric_chars(),
            'http_in_domain': self.get_http_in_domain()
        }


def _cek_ketersediaan_artefak():
    """Memastikan bahwa scaler.pkl dan best_model.pkl (serta feature names) ada."""
    hilang = []
    if not os.path.exists(SCALER_PATH): hilang.append("scaler.pkl")
    if not os.path.exists(MODEL_PATH): hilang.append("best_model.pkl")
    if not os.path.exists(FEAT_NAMES_PATH): hilang.append("feature_names.pkl")
    return hilang


def predict_url(url_string):
    """
    Fungsi utama untuk memprediksi sebuah URL string.
    
    Proses:
        1. Ekstraksi fitur real-time
        2. Scaler Transform
        3. Prediksi Model
        
    Returns:
        dict: Berisi status, label klasifikasi, probabilitas, dan fitur mentah.
    """
    # 1. Validasi Input Basic
    if not isinstance(url_string, str) or len(url_string.strip()) < 4:
        return {'status': 'error', 'message': 'Input URL kosong atau tidak valid.'}

    # 2. Cek Ketersediaan Output Model
    artefak_hilang = _cek_ketersediaan_artefak()
    if artefak_hilang:
        return {
            'status': 'error', 
            'message': f'Artefak ML hilang: {", ".join(artefak_hilang)}. Harap jalankan Tahap 4 & 5.'
        }

    try:
        # 3. Ekstraksi Fitur
        extractor = FeatureExtractor(url_string)
        raw_features_dict = extractor.extract_all()
        
        # Load nama fitur agar urutan sesuai persis 100%
        ordered_feature_names = joblib.load(FEAT_NAMES_PATH)
        
        # Konversi ke DataFrame 1 baris
        df_features = pd.DataFrame([raw_features_dict])
        
        # Terapkan ordering (Sangat penting agar dimensi scaler cocok)
        df_features = df_features[ordered_feature_names]
        
        # 4. Standardisasi (Scaling)
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(df_features)

        # 5. Load Model & Prediksi
        model = joblib.load(MODEL_PATH)
        prediksi = model.predict(X_scaled)[0]           # 0 (Phishing) atau 1 (Legitimate)
        probabilitas = model.predict_proba(X_scaled)[0] # Confidence [prob_0, prob_1]

        # 6. DEBUGGING: Tulis ke file (gunakan BASE_DIR jika OUTPUT tidak ada)
        debug_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else BASE_DIR
        with open(os.path.join(debug_dir, 'debug_scaled.txt'), 'w') as f:
            f.write(f"URL: {url_string}\nFeatures: {df_features.iloc[0].to_dict()}\nScaled: {list(X_scaled[0])}\nPrediksi: {prediksi}\n")

        # 6. Susun Output
        label_text = "Phishing" if prediksi == 0 else "Legitimate"
        confidence = probabilitas[prediksi] * 100       # Konversi ke persentase

        return {
            'status': 'success',
            'url': url_string,
            'prediction': label_text,
            'prediction_code': int(prediksi),
            'confidence': confidence,
            'probability_phishing': probabilitas[0] * 100,
            'probability_legitimate': probabilitas[1] * 100,
            'extracted_features': raw_features_dict
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# --- PENGUJIAN LOKAL (TEST) ---
if __name__ == "__main__":
    
    print("=" * 70)
    print("PENGUJIAN MODUL INFERENCE")
    print("=" * 70)
    
    test_urls = [
        # (1) URL Sah (Legitimate)
        "https://www.google.com/search?q=machine+learning",
        "https://github.com/login",
        
        # (2) URL Phishing Umum
        "http://secure-update-paypal.com/login.php",
        
        # (3) URL Phishing Anomali IP 
        "http://192.168.1.100/admin/update.exe",
        
        # (4) URL Phishing Anomali Custom (http di luar skema & banyak slash)
        "https://secure-login.http.paypal.com/xyz/abc/def/123/auth"
    ]

    for url in test_urls:
        hasil = predict_url(url)
        print(f"\n[?] URL : {url}")
        if hasil['status'] == 'error':
            print(f"  [X] Error: {hasil['message']}")
        else:
            print(f"  [v] Prediksi  : {hasil['prediction'].upper()}")
            print(f"  [v] Keyakinan : {hasil['confidence']:.2f}%")
            if hasil['prediction'] == 'Phishing':
                fitur = hasil['extracted_features']
                print(f"      - Bukti: Has IP: {fitur['has_ip_address']}, HTTP di Domain: {fitur['http_in_domain']}")
