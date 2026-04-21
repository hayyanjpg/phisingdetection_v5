import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from importlib import import_module

# Impor fungsi predict_url secara dinamis
try:
    inference_mod = import_module("07_inference")
    predict_url = getattr(inference_mod, "predict_url")
except Exception as e:
    st.error(f"Gagal memuat modul inference: {e}")
    st.stop()

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Anti-Phish AI | Deteksi URL Berbahaya",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Kustomisasi CSS untuk UI yang lebih modern
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #888;
        margin-bottom: 30px;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 12px 15px;
    }
    .phishing-box {
        background-color: #ffeaea;
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .legit-box {
        background-color: #ebffeb;
        border-left: 5px solid #00c04b;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .xai-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FUNGSI PEMBANTU (HELPER)
# ==============================================================================
def create_gauge_chart(confidence_score, prediction):
    """Membuat Gauge Chart interaktif untuk Confidence Score"""
    color = "#FF4B4B" if prediction == "Phishing" else "#00C04B"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f2f2f2'},
                {'range': [50, 80], 'color': '#e6e6e6'},
                {'range': [80, 100], 'color': '#cccccc'}],
        }
    ))
    
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def generate_xai_table(features, prediction):
    """
    Menghasilkan tabel alasan yang menerjemahkan fitur numerik
    menjadi penjelasan yang mudah dipahami manusia.
    """
    reasons = []
    
    # 1. IP Address
    if features.get('has_ip_address') == 1:
        reasons.append({"Fitur": "Alamat IP", "Nilai": "Ditemukan", 
                       "Penjelasan": "URL menggunakan IP address alih-alih nama domain (Taktik umum penyamaran URL)."})
    
    # 2. HTTPS Flag
    if features.get('https_flag') == 0:
        reasons.append({"Fitur": "Keamanan Protokol (HTTPS)", "Nilai": "Tidak Ada", 
                       "Penjelasan": "Koneksi tidak terenkripsi (HTTP). Sangat berisiko untuk transaksi/login."})
        
    # 3. HTTP di dalam Domain (Custom)
    if features.get('http_in_domain') == 1:
        reasons.append({"Fitur": "Kata 'HTTP' Tersembunyi", "Nilai": "Terdeteksi", 
                       "Penjelasan": "Domain mencoba mengelabui pengguna dengan menyematkan kata 'http/https' ke dalam nama domain."})
                       


    # 5. Length
    url_len = features.get('url_length', 0)
    if url_len > 75:
        reasons.append({"Fitur": "Panjang URL", "Nilai": f"{url_len} karakter", 
                       "Penjelasan": "URL sangat panjang melebihi batas wajar situs normal."})

    # 6. File Extension
    if features.get('suspicious_file_extension') == 1:
        reasons.append({"Fitur": "File Mencurigakan", "Nilai": "Terdeteksi", 
                       "Penjelasan": "URL mengarah langsung ke eksekusi file (.exe, .zip, dll)."})

    # 7. Entropy (Acak/Random)
    if features.get('url_entropy', 0) > 4.5:
        reasons.append({"Fitur": "Tingkat Keacakan Karakter", "Nilai": f"{features.get('url_entropy'):.2f}", 
                       "Penjelasan": "Karakter URL terlihat di-*generate* secara acak oleh mesin."})

    # Jika tidak ada bendera merah mencolok tapi di-prediksi phishing
    if prediction == "Phishing" and len(reasons) == 0:
         reasons.append({"Fitur": "Pola General", "Nilai": "Kombinasi Meta", 
                       "Penjelasan": "Kombinasi berbagai fitur leksikal lainnya membentuk pola matematis mirip phishing di dalam algoritma SVM."})
                       
    # Jika Legitimate
    if prediction == "Legitimate":
         reasons.append({"Fitur": "Keamanan Struktur", "Nilai": "Lulus Pengecekan", 
                       "Penjelasan": "Struktur URL rapi, tidak ada anomali leksikal yang signifikan."})

    return pd.DataFrame(reasons)


# ==============================================================================
# UI COMPONENTS
# ==============================================================================
st.markdown('<p class="main-title">Anti-Phish AI 🛡️</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Deteksi Phishing Leksikal berbasis Machine Learning (Support Vector Machine)</p>', unsafe_allow_html=True)

st.write("---")

# Input Section
input_url = st.text_input("🔗 Masukkan tautan (URL) yang ingin diperiksa:", 
                          placeholder="Contoh: https://secure-login.paypal.com/signin")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    analyze_btn = st.button("🔍 Analisis URL", use_container_width=True, type="primary")

# Result Section
if analyze_btn:
    if not input_url:
        st.warning("⚠️ Harap masukkan URL terlebih dahulu!")
    else:
        with st.spinner("🧠 Mengekstrak fitur leksikal dan AI sedang berpikir..."):
            hasil = predict_url(input_url)
            
        if hasil['status'] == 'error':
            st.error(f"Terjadi kesalahan: {hasil['message']}")
        else:
            pred = hasil['prediction']
            conf = hasil['confidence']
            fitur_mentah = hasil['extracted_features']
            
            # 1. Kotak Hasil (Merah / Hijau)
            if pred == "Phishing":
                st.markdown(f"""
                <div class="phishing-box">
                    <h2 style="color: #ff4b4b; margin-top: 0px;">🚨 PERINGATAN: SANGAT BERBAHAYA!</h2>
                    <p>Sistem mendeteksi bahwa URL ini memiliki pola yang identik dengan situs <b>Phishing</b> atau Penipuan Online. Harap JANGAN mengklik atau memasukkan data pribadi Anda!</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons() # Just for show, maybe an alert is better, but Streamlit lacks native red balloons
            else:
                st.markdown(f"""
                <div class="legit-box">
                    <h2 style="color: #00c04b; margin-top: 0px;">✅ AMAN: URL SAH (LEGITIMATE)</h2>
                    <p>URL ini tampaknya aman untuk dikunjungi. Tidak ditemukan pola leksikal yang mencurigakan.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("---")
            
            # 2. visualisasi Keyakinan AI & Alasan (XAI)
            col_gauge, col_reasons = st.columns([1, 1.2])
            
            with col_gauge:
                st.markdown('<p class="xai-title">Keyakinan AI (Confidence)</p>', unsafe_allow_html=True)
                fig = create_gauge_chart(conf, pred)
                st.plotly_chart(fig, use_container_width=True)
                
            with col_reasons:
                st.markdown('<p class="xai-title">🔎 Explainable AI: Alasan Deteksi</p>', unsafe_allow_html=True)
                df_reasons = generate_xai_table(fitur_mentah, pred)
                st.dataframe(df_reasons, hide_index=True, use_container_width=True)
            
            # 3. Data Pendukung (Detail 17 Fitur)
            with st.expander("📊 Lihat Detail 17 Fitur Leksikal yang Diekstraksi"):
                st.write("Sistem mengekstrak fitur matematis (leksikal) berikut secara _real-time_ sebelum diteruskan ke model SVM:")
                df_detail = pd.DataFrame([fitur_mentah]).T.reset_index()
                df_detail.columns = ["Nama Fitur", "Nilai Bobot"]
                st.dataframe(df_detail, hide_index=True, use_container_width=True)
