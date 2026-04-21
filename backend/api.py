import sys
import os

# 1. Pastikan python dapat menemukan 07_inference.py di folder root proyek (satu tingkat di atas backend/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS

# 2. Mengimpor fungsi prediksi yang sudah matang dan teruji dari 07_inference.py
try:
    inference_module = __import__("07_inference")
    predict_url = inference_module.predict_url
except ImportError as e:
    print(f"CRITICAL ERROR: Gagal memuat modul 07_inference dari root. Pesan: {e}")
    predict_url = None

# 3. Inisialisasi Flask Apps
app = Flask(__name__)
CORS(app) # Mengaktifkan CORS agar html frontend bisa ngobrol dengan backend ini

@app.route('/predict', methods=['POST'])
def predict():
    if not predict_url:
        return jsonify({
            "status": "error", 
            "message": "Backend gagal meload model inference. Pastikan struktur folder tepat."
        }), 500

    # Mengambil payload JSON {"url": "https://..."} dari frontend
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({"status": "error", "message": "Harap masukkan data JSON dengan key 'url'."}), 400
        
    input_url = data['url']
    
    # 4. Mengeksekusi pipeline prediksi secara menyeluruh menggunakan kode asli
    # Di dalam fungsi ini, ekstraksi 17 fitur, scaling, dan prediksi model dilakukan secara otomatis.
    hasil = predict_url(input_url)
    
    # Translasi Respons/Handle Error
    if hasil.get('status') == 'error':
        return jsonify(hasil), 400
        
    # 5. Mengirim Format Standar untuk Frontend Chat
    return jsonify({
        "status": "success",
        "url": hasil.get('url'),
        "prediction": hasil.get('prediction'),
        "confidence": hasil.get('confidence'),
        "features": hasil.get('extracted_features')
    })

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Backend Active", "project": "Anti-Phish AI API"})

if __name__ == '__main__':
    print("="*60)
    print("🚀 MEMULAI SERVER BACKEND APi!")
    print("Endpoint tersedia: POST http://127.0.0.1:5000/predict")
    print("="*60)
    app.run(debug=True, port=5000)
