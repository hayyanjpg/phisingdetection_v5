FROM python:3.10-slim

# Mengatur direktori kerja di dalam kontainer
WORKDIR /app

# Menyalin file dependensi backend
COPY backend/requirements.txt ./backend/

# Menginstall dependensi (menambahkan direktori cache untuk optimalisasi)
RUN pip install --no-cache-dir -r backend/requirements.txt

# Menyalin seluruh repositori ke dalam direktori kerja
COPY . .

# Berpindah ke folder backend karena Flask dijalankan dari sana
WORKDIR /app/backend

# Mengekspos port 7860 (Standar wajib port dari Hugging Face)
EXPOSE 7860

# Menjalankan server menggunakan Gunicorn (Performa produksi)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "api:app"]
