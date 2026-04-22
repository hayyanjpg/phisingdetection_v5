FROM python:3.10-slim

WORKDIR /app

# Salin semua file dari repositori ke /app
COPY . .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Port standar Hugging Face
EXPOSE 7860

# Jalankan Gunicorn. 
# Jika api.py ada di root, maka 'api:app' sudah benar.
CMD ["gunicorn", "-b", "0.0.0.0:7860", "api:app"]
