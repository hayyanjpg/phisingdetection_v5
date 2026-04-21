# BAB IV
# HASIL DAN PEMBAHASAN

Bab ini memuat hasil dari penelitian beserta analisis dan observasi terhadap implementasi sistem klasifikasi deteksi tautan phishing. Penjabaran pada bab ini diurutkan berdasarkan parameter dan metode yang telah diajukan sebelumnya pada bab metodologi penelitian. Bab ini secara garis besar terbagi menjadi dua bagian utama, yaitu bagian hasil yang menjelaskan secara rinci mengenai hasil dari setiap tahapan dari penelitian yang telah dilaksanakan, dan bagian pembahasan yang berisi uraian tentang analisis dan observasi dari hasil berdasarkan parameter-parameter yang telah ditentukan.

## 4.1 Hasil

Bagian ini memuat hasil murni dari implementasi sistem. Hasil dari tiap tahapan dijabarkan secara objektif berdasarkan kerangka kerja metode penelitian, mencakup pengumpulan data, prapemrosesan, rekayasa fitur, hingga pengujian dan implementasi antarmuka.

### 4.1.1 Hasil Pengumpulan Data

Hasil dari tahap pengumpulan data adalah diperolehnya dataset sekunder yang bersumber dari repositori Mendeley Data dengan nama LegitPhish Dataset. Berdasarkan hasil pengunduhan dan peninjauan awal, dataset mentah ini memiliki total sebanyak 101.219 entri baris data. Struktur dataset ini memiliki 16 fitur leksikal numerik bawaan beserta atribut kelas target yang bersifat biner, di mana nilai 0 merepresentasikan kelas tautan phishing dan nilai 1 merepresentasikan kelas tautan legitimate. Proporsi awal dari dataset ini menunjukkan adanya ketidakseimbangan kelas yang signifikan yang akan ditangani pada tahap prapemrosesan.

### 4.1.2 Hasil Prapemrosesan

Langkah pertama yang dilakukan setelah pengumpulan data adalah prapemrosesan guna menjamin integritas data sebelum memasuki tahapan rekayasa fitur. Hasil dari tahap pembersihan data menunjukkan adanya penghapusan sejumlah entri yang tidak valid. Secara rincian, dilakukan pembuangan terhadap 1 baris data yang memiliki nilai missing value (Null), 1.307 baris data yang bernilai anomali pada fitur subdomain_count, 1 baris URL bernilai offline, serta 346 baris data yang terdeteksi sebagai data duplikat secara absolut. Rincian penghapusan entri data ini ditunjukkan pada Gambar 4.1.

![Grafik Pra-pemrosesan](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/grafik_preprocessing_deletion.png)
*Gambar 4.1 Proporsi Reduksi Dataset Mentah pada Sesi Pra-pemrosesan*

Setelah tahapan pembersihan dilakukan, dataset yang tersisa berjumlah 99.564 baris. Namun, distribusi kelas pada data ini memiliki kecenderungan bias kelas mayoritas, yakni 62.184 sampel untuk kelas phishing (62,46%) dan 37.380 sampel untuk kelas legitimate (37,54%). Oleh karena itu, dilakukan proses penyeimbangan data menggunakan teknik random undersampling pada kelas phishing. Proses ini berhasil menyeimbangkan komposisi kelas menjadi setara, di mana masing-masing telah berjumlah 37.380 entri data per kelas. Distribusi kelas sebelum dan sesudah tahap penyeimbangan data dapat dilihat pada Gambar 4.2.

![Grafik Perbandingan Sebaran Kelas](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/grafik_balancing.png)
*Gambar 4.2 Keseragaman Label Kelas Sebelum dan Sesudah Proses Balancing*

Hasil akhir dari tahap prapemrosesan dan penyeimbangan data adalah sebuah format dataset baru yang berukuran 74.760 baris data tervalidasi yang siap diproses lebih lanjut.

### 4.1.3 Feature Engineering

Berdasarkan dataset yang telah dibersihkan dan diseimbangkan, tahapan rekayasa fitur (feature engineering) dilakukan untuk mentransformasi URL ke dalam fitur numerik yang komprehensif. Pada tahap ini, 16 fitur leksikal asli diekstrak, dan sistem mengimplementasikan tambahan satu buah pengekstraksi fitur kustom bernama http_in_domain. Fitur ini dirancang secara khusus untuk mendeteksi keberadaan karakter string "http" atau "https" di dalam susunan nama domain guna menangkap pola manipulatif. Keberhasilan penyaringan dan ekstraksi fitur di dalam dataset terepresentasikan pada hasil proporsi distribusi yang ditunjukkan pada Gambar 4.3.

![Distribusi Fitur Kustom](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/distribusi_fitur_kustom.png)
*Gambar 4.3 Sebaran Rasio Eksistensi String HTTP di Lingkup Domain Utama*

Seluruh susunan nilai ekstraksi fitur leksikal tambahan dan fitur bawaan kemudian dipersatukan ke dalam satu format dataset standar terpadu yang siap dilatih menggunakan algoritma machine learning.

### 4.1.4 Pelatihan Model Baseline dan Evaluasi

Data hasil ekstraksi fitur kemudian dibagi menjadi kelompok data latih (training set) dan data uji (testing set) menggunakan teknik stratified sampling. Proses pemisahan data ini dikonfigurasikan dengan rasio pembagian 80% untuk keperluan pemelajaran dan 20% sisanya dipertahankan sebagai alat pengujian. Gambaran pembagian set data secara struktural dapat dilihat pada Gambar 4.4.

![Grafik Proporsi Split Data](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/grafik_splitting.png)
*Gambar 4.4 Konfigurasi Partisi Data Pelatih dan Penguji*

Setelah rentang data selesai distandarisasi skalanya, tahap selanjutnya adalah melakukan pelatihan model baseline. Tiga kelompok algoritma dasar, yaitu algoritma Logistic Regression, Random Forest, dan Support Vector Machine, dieksekusi menggunakan hiperparameter default bawaan pustaka pembangunnya. Hasil evaluasi awal terhadap metrik akurasi dari uji data ini menunjukkan bahwa model Logistic Regression menghasilkan akurasi sebesar 99,70%. Di sisi lain, algoritma kompleksitas tinggi menghasilkan performa yang lebih kompetitif, di mana Support Vector Machine mencatatkan skor akurasi sebesar 99,83%, dan model klasifikasi Random Forest memperoleh titik akurasi dasar di angka 99,91%. Visualisasi komparatif kinerja model awal diringkas pada histogram di Gambar 4.5.

![Grafik Performa Baseline](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/perbandingan_algoritma_baseline.png)
*Gambar 4.5 Histogram Metrik Evaluasi pada Fase Pemelajaran Model Baseline*

### 4.1.5 Tuning dan Evaluasi

Dalam upaya mencapai performa deteksi klasifikasi yang paling maksimal, langkah berikutnya adalah mengoptimasi hiperparameter pada model yang telah dievaluasi menggunakan teknik pencarian menyeluruh berjenis GridSearchCV. Proses eksplorasi variasi parameter pada algoritma Random Forest meliputi konfigurasi batasan kedalaman batas dan jumlah penunjang *estimator* berhasil meningkatkan metrik pengujian algoritma tersebut ke angka 99,90%. Di saat yang bersamaan, optimasi validasi silang pada model Support Vector Machine menggunakan konfigurasi parameter C, pengaturan koefisien gamma, dan perizinan kernel RBF mampu mencatatkan peningkatan kinerja secara drastis hingga menduduki skor metrik tertinggi yakni senilai 99,92%.

### 4.1.6 Pemilihan Model Terbaik

Berdasarkan keseluruhan capaian yang diperoleh dari keseluruhan proses pelatihan dan optimasi tuning, algoritma Support Vector Machine Tuned dipilih sebagai model identifikasi terbaik untuk melandasi arsitektur sistem deteksi ini. Perbandingan komperhensif antara nilai capaian metrik ujicoba evaluasi yang meliputi Akurasi, Presisi, Recall, dan F1-Score dicantumkan ke dalam bagan komparatif pada Gambar 4.6.

![Grafik Komparasi Detail](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/perbandingan_algoritma_detail_semua.png)
*Gambar 4.6 Histogram Metrik Skoring Menyeluruh terhadap Model Uji*

Validasi pemilihan model SVM didukung pula saat mengkomparasikan kuantitas margin defisit tebakan atau eror kekeliruan prediksi berwujud False Negative maupun False Positive. Angka ketidakakuratan prediksi SVM terbukti mampu meredam kerugian yang disajikan secara kontras oleh sistem pelaporan seperti yang tergambar pada paparan di Gambar 4.7.

![Grafik Error FN FP](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/perbandingan_tp_tn_fp_fn.png)
*Gambar 4.7 Distribusi Output Kalkulasi Keputusan Prediksi TP, TN, FP, dan FN*

Hasil di atas membuktikan bahwa model algoritma SVM dengan kombinasi pencarian parameter utuh sanggup meminimalkan tingkat kekeliruan prediksi menjadi unit satuan numerik terendah per klasifikasinya, sehingga parameter tersebut ditetapkan sebagai parameter mutlak model pada fase siap pakai (deployment).

### 4.1.7 Implementasi Sistem Web

Tahap akhir dari pemodelan adalah merealisasikan algoritma model menjadi modul aplikasi berbasis web yang fungsional menggunakan bahasa pemrograman Python sebagai perancang komponen interaksi (backend). Konstruksi server backend difokuskan secara eksklusif agar memecah format struktur teks URL yang dilemparkan pada parameter masukan dari sisi antarmuka klien, guna diarahkan dalam proses konversi matriks matematis oleh SVM untuk merinci hasil klasifikasi dalam hitungan waktu nyata (real-time). Kerangka antar rupa program (User Interface) dirancang dari kerangka susunan desain Tailwind CSS. Sistem berhasil melakukan demonstrasi operasional di mana masukan URL dari luar lingkungan model langsung dibedah dan disuguhkan peringatan identifikasi berwujud status legitimate yang aman atau peringatan terindikasi kelas phishing.

<!-- Silakan lekatkan tangkapan layar wujud antarmuka aplikasi Anda (screenshot program web browser) di area kosong ini pada dokumen Word. -->

## 4.2 Pembahasan

Bagian ini berisi uraian tentang analisis dan observasi dari hasil yang telah diuraikan pada bagian sebelumnya. Uraian analisis tersebut dilakukan untuk menerjemahkan kumpulan nilai kuantitatif eksperimen yang ada, menjelaskannya secara kausatif, serta mengarahkannya pada sintesis suatu kesimpulan logis yang menjawab rumusan masalah.

### 4.2.1 Analisis Dampak Fitur Leksikal Kustom

Ekstraksi parameter fitur dari pemodelan grafik Gambar 4.3 menjelaskan tingginya representasi kecenderungan keberadaan string menyerupai kata kunci penunjuk arah protokol internet 'http' di ranah komponen susunan level root dari suatu tautan phishing. Observasi atas keberadaan ini mengungkap secara sistematis siasat dari penyerang maya yang selalu memanfaatkan insting dan memanipulasi persepsi visual (social engineering), dengan harapan besar agar korban memiliki praduga telah mengakses titik laman bersertifikat aman saat mengakses suatu tautan panjang. Kemampuan model untuk membaca fenomena ini sebagai anomali esensial sangat membantu algoritma agar tidak menggeneralisasi tautan secara kaku hanya menggunakan ukuran dimensi kepanjangan domain konvensional.

### 4.2.2 Analisis Performa dan Kesalahan Klasifikasi (Confusion Matrix)

Titik kemenangan performa model SVM dalam tahap evaluasi tidak semata-mata dimonopoli oleh indikator capaian akurasi persentase semata, melainkan perlu dianalisa lewat rasio kegagalan matrikular atau Heatmap Confusion Matrix. Visualisasi matriks untuk tes keakuratan terakhir SVM diukur atas penempatan uji dataset acak sebesar 14.952 korpus yang tampak pada Gambar 4.8.

![Confusion Matrix SVM](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/grafik_confusion_matrix_svm_tuned.png)
*Gambar 4.8 Heatmap Sebaran Blok Matriks Prediksi Kesalahan Support Vector Machine*

Hasil peninjauan menunjukkan nilai True Positive serta True Negative mengelompok pada blok keberhasilan dengan angka absolut saling ekuivalen sebesar 7.470 unit data. Sistem ini terbukti tidak terjatuh kepada bias mayoritas maupun tebakan satu belah sepihak saat melakukan eksekusi prediksinya. Kegagalan margin hanya menderita defisit sebanyak 6 luaran pada luput label klasifikasi aman (False Positive). Angka defisit yang sama yakni luput 6 klasifikasi terjadi pada pelabelan tebakan luput kelas target bahaya (False Negative). Fenomena kerataan nilai deteksi secara observasional ini dapat ditafsirkan sebagai efek stabil dan efisien dari keberhasilan modifikasi fungsi Radial Basis Function (RBF) yang difungsikan kernel non-linier agar mampu menggeser kompleksitas parameter yang kusut menjadi dimensi separasi garis margin luang dan dinamis. 

### 4.2.3 Analisis Signifikansi Variabel Fitur (Feature Importance)

Pemahaman arsitektur yang bekerja secara non-abstrak di dalam menarik klasifikasi SVM dievaluasi untuk menghindari perumusan tebak layaknya kotak hitam (Black-Box mechanism). Penilaian yang digunakan untuk memilah tingkatan urgensitas fungsi leksikal dilakukan via teknik de-konstruksi abstraksi variabel numerik atau Permutation Feature Importance yang dilampirkan seperti pada urutan diagram Gambar 4.9.

![Feature Importance SVM](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/feature_importance_svm_tuned.png)
*Gambar 4.9 Kalkulasi Nilai Kohesi Keberpengaruhan Permutasi pada Sumbu Keputusan Deskriptif SVM*

Hasil kuantitas abstraksi fitur mendemonstrasikan penurunan batas akurasi tertinggi dialami seketika nilai variabel url_length, domain_name_length, dan token_count diamputasi dari tubuh model. Asosiasi dominansi hierarkis yang dikandung ketiga fitur di awal secara fundamental memang mendeskripsikan sifat struktural penyusupan URL Phishing di pangkalan web publik, yang mana eksistensi kepanjangan string disengajakan berlimpah dan bertumpuk (obfuscation) agar mampu mengkover kelengkapan string asli dari pelabel filter direktori publik. Observasi ini mempertegas fakta efektivitas penggunaan leksikal terfokus dalam skenario deteksi tautan ancaman dunia maya tanpa intervensi data proksimal jaringan yang berat komputasinya.

### 4.2.4 Observasi Validasi Prediksi Skala Masal (Bulk Audit) 

Sebagai pembuktian kemampuan operasional dan keampuhan sesungguhnya pada ruang lingkup data empirik yang lebih luas, peluncuran simulasi ulang dengan kapabilitas eksekusi Bulk Audit direalisasikan pada keseluruhan kerangka eksperimental ke arah dataset orisinal murni sebelum ditangani operasi distribusi. Hasil temuan kemampuan prediksi deteksi berhadapan secara langsung terhadap 99.564 kelas murni disajikan ke bagan visual donat lingkaran yang termaktub di Gambar 4.10.

![Grafik Doughnut Audit](file:///d:/SEMESTER%207/SKRIPSI/model5/OUTPUT/audit_massal_doughnut.png)
*Gambar 4.10 Hasil Rekaman Performa Prediksi Deteksi Audit Komputasi pada Seluruh Entitas Dataset Murni*

Temuan akhir pada grafik evaluasi ini menjelaskan bertahannya integritas model deteksi saat memperoleh metrik keakuratan tinggi (sebesar nyaris 99,94%), pun ketika dipertentangkan kembali menghadapi beban proporsi kelas asli korpus yang membebani model secara bias. Kestabilan pada arsitektur berbasis evaluasi GridSearchCV parameter tersebut memberi satu rujukan analitis penutup yang menjawab pertanyaan di rumusan masalah. Observasi membuktikan rancangan model mesin algoritma cerdas Support Vector Machine mampu memberikan hasil kapabilitas dan reabilitas yang andal ketika dipergunakan sebagai filter eksternal penjagaan ancaman siber phishing skala masif melalui metode pendeteksian ekstraksi susunan teks dari kerangka URL leksikal, di mana seluruh prosedurnya bebas interupsi manual, objektif, optimal, dan akurat untuk wujud pelarapan produksi rilis daring secara mutlak.
