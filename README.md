# Simulasi & Animasi Interaktif PSO (Particle Swarm Optimization)
## Optimasi Hyperparameter SVM ($C$ & $\gamma$) pada Dataset Red Wine Quality

Repositori ini dibuat khusus untuk mempermudah pemahaman mahasiswa mengenai cara kerja algoritma **Particle Swarm Optimization (PSO)** secara visual dan interaktif, khususnya dalam mencari hyperparameter terbaik ($C$ dan $\gamma$) pada model Support Vector Classifier (SVC).

---

## 📁 Struktur File

1. **`index.html`** *(Web Animasi Interaktif)*
   - Aplikasi web visual interaktif berbasis HTML5 Canvas, Tailwind CSS, dan Chart.js.
   - **Dapat langsung dibuka di browser apa saja (Chrome, Edge, Safari, Firefox)** tanpa perlu instalasi server atau `pip`.
   - Fitur lengkap:
     - **Pengaturan Skala & Batas**: Bebas mengatur batas minimum dan maksimum $C$ dan $\gamma$.
     - **Mode Skala**: Linear vs Log10 (sangat membantu karena hyperparameter SVM sering diuji dalam skala logaritmik).
     - **Pengaturan Parameter PSO**: Slider untuk Jumlah Partikel ($N$), Max Iterasi, Bobot Inersia ($w$), Koefisien Kognitif ($c_1$), dan Koefisien Sosial ($c_2$).
     - **Preset Pembelajaran**:
       - *Sesuai Source Code* (`pso_wine_svm.py`) ($w=0.9, c_1=0.5, c_2=0.3$)
       - *Standard Swarm Clerc & Kennedy* ($w=0.729, c_1=1.494, c_2=1.494$)
       - *Eksplorasi Tinggi* ($w$ besar)
       - *Konvergensi Cepat* ($c_2$ besar)
     - **Animasi Step-by-Step**: Play, Pause, Langkah Mundur/Maju 1 Iterasi, Scrubbing Slider Timeline, serta Pengaturan Kecepatan Animasi.
     - **Visual Komprehensif**:
       - *Posisi partikel saat ini* (titik biru muda)
       - *Personal Best ($pBest$)* (wajik ungu)
       - *Global Best ($gBest$)* (bintang emas)
       - *Vektor arah kecepatan* (panah merah)
       - *Heatmap / Contour Fitness Landscape* (hijau = optimum, merah = loss tinggi)
     - **Grafik Konvergensi Real-Time & Tabel Evaluasi per Iterasi**.
     - **Inspektur Partikel**: Klik atau hover partikel pada canvas untuk menginspeksi nilai $C$, $\gamma$, Loss, dan Vektor Kecepatannya.
     - **Modal Visualisasi Dataset**: Eksplorasi karakteristik dataset Wine Quality UCI (distribusi 6 kelas target, deteksi outlier metode IQR 1.5x, dan statistik 5-angka untuk 11 fitur fisikokimia).
     - **Upload JSON Hasil Python**: Mahasiswa dapat memuat file `pso_results.json` dari hasil running skrip Python asli ke dalam web ini.

2. **`pso_wine_svm.py`** *(Skrip Python Asli)*
   - Source code Python menggunakan library `pyswarms`, `scikit-learn`, `pandas`, dan `matplotlib`.
   - Mengunduh dataset Red Wine Quality langsung dari UCI Machine Learning Repository.
   - Menjalankan 5-Fold Cross Validation dengan metrik $1 - \text{F1-score}$.
   - Otomatis mengekspor hasil ke `pso_results.json` dan menampilkan grafik matplotlib.

---

## 🚀 Cara Menggunakan

### 1. Menjalankan Web Animasi (Paling Mudah)
Cukup **klik dua kali** file `index.html` pada komputer Anda, atau buka file tersebut menggunakan browser pilihan Anda (Google Chrome, Firefox, Microsoft Edge, Safari).

### 2. Menjalankan Skrip Python
Pastikan dependencies terpasang:
```bash
pip install pyswarms scikit-learn pandas numpy matplotlib seaborn
```

Jalankan skrip:
```bash
python pso_wine_svm.py
```

Setelah selesai:
- Grafik konvergensi dan scatter pergerakan partikel akan muncul di Matplotlib.
- File `pso_results.json` akan otomatis dibuat.
- Anda dapat mengunggah file `pso_results.json` tersebut ke `index.html` melalui tombol **"Upload JSON"** di panel kiri!

---

## 🎓 Panduan Diskusi & Praktikum untuk Mahasiswa

Ajak mahasiswa mengamati fenomena berikut pada web animasi:

1. **Pengaruh Bobot Inersia ($w$):**
   - Naikkan $w > 1.1$: Partikel akan terbang liar keluar batas (eksplorasi sangat kuat tapi sulit konvergen).
   - Turunkan $w < 0.3$: Partikel cepat kehilangan momentum dan berhenti bergerak.

2. **Kognitif ($c_1$) vs Sosial ($c_2$):**
   - Set $c_1$ tinggi dan $c_2 = 0$: Partikel hanya berputar-putar di sekitar pengalaman terbaik dirinya sendiri (selfish).
   - Set $c_2$ tinggi dan $c_1 = 0$: Seluruh kawanan langsung tertarik agresif ke partikel yang menemukan titik terbaik pertama kali (swarm mentality / konvergensi prematur).

3. **Perubahan Skala (Linear vs Log10):**
   - Amati bagaimana distribusi partikel terlihat ketika tombol **Log10** ditekan, yang merefleksikan bagaimana hyperparameter $C$ dan $\gamma$ dieksplorasi secara eksponensial.
