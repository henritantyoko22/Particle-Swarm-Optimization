import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pyswarms as ps

def main():
    # 1. LOAD DATASET RED WINE QUALITY
    print("=" * 60)
    print("1. MEMUAT DATASET RED WINE QUALITY")
    print("=" * 60)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    wine_data = pd.read_csv(url, sep=';')
    print(f"Dataset berhasil dimuat: {wine_data.shape[0]} baris, {wine_data.shape[1]} kolom")

    X = wine_data.drop('quality', axis=1)
    y = wine_data['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standarisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. FITNESS FUNCTION (Loss = 1 - Weighted F1-score)
    def f_per_particle(m, X_tr, y_tr):
        c_val = float(m[0])
        gamma_val = float(m[1])
        
        # Penalti jika nilai parameter tidak valid / negatif
        if c_val <= 0 or gamma_val <= 0:
            return 1.0 
        
        clf = SVC(C=c_val, gamma=gamma_val, random_state=42)
        # Menggunakan 5-fold cross validation
        scores = cross_val_score(clf, X_tr, y_tr, cv=5, scoring='f1_weighted', n_jobs=-1)
        return float(1.0 - np.mean(scores))

    def fitness_function(swarm):
        n_particles = swarm.shape[0]
        j = [f_per_particle(swarm[i], X_train_scaled, y_train) for i in range(n_particles)]
        return np.array(j)

    # 3. SETTING PARAMETER PSO
    print("\n" + "=" * 60)
    print("2. KONFIGURASI PARAMETER PSO")
    print("=" * 60)
    min_bound = np.array([0.1, 0.001])
    max_bound = np.array([50.0, 1.0])
    bounds = (min_bound, max_bound)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    n_particles = 10
    iters = 15

    print(f"Bounds C     : [{min_bound[0]}, {max_bound[0]}]")
    print(f"Bounds Gamma : [{min_bound[1]}, {max_bound[1]}]")
    print(f"Parameters   : w={options['w']}, c1={options['c1']}, c2={options['c2']}")
    print(f"Swarm Size   : {n_particles} partikel, {iters} iterasi")

    # 4. RUN PSO OPTIMIZATION
    print("\n" + "=" * 60)
    print("3. MEMULAI OPTIMASI PSO...")
    print("=" * 60)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options=options,
        bounds=bounds
    )
    best_cost, best_pos = optimizer.optimize(fitness_function, iters=iters)

    cost_history = optimizer.cost_history
    pos_history = np.array(optimizer.pos_history)  # Shape: (iters, n_particles, 2)

    # 5. PRINT TABEL EVALUASI PER ITERASI
    print("\n" + "=" * 60)
    print("TABEL CAPTURE EVALUASI PER ITERASI")
    print("=" * 60)
    print(f"{'Iterasi':<10} | {'Best Loss (1-F1)':<18} | {'F1-Score (%)':<15}")
    print("-" * 60)
    for it, cost in enumerate(cost_history, 1):
        f1_percentage = (1.0 - cost) * 100
        print(f"Iterasi {it:<2}  | {cost:<18.4f} | {f1_percentage:<15.2f}%")

    print("-" * 60)
    print(f"Hasil Terbaik: C = {best_pos[0]:.4f}, Gamma = {best_pos[1]:.5f}")
    print(f"Best Loss: {best_cost:.4f} (F1-Score: {(1.0 - best_cost)*100:.2f}%)")

    # 6. SIMPAN DATA KE JSON AGAR BISA DILIHAT DI WEB ANIMASI
    export_payload = {
        "dataset": "Wine Quality Red",
        "bounds": {
            "cMin": float(min_bound[0]),
            "cMax": float(max_bound[0]),
            "gMin": float(min_bound[1]),
            "gMax": float(max_bound[1])
        },
        "params": {
            "nParticles": n_particles,
            "maxIters": iters,
            "w": options['w'],
            "c1": options['c1'],
            "c2": options['c2']
        },
        "costHistory": [float(c) for c in cost_history],
        "bestPos": [float(best_pos[0]), float(best_pos[1])],
        "posHistory": pos_history.tolist()  # [iter][particle][dim]
    }

    output_json_path = "pso_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[INFO] Data riwayat iterasi berhasil disimpan ke '{output_json_path}'!")
    print("[INFO] Buka 'index.html' di browser Anda untuk melihat animasi pergerakannya secara interaktif!")

    # 7. VISUALISASI MATPLOTLIB STATIS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # A. Grafik Konvergensi
    iterations = range(1, len(cost_history) + 1)
    ax1.plot(iterations, cost_history, marker='o', color='b', linewidth=2, markersize=6)
    ax1.set_title('Grafik Konvergensi PSO (Loss vs Iterasi)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iterasi ke-')
    ax1.set_ylabel('Validation Loss (1 - F1 Score)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    min_idx = np.argmin(cost_history)
    ax1.annotate(
        f'Best Loss: {cost_history[min_idx]:.4f}', 
        xy=(min_idx + 1, cost_history[min_idx]), 
        xytext=(min_idx + 2, cost_history[min_idx] + 0.01),
        arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6)
    )

    # B. Pergerakan Partikel di Ruang Pencarian (C vs Gamma)
    for p in range(pos_history.shape[1]):
        ax2.plot(pos_history[:, p, 0], pos_history[:, p, 1], alpha=0.4, linestyle=':')
        ax2.scatter(pos_history[0, p, 0], pos_history[0, p, 1], color='gray', s=30, zorder=2)
        ax2.scatter(pos_history[-1, p, 0], pos_history[-1, p, 1], color='blue', s=40, zorder=3)

    ax2.scatter(best_pos[0], best_pos[1], color='red', s=120, marker='*', zorder=4, label='Global Best')
    ax2.set_title('Pergerakan Partikel (Search Space)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Nilai Parameter C')
    ax2.set_ylabel('Nilai Parameter Gamma')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
