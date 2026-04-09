"""
=============================================================
  PREPROCESSING PIPELINE UNTUK FUZZY C-MEANS CLUSTERING
  Studi: COVID-19 & E-Commerce Eropa (Gen X & Gen Y)
=============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

# ─── WARNA TERMINAL ───────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"

def header(text):
    bar = "═" * 60
    print(f"\n{C.CYAN}{C.BOLD}{bar}")
    print(f"  {text}")
    print(f"{bar}{C.RESET}\n")

def section(text):
    print(f"\n{C.BLUE}{C.BOLD}{'─'*50}")
    print(f"  {text}")
    print(f"{'─'*50}{C.RESET}")

def info(text):
    print(f"  {C.WHITE}ℹ  {text}{C.RESET}")

def success(text):
    print(f"  {C.GREEN}✔  {text}{C.RESET}")

def warning(text):
    print(f"  {C.YELLOW}⚠  {text}{C.RESET}")

def error(text):
    print(f"  {C.RED}✘  {text}{C.RESET}")

def prompt(text):
    return input(f"\n  {C.MAGENTA}{C.BOLD}▶  {text}{C.RESET} ")

def show_table(df, title=""):
    if title:
        print(f"\n  {C.DIM}{title}{C.RESET}")
    print()
    # Format untuk terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    lines = df.to_string(index=True).split('\n')
    for line in lines:
        print(f"    {C.DIM}{line}{C.RESET}")
    print()


# ═══════════════════════════════════════════════════════════════
# 1. LOAD CSV
# ═══════════════════════════════════════════════════════════════

def load_csv():
    header("STEP 1 — UPLOAD FILE CSV")

    while True:
        path = prompt("Masukkan path file CSV Anda (contoh: data.csv):").strip()
        if not path:
            error("Path tidak boleh kosong.")
            continue
        if not os.path.exists(path):
            error(f"File tidak ditemukan: '{path}'")
            continue
        if not path.lower().endswith('.csv'):
            warning("File bukan .csv, mencoba membaca tetap...")
        try:
            df = pd.read_csv(path)
            success(f"File berhasil dimuat: {C.CYAN}'{path}'{C.RESET}")
            info(f"Ukuran data  : {df.shape[0]} baris × {df.shape[1]} kolom")
            info(f"Kolom        : {list(df.columns)}")
            show_table(df, "Raw Data:")
            return df, path
        except Exception as e:
            error(f"Gagal membaca file: {e}")


# ═══════════════════════════════════════════════════════════════
# 2. PILIHAN PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def ask_preprocessing(df):
    header("STEP 2 — PILIHAN PREPROCESSING")
    print(f"  {C.WHITE}Apakah Anda ingin melakukan preprocessing pada data ini?{C.RESET}\n")
    print(f"  {C.CYAN}[1]{C.RESET} Ya, lakukan preprocessing (Missing Values → Outlier → Normalisasi)")
    print(f"  {C.CYAN}[2]{C.RESET} Tidak, gunakan data mentah langsung\n")

    while True:
        choice = prompt("Pilihan Anda [1/2]:").strip()
        if choice == '1':
            success("Preprocessing akan dijalankan.")
            return True
        elif choice == '2':
            warning("Preprocessing dilewati. Data akan digunakan dalam kondisi mentah.")
            return False
        else:
            error("Input tidak valid. Masukkan 1 atau 2.")


# ═══════════════════════════════════════════════════════════════
# 3. HANDLE MISSING VALUES
# ═══════════════════════════════════════════════════════════════

def handle_missing(df):
    header("STEP 3 — HANDLE MISSING VALUES")

    # Deteksi kolom numerik
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Tangkap 'N/A' string sebagai missing
    df_check = df.copy()
    for col in df.columns:
        if df_check[col].dtype == object and col not in ['Country']:
            df_check[col] = pd.to_numeric(df_check[col], errors='coerce')

    # ── Laporan Missing ──
    section("Laporan Missing Values")
    total_missing = 0
    missing_info = {}

    for col in df_check.columns:
        n_missing = df_check[col].isna().sum()
        missing_info[col] = n_missing
        if n_missing > 0:
            total_missing += n_missing
            pct = n_missing / len(df_check) * 100
            warning(f"Kolom '{C.YELLOW}{col}{C.RESET}{C.YELLOW}': "
                    f"{n_missing} missing ({pct:.1f}% dari data)")
            # Tunjukkan baris mana
            missing_rows = df_check[df_check[col].isna()].index.tolist()
            info(f"  → Baris index: {missing_rows}")

    if total_missing == 0:
        success("Tidak ditemukan missing value pada data.")
    else:
        print(f"\n  {C.BOLD}{C.YELLOW}Total missing value: {total_missing}{C.RESET}")

    # ── Pilih Strategi ──
    section("Pilih Strategi Handling Missing Values")
    print(f"  {C.CYAN}[1]{C.RESET} Imputasi Median")
    print(f"      → Ganti missing dengan nilai median per kolom")
    print(f"      → Mempertahankan jumlah baris, cocok untuk data kecil\n")
    print(f"  {C.CYAN}[2]{C.RESET} Imputasi Mean")
    print(f"      → Ganti missing dengan nilai rata-rata per kolom")
    print(f"      → Sensitif terhadap outlier\n")
    print(f"  {C.CYAN}[3]{C.RESET} Drop Rows")
    print(f"      → Hapus baris yang mengandung missing value")
    print(f"      → Data lebih bersih tapi jumlah baris berkurang\n")

    while True:
        choice = prompt("Pilihan strategi [1/2/3]:").strip()
        if choice in ['1', '2', '3']:
            break
        error("Input tidak valid. Masukkan 1, 2, atau 3.")

    # ── Terapkan Strategi ──
    df_result = df_check.copy()

    if choice == '1':
        strategy = "Imputasi Median"
        for col in df_check.columns:
            if df_check[col].isna().sum() > 0 and col != 'Country':
                med = df_check[col].median()
                df_result[col] = df_check[col].fillna(med)
                success(f"Kolom '{col}': {missing_info[col]} nilai missing → imputasi median ({med:.2f})")

    elif choice == '2':
        strategy = "Imputasi Mean"
        for col in df_check.columns:
            if df_check[col].isna().sum() > 0 and col != 'Country':
                avg = df_check[col].mean()
                df_result[col] = df_check[col].fillna(avg)
                success(f"Kolom '{col}': {missing_info[col]} nilai missing → imputasi mean ({avg:.2f})")

    else:
        strategy = "Drop Rows"
        before = len(df_result)
        df_result = df_result.dropna()
        after = len(df_result)
        dropped = before - after
        if dropped > 0:
            warning(f"{dropped} baris dihapus karena mengandung missing value.")
        else:
            success("Tidak ada baris yang dihapus.")

    section("Ringkasan Handling Missing Values")
    info(f"Strategi       : {C.CYAN}{strategy}{C.RESET}")
    info(f"Total missing  : {C.YELLOW}{total_missing}{C.RESET}")
    info(f"Baris sebelum  : {len(df_check)}")
    info(f"Baris sesudah  : {len(df_result)}")

    show_table(df_result, "Data setelah handling missing values:")

    print(f"\n  {C.GREEN}{C.BOLD}PENJELASAN:{C.RESET}")
    if choice == '1':
        print(f"  {C.WHITE}Imputasi median dipilih karena robust terhadap outlier dibanding mean.")
        print(f"  Pada data kecil (n=5), menghapus baris akan sangat mengurangi informasi.")
        print(f"  Median memastikan distribusi data tidak terdistorsi oleh nilai ekstrem,")
        print(f"  yang penting agar centroid Fuzzy C-means tidak bias.{C.RESET}")
    elif choice == '2':
        print(f"  {C.WHITE}Imputasi mean mengisi missing value dengan rata-rata kolom.")
        print(f"  Cocok bila data berdistribusi normal tanpa outlier signifikan.")
        print(f"  Perhatikan bahwa mean sensitif terhadap outlier, sehingga bila ada")
        print(f"  nilai ekstrem, centroid Fuzzy C-means bisa terpengaruh.{C.RESET}")
    else:
        print(f"  {C.WHITE}Drop rows memastikan hanya data lengkap yang digunakan.")
        print(f"  Strategi ini tepat bila missing value sedikit (<5% dari data).")
        print(f"  Namun pada dataset kecil, setiap baris yang hilang berdampak besar")
        print(f"  terhadap representasi cluster dalam Fuzzy C-means.{C.RESET}")

    return df_result


# ═══════════════════════════════════════════════════════════════
# 4. DETEKSI & KOREKSI OUTLIER
# ═══════════════════════════════════════════════════════════════

def handle_outlier(df):
    header("STEP 4 — DETEKSI & KOREKSI OUTLIER")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Pilih Metode ──
    section("Pilih Metode Deteksi Outlier")
    print(f"  {C.CYAN}[1]{C.RESET} Z-Score (threshold = 2.5σ)")
    print(f"      → Outlier = nilai dengan |z-score| > 2.5")
    print(f"      → Cocok untuk data berdistribusi normal\n")
    print(f"  {C.CYAN}[2]{C.RESET} IQR Method (faktor = 1.5×IQR)")
    print(f"      → Outlier = nilai di luar [Q1 - 1.5×IQR, Q3 + 1.5×IQR]")
    print(f"      → Lebih robust, cocok untuk distribusi tidak normal\n")

    while True:
        choice = prompt("Pilihan metode [1/2]:").strip()
        if choice in ['1', '2']:
            break
        error("Input tidak valid. Masukkan 1 atau 2.")

    # ── Deteksi & Cap ──
    section("Laporan Deteksi Outlier")
    df_result = df.copy()
    total_outliers = 0
    outlier_log = []

    for col in num_cols:
        vals = df[col].dropna()
        if len(vals) < 2:
            continue

        if choice == '1':
            method = "Z-Score"
            z_scores = np.abs(stats.zscore(vals))
            threshold = 2.5
            outlier_mask = z_scores > threshold
            outlier_idx = vals[outlier_mask].index

            mean_val = vals.mean()
            std_val  = vals.std()
            lower_cap = mean_val - threshold * std_val
            upper_cap = mean_val + threshold * std_val

        else:
            method = "IQR"
            Q1 = vals.quantile(0.25)
            Q3 = vals.quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - 1.5 * IQR
            upper_cap = Q3 + 1.5 * IQR
            outlier_mask = (vals < lower_cap) | (vals > upper_cap)
            outlier_idx = vals[outlier_mask].index

        if len(outlier_idx) > 0:
            for idx in outlier_idx:
                orig_val = df_result.loc[idx, col]
                capped_val = np.clip(orig_val, lower_cap, upper_cap)
                country = df_result.loc[idx, 'Country'] if 'Country' in df_result.columns else idx
                warning(f"Outlier [{method}] — Negara: '{C.YELLOW}{country}{C.RESET}{C.YELLOW}', "
                        f"Kolom: '{col}'")
                info(f"  Nilai asli    : {orig_val}")
                info(f"  Batas cap     : [{lower_cap:.2f}, {upper_cap:.2f}]")
                info(f"  Nilai setelah : {capped_val:.2f}")
                df_result.loc[idx, col] = round(capped_val, 4)
                total_outliers += 1
                outlier_log.append({'Country': country, 'Column': col,
                                    'Original': orig_val, 'Capped': round(capped_val, 4)})

    if total_outliers == 0:
        success("Tidak ditemukan outlier pada semua kolom numerik.")
    else:
        print(f"\n  {C.BOLD}{C.YELLOW}Total outlier ditemukan & dikoreksi: {total_outliers}{C.RESET}")

    # ── Ringkasan ──
    section("Ringkasan Koreksi Outlier")
    info(f"Metode         : {C.CYAN}{'Z-Score (2.5σ)' if choice=='1' else 'IQR (1.5x)'}{C.RESET}")
    info(f"Kolom diperiksa: {len(num_cols)}")
    info(f"Total outlier  : {C.YELLOW}{total_outliers}{C.RESET}")

    if outlier_log:
        log_df = pd.DataFrame(outlier_log)
        show_table(log_df, "Detail outlier yang dikoreksi (winsorization):")

    show_table(df_result, "Data setelah koreksi outlier:")

    print(f"\n  {C.GREEN}{C.BOLD}PENJELASAN:{C.RESET}")
    print(f"  {C.WHITE}Outlier berbahaya untuk Fuzzy C-means karena algoritma ini")
    print(f"  menggunakan jarak Euclidean dalam perhitungan keanggotaan cluster.")
    print(f"  Satu nilai ekstrem bisa menarik centroid cluster jauh dari posisi")
    print(f"  yang seharusnya, menyebabkan cluster tidak merepresentasikan")
    print(f"  pola sebenarnya dalam data e-commerce. Winsorization (capping)")
    print(f"  dipilih karena mempertahankan jumlah data sambil membatasi")
    print(f"  pengaruh nilai ekstrem terhadap perhitungan centroid.{C.RESET}")

    return df_result


# ═══════════════════════════════════════════════════════════════
# 5. NORMALISASI / STANDARISASI
# ═══════════════════════════════════════════════════════════════

def handle_normalize(df):
    header("STEP 5 — NORMALISASI / STANDARISASI")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Statistik sebelum normalisasi ──
    section("Statistik Sebelum Normalisasi")
    stats_before = df[num_cols].describe().loc[['min', 'max', 'mean', 'std']].round(4)
    show_table(stats_before, "Min, Max, Mean, Std sebelum normalisasi:")

    # ── Pilih Metode ──
    section("Pilih Metode Normalisasi")
    print(f"  {C.CYAN}[1]{C.RESET} Min-Max Scaling (range 0–1)")
    print(f"      → Formula: (x - min) / (max - min)")
    print(f"      → Semua nilai dipetakan ke rentang [0, 1]")
    print(f"      → Direkomendasikan untuk Fuzzy C-means\n")
    print(f"  {C.CYAN}[2]{C.RESET} Z-Score Standardization")
    print(f"      → Formula: (x - mean) / std")
    print(f"      → Hasil: mean=0, std=1")
    print(f"      → Cocok bila data berdistribusi normal\n")

    while True:
        choice = prompt("Pilihan metode [1/2]:").strip()
        if choice in ['1', '2']:
            break
        error("Input tidak valid. Masukkan 1 atau 2.")

    # ── Terapkan ──
    df_result = df.copy()
    norm_log = []

    for col in num_cols:
        vals = df[col].dropna()
        orig_min = vals.min()
        orig_max = vals.max()
        orig_mean = vals.mean()
        orig_std = vals.std()

        if choice == '1':
            method = "Min-Max"
            if orig_max == orig_min:
                df_result[col] = 0.0
                new_min, new_max = 0.0, 0.0
            else:
                df_result[col] = (df[col] - orig_min) / (orig_max - orig_min)
                new_min = df_result[col].min()
                new_max = df_result[col].max()
        else:
            method = "Z-Score"
            if orig_std == 0:
                df_result[col] = 0.0
                new_min, new_max = 0.0, 0.0
            else:
                df_result[col] = (df[col] - orig_mean) / orig_std
                new_min = df_result[col].min()
                new_max = df_result[col].max()

        df_result[col] = df_result[col].round(6)
        norm_log.append({
            'Kolom': col,
            'Min (asli)': round(orig_min, 2),
            'Max (asli)': round(orig_max, 2),
            'Min (norm)': round(new_min, 4),
            'Max (norm)': round(new_max, 4)
        })
        success(f"Kolom '{col}': [{orig_min:.2f}, {orig_max:.2f}] → [{new_min:.4f}, {new_max:.4f}]")

    # ── Ringkasan ──
    section("Ringkasan Normalisasi")
    info(f"Metode         : {C.CYAN}{'Min-Max Scaling (0–1)' if choice=='1' else 'Z-Score Standardization'}{C.RESET}")
    info(f"Kolom dinormalisasi: {len(num_cols)}")

    log_df = pd.DataFrame(norm_log)
    show_table(log_df, "Perbandingan rentang nilai sebelum vs sesudah normalisasi:")

    show_table(df_result, "Data setelah normalisasi (siap untuk Fuzzy C-means):")

    section("Statistik Setelah Normalisasi")
    stats_after = df_result[num_cols].describe().loc[['min', 'max', 'mean', 'std']].round(6)
    show_table(stats_after, "Min, Max, Mean, Std sesudah normalisasi:")

    print(f"\n  {C.GREEN}{C.BOLD}PENJELASAN:{C.RESET}")
    if choice == '1':
        print(f"  {C.WHITE}Min-Max Scaling sangat direkomendasikan untuk Fuzzy C-means karena")
        print(f"  algoritma ini menghitung derajat keanggotaan berdasarkan jarak antar")
        print(f"  titik data. Tanpa normalisasi, kolom dengan skala besar (misal 0–100)")
        print(f"  akan mendominasi perhitungan jarak dibanding kolom skala kecil (0–10),")
        print(f"  menghasilkan cluster yang bias. Dengan rentang [0,1], setiap fitur")
        print(f"  berkontribusi setara terhadap pembentukan cluster.{C.RESET}")
    else:
        print(f"  {C.WHITE}Z-Score Standardization memastikan semua fitur memiliki mean=0 dan")
        print(f"  std=1, sehingga tidak ada satu fitur pun yang mendominasi jarak Euclidean")
        print(f"  dalam Fuzzy C-means. Metode ini cocok bila distribusi data mendekati")
        print(f"  normal, seperti yang ditunjukkan oleh Shapiro-Wilk test pada paper.{C.RESET}")

    return df_result


# ═══════════════════════════════════════════════════════════════
# 6. SIMPAN HASIL
# ═══════════════════════════════════════════════════════════════

def save_result(df, original_path):
    header("STEP 6 — SIMPAN HASIL")

    base = os.path.splitext(os.path.basename(original_path))[0]
    default_name = f"{base}_preprocessed.csv"

    print(f"  {C.WHITE}File hasil preprocessing akan disimpan sebagai CSV.{C.RESET}")
    out_path = prompt(f"Nama file output (Enter untuk '{default_name}'):").strip()
    if not out_path:
        out_path = default_name
    if not out_path.endswith('.csv'):
        out_path += '.csv'

    df.to_csv(out_path, index=False)
    success(f"File berhasil disimpan: {C.CYAN}'{out_path}'{C.RESET}")
    info(f"Ukuran akhir  : {df.shape[0]} baris × {df.shape[1]} kolom")
    return out_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║      PREPROCESSING PIPELINE — FUZZY C-MEANS CLUSTERING      ║
║      COVID-19 & E-Commerce Eropa · Generasi X & Y           ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")

    # Step 1: Load
    df, path = load_csv()

    # Step 2: Pilihan
    do_preprocess = ask_preprocessing(df)

    if not do_preprocess:
        header("SELESAI — TANPA PREPROCESSING")
        warning("Data digunakan dalam kondisi mentah.")
        show_table(df, "Data final (tanpa preprocessing):")
        save_result(df, path)
    else:
        # Step 3: Missing Values
        df = handle_missing(df)

        # Step 4: Outlier
        df = handle_outlier(df)

        # Step 5: Normalisasi
        df = handle_normalize(df)

        # Step 6: Simpan
        out_path = save_result(df, path)

        header("PREPROCESSING SELESAI")
        success("Semua tahap preprocessing berhasil dijalankan!")
        print()
        info("Ringkasan pipeline:")
        print(f"    {C.CYAN}[3]{C.RESET} Handle Missing Values  {C.GREEN}✔{C.RESET}")
        print(f"    {C.CYAN}[4]{C.RESET} Deteksi & Koreksi Outlier  {C.GREEN}✔{C.RESET}")
        print(f"    {C.CYAN}[5]{C.RESET} Normalisasi / Standarisasi  {C.GREEN}✔{C.RESET}")
        print()
        success(f"Data siap digunakan untuk Fuzzy C-means clustering.")
        info(f"File tersimpan: {C.CYAN}{out_path}{C.RESET}")

    print(f"\n{C.DIM}{'─'*62}{C.RESET}\n")


if __name__ == "__main__":
    main()