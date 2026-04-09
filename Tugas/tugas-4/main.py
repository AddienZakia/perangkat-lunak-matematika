import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sympy import *
from scipy.special import digamma, polygamma
from scipy.integrate import quad


# ==========================================
# NOMOR 1
# Gambarkan f(x) = e^(-x^2)
# ==========================================

def nomor_1():
    x_sym = symbols('x')
    fx = exp(-x_sym**2)

    df  = diff(fx, x_sym)      # f'(x)
    ddf = diff(df, x_sym)      # f''(x)

    # Cari titik kritis (f'(x) = 0)
    kritis = solve(df, x_sym)

    # Cari titik infleksi (f''(x) = 0)
    infleksi = solve(ddf, x_sym)

    print("=" * 50)
    print("NOMOR 1 - f(x) = e^(-x^2)")
    print("=" * 50)
    print(f"f'(x)  = {df}")
    print(f"f''(x) = {ddf}")
    print(f"Titik kritis   : x = {kritis}")
    print(f"Titik infleksi : x = {[round(float(v), 4) for v in infleksi]}")

    # Analisis naik/turun
    print("\nAnalisis Naik/Turun:")
    print("  f'(x) = -2x * e^(-x^2)")
    print("  f'(x) > 0  <=>  x < 0  => f naik pada (-inf, 0)")
    print("  f'(x) < 0  <=>  x > 0  => f turun pada (0, +inf)")

    # Analisis cekung
    print("\nAnalisis Cekung:")
    infleksi_val = [float(v) for v in infleksi]
    infleksi_val.sort()
    print(f"  Titik infleksi di x = {[round(v, 4) for v in infleksi_val]}")
    print("  f cekung ke atas  pada (x < -1/sqrt(2)) dan (x > 1/sqrt(2))")
    print("  f cekung ke bawah pada (-1/sqrt(2) < x < 1/sqrt(2))")

    # Plot
    x_vals = np.linspace(-4, 4, 500)
    y_vals = np.exp(-x_vals**2)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x_vals, y_vals, color='steelblue', linewidth=2, label=r'$f(x)=e^{-x^2}$')

    # Asimtot datar y = 0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label='Asimtot datar: y = 0')

    # Tidak ada asimtot tegak (domain semua bilangan real)
    # Tandai titik kritis
    ax.plot(0, 1, 'ko', markersize=6, label='Titik maksimum (0, 1)')

    # Tandai titik infleksi
    for xi in infleksi_val:
        yi = float(fx.subs(x_sym, xi))
        ax.plot(xi, yi, 'r^', markersize=6)

    ax.plot([], [], 'r^', markersize=6, label=f'Titik infleksi x ≈ ±{round(abs(infleksi_val[0]), 4)}')

    ax.set_title(r'Grafik $f(x) = e^{-x^2}$', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ==========================================
# NOMOR 2
# Luas daerah A: y = x+4 dan y = x^2-2
# ==========================================

def nomor_2():
    x_sym = symbols('x')
    f1 = x_sym + 4
    f2 = x_sym**2 - 2

    # Cari titik potong
    intersections = solve(f1 - f2, x_sym)
    a, b = sorted([float(v) for v in intersections])

    print("=" * 50)
    print("NOMOR 2 - Luas Daerah antara y=x+4 dan y=x^2-2")
    print("=" * 50)
    print(f"Titik potong: x = {a}, x = {b}")

    # Hitung luas secara simbolik
    integrand = f1 - f2
    luas = integrate(integrand, (x_sym, a, b))
    print(f"Luas A = integral dari {a} ke {b} dari (x+4 - (x^2-2)) dx = {luas}")

    # Plot
    x_vals = np.linspace(a - 1, b + 1, 500)
    y1 = x_vals + 4
    y2 = x_vals**2 - 2

    x_fill = np.linspace(a, b, 500)
    y1_fill = x_fill + 4
    y2_fill = x_fill**2 - 2

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_vals, y1, color='steelblue', linewidth=2, label=r'$y = x + 4$')
    ax.plot(x_vals, y2, color='tomato', linewidth=2, label=r'$y = x^2 - 2$')
    ax.fill_between(x_fill, y2_fill, y1_fill, alpha=0.3, color='green', label=f'Luas A = {float(luas):.4f}')

    ax.plot([a, b], [a + 4, b + 4], 'ko', markersize=6)
    ax.annotate(f'({a:.1f}, {a+4:.1f})', xy=(a, a + 4), xytext=(a - 0.8, a + 5.5), fontsize=9)
    ax.annotate(f'({b:.1f}, {b+4:.1f})', xy=(b, b + 4), xytext=(b + 0.1, b + 5.0), fontsize=9)

    ax.set_title('Daerah antara $y=x+4$ dan $y=x^2-2$', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/nomor2.png', dpi=150)
    plt.show()
    print("Plot disimpan: nomor2.png\n")


# ==========================================
# NOMOR 3
# Volume Benda Putar - Metode Cakram
# y = 3 + 2x - x^2
# ==========================================

def nomor_3():
    x_sym, y_sym = symbols('x y')
    f = 3 + 2*x_sym - x_sym**2

    # Cari batas (f(x) >= 0)
    roots = solve(f, x_sym)
    x_a, x_b = sorted([float(v) for v in roots])

    print("=" * 50)
    print("NOMOR 3 - Volume Benda Putar y = 3+2x-x^2")
    print("=" * 50)
    print(f"Kurva memotong sumbu-x di x = {x_a} dan x = {x_b}")

    # Cari batas y (f(y) untuk sumbu-y)
    # Ungkapkan x dalam y: y = 3+2x-x^2 => x = 1 ± sqrt(4-y)
    y_max = float(f.subs(x_sym, 1))  # titik puncak di x=1
    print(f"Titik puncak parabola: ({1}, {y_max})")

    # --- a) Putar terhadap sumbu-x ---
    V_a = integrate(pi * f**2, (x_sym, x_a, x_b))
    print(f"\na) Volume putar sumbu-x = {V_a} ≈ {float(V_a):.6f}")

    # --- b) Putar terhadap sumbu-y ---
    # Metode cakram arah y: x dari kanan - kiri
    # x_kanan = 1 + sqrt(4 - y), x_kiri = 1 - sqrt(4 - y)
    x_kanan = 1 + sqrt(4 - y_sym)
    x_kiri  = 1 - sqrt(4 - y_sym)
    V_b = integrate(pi * (x_kanan**2 - x_kiri**2), (y_sym, 0, y_max))
    print(f"b) Volume putar sumbu-y = {V_b} ≈ {float(V_b):.6f}")

    # --- c) Putar terhadap garis y = -1 ---
    V_c = integrate(pi * ((f + 1)**2 - (0 + 1)**2), (x_sym, x_a, x_b))
    print(f"c) Volume putar y = -1  = {V_c} ≈ {float(V_c):.6f}")

    # --- d) Putar terhadap garis x = 4 ---
    # Metode cakram: R(y) = 4 - x_kiri, r(y) = 4 - x_kanan
    R_y = 4 - x_kiri
    r_y = 4 - x_kanan
    V_d = integrate(pi * (R_y**2 - r_y**2), (y_sym, 0, y_max))
    print(f"d) Volume putar x = 4   = {V_d} ≈ {float(V_d):.6f}")

    # --- Ilustrasi 4 subplot ---
    x_vals = np.linspace(x_a, x_b, 300)
    y_vals = 3 + 2*x_vals - x_vals**2

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    titles = [
        'a) Putar terhadap Sumbu-x',
        'b) Putar terhadap Sumbu-y',
        'c) Putar terhadap y = -1',
        'd) Putar terhadap x = 4',
    ]
    hlines = [0, None, -1, None]
    vlines = [None, 0, None, 4]

    for idx, ax in enumerate(axes.flatten()):
        ax.plot(x_vals, y_vals, color='steelblue', linewidth=2, label=r'$y=3+2x-x^2$')
        ax.fill_between(x_vals, 0, y_vals, alpha=0.2, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

        if hlines[idx] is not None:
            ax.axhline(hlines[idx], color='red', linestyle='--', linewidth=1.5,
                       label=f'y = {hlines[idx]}')
        if vlines[idx] is not None:
            ax.axvline(vlines[idx], color='red', linestyle='--', linewidth=1.5,
                       label=f'x = {vlines[idx]}')

        ax.set_title(titles[idx], fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Ilustrasi Volume Benda Putar - Metode Cakram', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/nomor3.png', dpi=150)
    plt.show()
    print("Plot disimpan: nomor3.png\n")


# ==========================================
# NOMOR 4
# Ilustrasi Newton-Raphson mencari optimum
# Kasus konvergen dan tidak konvergen
# ==========================================

def nomor_4():
    print("=" * 50)
    print("NOMOR 4 - Ilustrasi Newton-Raphson Optimum")
    print("=" * 50)

    x_sym = symbols('x')

    def iterasi_nr(fx_expr, x0, max_iter=20):
        df  = diff(fx_expr, x_sym)
        ddf = diff(df, x_sym)

        history = [float(x0)]
        x_n = x0

        for i in range(max_iter):
            df_val  = float(df.subs(x_sym, x_n))
            ddf_val = float(ddf.subs(x_sym, x_n))

            if abs(ddf_val) < 1e-12:
                print(f"  f''(x) ≈ 0, iterasi berhenti di i={i}")
                break

            x_np1 = x_n - df_val / ddf_val
            history.append(float(x_np1))

            if abs(x_np1 - x_n) < 1e-6:
                print(f"  Konvergen di x = {float(x_np1):.6f} setelah {i+1} iterasi")
                break

            x_n = x_np1
        else:
            print(f"  Tidak konvergen setelah {max_iter} iterasi")

        return history

    # Kasus 1: Konvergen - f(x) = x^4 - 4x^2, x0 = 2.5
    fx_konvergen = x_sym**4 - 4*x_sym**2
    x0_konvergen = Float(2.5)
    print("\nKasus Konvergen: f(x) = x^4 - 4x^2, x0 = 2.5")
    hist_conv = iterasi_nr(fx_konvergen, x0_konvergen)

    # Kasus 2: Tidak Konvergen - f(x) = x^3 - 3x, x0 = 0.5 (dekat sadel)
    fx_diverge = x_sym**3 - 3*x_sym
    x0_diverge = Float(0.5)
    print("\nKasus Tidak Konvergen: f(x) = x^3 - 3x, x0 = 0.5")
    hist_div = iterasi_nr(fx_diverge, x0_diverge, max_iter=15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot kasus konvergen ---
    ax = axes[0]
    x_range = np.linspace(-3, 3, 500)
    y_range = x_range**4 - 4*x_range**2
    ax.plot(x_range, y_range, color='steelblue', linewidth=2, label=r'$f(x)=x^4-4x^2$')

    df_conv = diff(fx_konvergen, x_sym)
    for i, xi in enumerate(hist_conv[:-1]):
        yi     = float(fx_konvergen.subs(x_sym, xi))
        dfi    = float(df_conv.subs(x_sym, xi))
        x_next = hist_conv[i + 1]

        ax.plot(xi, yi, 'ro', markersize=6)
        ax.annotate(f'$x_{i}$', xy=(xi, yi), xytext=(xi + 0.1, yi + 0.5), fontsize=8, color='red')
        ax.axvline(xi, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.plot(hist_conv[-1], float(fx_konvergen.subs(x_sym, hist_conv[-1])), 'g*',
            markersize=12, label=f'Konvergen x* ≈ {hist_conv[-1]:.4f}')
    ax.set_title('Newton-Raphson: KONVERGEN\n$f(x)=x^4-4x^2$, $x_0=2.5$', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Plot kasus tidak konvergen ---
    ax = axes[1]
    x_range2 = np.linspace(-3, 3, 500)
    y_range2 = x_range2**3 - 3*x_range2
    ax.plot(x_range2, y_range2, color='tomato', linewidth=2, label=r'$f(x)=x^3-3x$')

    df_div = diff(fx_diverge, x_sym)
    colors_div = plt.cm.plasma(np.linspace(0.1, 0.9, len(hist_div)))
    for i, xi in enumerate(hist_div):
        try:
            yi = float(fx_diverge.subs(x_sym, xi))
            ax.plot(xi, yi, 'o', color=colors_div[i], markersize=6)
            ax.annotate(f'$x_{i}$', xy=(xi, yi), xytext=(xi + 0.1, yi + 0.3), fontsize=8)
        except Exception:
            break

    ax.set_title('Newton-Raphson: TIDAK KONVERGEN\n$f(x)=x^3-3x$, $x_0=0.5$', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Ilustrasi Metode Newton-Raphson untuk Nilai Optimum', fontsize=13)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/nomor4.png', dpi=150)
    plt.show()
    print("Plot disimpan: nomor4.png\n")


# ==========================================
# NOMOR 5
# Nilai minimum, maksimum, dan titik sadel
# f(x,y) = xy - x^3 - y^3
# ==========================================

def nomor_5():
    x_sym, y_sym = symbols('x y')
    f = x_sym * y_sym - x_sym**3 - y_sym**3

    fx  = diff(f, x_sym)    # df/dx
    fy  = diff(f, y_sym)    # df/dy
    fxx = diff(fx, x_sym)
    fyy = diff(fy, y_sym)
    fxy = diff(fx, y_sym)

    # Cari titik kritis
    kritis = solve([fx, fy], [x_sym, y_sym])

    print("=" * 50)
    print("NOMOR 5 - f(x,y) = xy - x^3 - y^3")
    print("=" * 50)
    print(f"fx  = {fx}")
    print(f"fy  = {fy}")
    print(f"fxx = {fxx}")
    print(f"fyy = {fyy}")
    print(f"fxy = {fxy}")
    print(f"Titik kritis: {kritis}")

    # Filter hanya titik kritis real
    kritis_real = [(xp, yp) for xp, yp in kritis if im(xp) == 0 and im(yp) == 0]

    for pt in kritis_real:
        xp, yp = pt
        D = (fxx * fyy - fxy**2).subs([(x_sym, xp), (y_sym, yp)])
        fxx_val = fxx.subs([(x_sym, xp), (y_sym, yp)])
        fp_val  = f.subs([(x_sym, xp), (y_sym, yp)])

        D_val       = float(D.evalf())
        fxx_num     = float(fxx_val.evalf())

        print(f"\nTitik ({float(xp.evalf()):.4f}, {float(yp.evalf()):.4f}):")
        print(f"  D = {D_val:.4f}")
        print(f"  fxx = {fxx_num:.4f}")
        print(f"  f = {float(fp_val.evalf()):.6f}")

        if D_val > 0 and fxx_num < 0:
            print("  => MAKSIMUM RELATIF")
        elif D_val > 0 and fxx_num > 0:
            print("  => MINIMUM RELATIF")
        elif D_val < 0:
            print("  => TITIK SADEL")
        else:
            print("  => Tidak dapat ditentukan (D = 0)")

    # Plot 3D
    x_vals = np.linspace(-1.5, 1.5, 200)
    y_vals = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = X * Y - X**3 - Y**3

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')

    # Tandai titik kritis
    for pt in kritis_real:
        xp, yp = float(pt[0].evalf()), float(pt[1].evalf())
        zp = xp * yp - xp**3 - yp**3
        ax.scatter(xp, yp, zp, color='black', s=60, zorder=5)
        ax.text(xp, yp, zp + 0.1, f'({xp:.2f},{yp:.2f})', fontsize=8)

    fig.colorbar(surf, shrink=0.5, aspect=8)
    ax.set_title(r'$f(x,y) = xy - x^3 - y^3$', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/nomor5.png', dpi=150)
    plt.show()
    print("Plot disimpan: nomor5.png\n")


# ==========================================
# NOMOR 7 - Distribusi Log-Logistik
# ==========================================

def survival_loglogistik(t, alpha, beta):
    return 1 / (1 + (t / alpha)**beta)

def densitas_loglogistik(t, alpha, beta):
    num = (beta / alpha) * (t / alpha)**(beta - 1)
    den = (1 + (t / alpha)**beta)**2
    return num / den

def hazard_loglogistik(t, alpha, beta):
    ft = densitas_loglogistik(t, alpha, beta)
    St = survival_loglogistik(t, alpha, beta)
    return ft / St if St > 0 else 0

def hazard_kumulatif_loglogistik(t, alpha, beta):
    St = survival_loglogistik(t, alpha, beta)
    return -math.log(St) if St > 0 else float('inf')


def nomor_7a(alpha, beta_list, t_vals):
    print("=" * 50)
    print("NOMOR 7a - Tabel Distribusi Log-Logistik")
    print(f"alpha = {alpha}, beta variasi = {beta_list}")
    print("=" * 50)

    # Tulis fungsi survival dan hazard secara simbolik
    t_sym, a_sym, b_sym = symbols('t alpha beta', positive=True)
    F_sym = 1 / (1 + (t_sym / a_sym)**(-b_sym))
    S_sym = 1 - F_sym
    f_sym = diff(F_sym, t_sym)
    h_sym = simplify(f_sym / S_sym)
    H_sym = integrate(h_sym, t_sym)

    print(f"\nS(t) = {simplify(S_sym)}")
    print(f"h(t) = {h_sym}")

    def cetak_tabel(beta):
        print(f"\n--- Tabel untuk beta = {beta} ---")
        header = f"{'t':>6} {'f(t)':>12} {'S(t)':>12} {'h(t)':>12} {'H(t)':>12}"
        print(header)
        print("-" * len(header))

        for t in t_vals:
            ft = densitas_loglogistik(t, alpha, beta)
            St = survival_loglogistik(t, alpha, beta)
            ht = hazard_loglogistik(t, alpha, beta)
            Ht = hazard_kumulatif_loglogistik(t, alpha, beta)
            print(f"{t:>6.2f} {ft:>12.6f} {St:>12.6f} {ht:>12.6f} {Ht:>12.6f}")

    for beta in beta_list:
        cetak_tabel(beta)


def nomor_7b(alpha, beta_list, t_vals):
    print("\n" + "=" * 50)
    print("NOMOR 7b - Grafik Distribusi Log-Logistik")
    print("=" * 50)

    colors = ['steelblue', 'tomato', 'seagreen']
    t_plot = np.linspace(0.01, max(t_vals), 500)

    judul_grafik = ['Fungsi Densitas f(t)', 'Fungsi Survival S(t)',
                    'Fungsi Hazard h(t)', 'Hazard Kumulatif H(t)']
    label_y = ['f(t)', 'S(t)', 'h(t)', 'H(t)']

    fungsi = [
        densitas_loglogistik,
        survival_loglogistik,
        hazard_loglogistik,
        hazard_kumulatif_loglogistik,
    ]

    for idx, (fn, judul, yl) in enumerate(zip(fungsi, judul_grafik, label_y)):
        fig, ax = plt.subplots(figsize=(8, 5))

        for beta, warna in zip(beta_list, colors):
            y_plot = [fn(t, alpha, beta) for t in t_plot]
            ax.plot(t_plot, y_plot, color=warna, linewidth=2, label=f'β = {beta}')

        ax.set_title(f'{judul} - Distribusi Log-Logistik (α={alpha})', fontsize=12)
        ax.set_xlabel('t')
        ax.set_ylabel(yl)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        fname = f'/mnt/user-data/outputs/nomor7_{["densitas","survival","hazard","hazardkum"][idx]}.png'
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"Plot disimpan: {fname}")


# ==========================================
# MAIN - Jalankan semua nomor
# ==========================================

if __name__ == "__main__":
    nomor_1()
    # nomor_2()
    # nomor_3()
    # nomor_4()
    # nomor_5()

    # Nomor 7
    # alpha     = 1
    # beta_list = [0.5, 1.0, 2.0]
    # t_vals    = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    # nomor_7a(alpha, beta_list, t_vals)
    # nomor_7b(alpha, beta_list, t_vals)