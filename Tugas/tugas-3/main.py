import math
import random
import pandas as pd
import json
import matplotlib.pyplot as plt

class Preproccesing:
    def read_data_csv(self):
        df = pd.read_csv(
            './Tugas/tugas-3/data-plm.csv',
            delimiter=',',      
            encoding='utf-8',   
        )

        json_data = df.to_json(orient='records', indent=2)
        json_data = json.loads(json_data)
        result = {}

        for items in json_data:
            val = [int(values) for keys, values in items.items() if keys != 'Country']

            result[items['Country']] = val
        
        self.data = result

class Utilitas(Preproccesing):
    def __init__(self):
        super().__init__()
        self.read_data_csv()
    
    # convert dari Object country data ke Object per fitur
    def to_object_fitur(self):
        keys = ["All_Individuals_2018", "All_Individuals_2021",	"GenY_2018",	"GenY_2021",	
        "GenX_2018",	"GenX_2021",	"Men_GenY_2018",	"Men_GenY_2021",	
        "Men_GenX_2018", "Men_GenX_2021", "Women_GenY_2018", "Women_GenY_2021",	
        "Women_GenX_2018", "Women_GenX_2021"]

        len_fitur = len(list(self.data.values())[0])
        values = [[] for _ in range(len_fitur)]
        object_val = list(self.data.values())

        for i in range(len(object_val)):
            for j in range(len(object_val[i])):
                values[j].append(object_val[i][j])

        result = {}
        for i, k in enumerate(keys):
            result[k] = values[i]

        return result

    def filter_year(self, data):
        YEAR_COLUMNS = {
            2018: [0, 2, 4, 6, 8, 10, 12],   # 7 fitur tahun 2018
            2021: [1, 3, 5, 7, 9, 11, 13],   # 7 fitur tahun 2021
        }

        if self.year is None:
            return data
        cols = YEAR_COLUMNS[self.year]
        return [[row[i] for i in cols] for row in data]

class FCM(Utilitas):
    """
        All_Individuals_2018, All_Individuals_2021,	GenY_2018,	GenY_2021,	
        GenX_2018,	GenX_2021,	Men_GenY_2018,	Men_GenY_2021,	
        Men_GenX_2018, Men_GenX_2021, Women_GenY_2018, Women_GenY_2021,	
        Women_GenX_2018, Women_GenX_2021 (14 features - berurutan)
    """

    def __init__(self, C, m, Year=None):
        super().__init__()
        self.read_data_csv()

        self.total_fitur = len(list(self.data.values()))
        self.total_country = len(list(self.data.keys()))
        self.C = C
        self.m = m
        self.year = Year

    def normalisation_data(self):
        data_country = list(self.data.values())

        if self.year is not None:
            data_country = self.filter_year(data_country)

        data_index = [[] for _ in range(len(data_country[0]))]

        for i in range(len(data_country)):
            for j in range(len(data_country[i])):
                data_index[j].append(data_country[i][j])
        
        min_value = [min(data_index[i]) for i in range(len(data_index))]
        max_value = [max(data_index[i]) for i in range(len(data_index))]

        data_norm = []
        for i in range(len(data_country)):
            data_i = []

            for j in range(len(data_country[i])):
                x_norm = (data_country[i][j] - min_value[j]) / (max_value[j] - min_value[j])
                data_i.append(round(x_norm, 4))
            
            data_norm.append(data_i)
        
        list_country = list(self.data.keys())
        
        result = {}
        for keys, country in enumerate(list_country):
            result[country] = data_norm[keys]
        
        return result
    
    def init_membership(self, n):
        random.seed(42)

        U = []

        for _ in range(n):
            row = [random.random() for _ in range(self.C)]
            total = sum(row)

            row = [round(val / total, 2) for val in row]
            U.append(row)
        
        return U
    
    def centroid_calculation(self, U):
        # initial_bobot = self.init_membership(n=self.total_country)
        norm_data = list(self.normalisation_data().values())
        exp = 2 / (self.m - 1)

        sum_cluster = [0 for _ in range(self.C)]
        Cluster = [[] for _ in range(self.C)] # cluster setelah dikuadrat

        for bobot in U:
            cluster = [round(cluster ** exp, 4) for cluster in bobot]
            
            for keys, c in enumerate(cluster):
                Cluster[keys].append(c)
                sum_cluster[keys] += c

        result_cluster = [
            [
                round(
                    sum(Cluster[i][j] * norm_data[j][k] for j in range(len(norm_data))) / sum_cluster[i],
                    4
                )
                for k in range(len(norm_data[0]))
            ]
            for i in range(len(Cluster))
        ]

        return result_cluster
    
    
    def update_membership(self, U):
        list_centroid = self.centroid_calculation(U)
        norm_data = list(self.normalisation_data().values())

        distance = [
            [
                round(math.sqrt(sum(
                    (norm_data[j][k] - list_centroid[i][k]) ** 2
                    for k in range(len(norm_data[0]))
                )), 4)
                for j in range(len(norm_data))
            ]
            for i in range(len(list_centroid))
        ]

        new_U = []
        for j in range(len(norm_data)):  # tiap data
            row = []

            for i in range(len(list_centroid)):  # tiap cluster
                if distance[i][j] == 0:
                    val = 1
                else:
                    denom = 0
                    for k in range(len(list_centroid)):
                        if distance[k][j] == 0:
                            continue
                        ratio = distance[i][j] / distance[k][j]
                        denom += ratio ** (2 / (self.m - 1))

                    val = 1 / denom if denom != 0 else 0

                row.append(round(val, 4))

            new_U.append(row)

        return new_U
    
    def objective_function(self, U, list_centroid):
        norm_data = list(self.normalisation_data().values())

        J = 0
        for i in range(self.C):
            for j in range(len(norm_data)):
                d_ij = math.sqrt(sum(
                    (norm_data[j][k] - list_centroid[i][k]) ** 2
                    for k in range(len(norm_data[0]))
                ))
                J += (U[j][i] ** self.m) * (d_ij ** 2)

        return round(J, 6)
    
    def fit(self, max_iter=100, epsilon=1e-4):
        norm_data = list(self.normalisation_data().values())
        n = len(norm_data)

        U = self.init_membership(n)
        history_J = []

        print(f"\nFCM | C={self.C}, m={self.m}, ε={epsilon}")
        print(f"{'─'*45}")
        print(f"{'Iterasi':>8} │ {'Objective Function (J)':>22} │ {'ΔU':>12}")
        print(f"{'─'*45}")

        for iterasi in range(1, max_iter + 1):
            U_lama = [row[:] for row in U]

            centroids = self.centroid_calculation(U)
            U = self.update_membership(U)

            J = self.objective_function(U, centroids)
            history_J.append(J)

            delta = math.sqrt(sum(
                (U[j][i] - U_lama[j][i]) ** 2
                for j in range(n)
                for i in range(self.C)
            ))

            print(f"{iterasi:>8} │ {J:>22.6f} │ {delta:>12.6f}")

            if delta < epsilon:
                print(f"{'─'*45}")
                print(f"Konvergen di iterasi {iterasi}")
                break
        else:
            print(f"{'─'*45}")
            print(f"Belum konvergen setelah {max_iter} iterasi")

        self.U_final   = U
        self.centroids = centroids
        self.history_J = history_J
        self.labels    = [row.index(max(row)) for row in U]

        return self

    # ─────────────────────────────────
    #  VISUALISASI SCATTER PLOT
    # ─────────────────────────────────
    def visualize(self, title="FCM Clustering"):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np  # tambah ini

        if not hasattr(self, 'labels'):
            raise RuntimeError("Jalankan .fit() dulu sebelum visualize().")

        norm_data = list(self.normalisation_data().values())
        countries = list(self.data.keys())

        # t-SNE: 14 fitur → 2 dimensi
        # perplexity disesuaikan karena data cuma 35 baris
        norm_array = np.array(norm_data)  # ← tambah ini
        tsne = TSNE(n_components=2, perplexity=4, random_state=42)
        projected = tsne.fit_transform(norm_array)  # shape (35, 2)

        colors = ['steelblue', 'tomato', 'seagreen', 'darkorange',
                'mediumpurple', 'gold', 'deeppink', 'sienna']

        plt.figure(figsize=(11, 7))

        for c in range(self.C):
            idx = [j for j, lbl in enumerate(self.labels) if lbl == c]
            xs  = [projected[j][0] for j in idx]
            ys  = [projected[j][1] for j in idx]

            plt.scatter(xs, ys,
                        color=colors[c % len(colors)],
                        label=f"Cluster {c}",
                        s=100,
                        edgecolors='white',
                        linewidths=0.6,
                        zorder=3)

            # label nama negara
            for j in idx:
                plt.annotate(
                    countries[j],
                    (projected[j][0], projected[j][1]),
                    fontsize=7.5,
                    ha='left', va='bottom',
                    xytext=(5, 5),
                    textcoords='offset points'
                )

        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.title(f"{title} — t-SNE (C={self.C}, m={self.m})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

# FCM.elbow(data_csv_path='./Tugas/tugas-3/data-plm.csv', c_range=range(2, 8))
# semua fitur (14 kolom)
# model_all = FCM(C=3, m=2)
# model_all.fit()
# model_all.visualize(title="2018 & 2021")

# khusus 2018 (7 kolom)
# model_2018 = FCM(C=3, m=2, Year=2018)
# model_2018.fit()
# model_2018.visualize(title="E-Commerce 2018")

# khusus 2021 (7 kolom)
# model_2021 = FCM(C=4, m=2, Year=2021)
# model_2021.fit()
# model_2021.visualize(title="E-Commerce 2021")

def elbow(c_range=range(2, 9), m=2, max_iter=100, epsilon=1e-4):
    J_list = []

    print(f"\n{'─'*40}")
    print(f"{'Elbow Method — Fuzzy C-Means':^40}")
    print(f"{'─'*40}")
    print(f"{'C':>5} │ {'J Akhir':>15} │ {'Iterasi':>8}")
    print(f"{'─'*40}")

    for c in c_range:
        model = FCM(C=c, m=m)
        model.fit(max_iter=max_iter, epsilon=epsilon)
        J_akhir = model.history_J[-1]
        n_iter  = model.n_iter if hasattr(model, 'n_iter') else '-'
        J_list.append(J_akhir)
        print(f"{c:>5} │ {J_akhir:>15.6f} │ {n_iter:>8}")

    print(f"{'─'*40}")

    c_values = list(c_range)
    plt.figure(figsize=(8, 5))
    plt.plot(c_values, J_list, marker='o', color='steelblue', linewidth=2)
    plt.xlabel("Jumlah Cluster (C)")
    plt.ylabel("Objective Function (J)")
    plt.title("Elbow Method — Fuzzy C-Means")
    plt.xticks(c_values)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

elbow(c_range=range(2, 8))