"""
Demonstração do módulo de visualização do RecursiveClustering.

Execute com:
    python examples/visualization_demo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "recursive-kmeans"))
from visualizer import fit_and_visualize


# ─── Dados sintéticos: 4 regiões geográficas com pontos espalhados ──────────

np.random.seed(42)

# 4 centros em coordenadas lat/lon realistas (SP, RJ, BH, Curitiba)
centers = [
    (-23.55, -46.63),  # São Paulo
    (-22.90, -43.17),  # Rio de Janeiro
    (-19.92, -43.94),  # Belo Horizonte
    (-25.43, -49.27),  # Curitiba
]

records = []
for i, (lat, lon) in enumerate(centers):
    n_pts = np.random.randint(20, 40)
    for j in range(n_pts):
        records.append({
            "latitude":  lat  + np.random.normal(0, 0.4),
            "longitude": lon  + np.random.normal(0, 0.5),
            "cliente_id": f"C{i:02d}{j:03d}",
            "regiao": f"Regiao_{i+1}",
        })

df = pd.DataFrame(records)
print(f"Dataset: {len(df)} pontos em 4 regiões")


# ─── Ajuste e visualização ──────────────────────────────────────────────────

viz = fit_and_visualize(
    df,
    geoloc_columns=["latitude", "longitude"],
    vars_encode=["cliente_id"],
    min_cluster_size=10,
    max_cluster_size=20,
    encode=True,
)

clustered_df, centroids = viz.rc.get_cluster_dataframe()
print(f"Clusters gerados: {clustered_df['cluster'].nunique()}")
print(f"Snapshots capturados (frames da animação): {len(viz.snapshots)}")


# ─── Plot 1: Estado final ──────────────────────────────────────────────────

fig1, ax1 = viz.plot_final(figsize=(11, 8))
plt.savefig("examples/output_final.png", dpi=120, bbox_inches="tight")
print("Plot final salvo em examples/output_final.png")


# ─── Plot 2: Animação (exibe interativamente) ──────────────────────────────

anim, fig2 = viz.animate_splits(
    figsize=(14, 8),
    interval=1100,
    repeat=True,
)

# Descomente para salvar como GIF:
# anim, fig2 = viz.animate_splits(save_path="examples/output_animation.gif", interval=1100)

plt.show()
