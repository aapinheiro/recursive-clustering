"""
visualizer.py — Visualização espacial 2D e animação do algoritmo RecursiveClustering.

Uso rápido:
    from recursive_kmeans.visualizer import fit_and_visualize

    viz = fit_and_visualize(
        df,
        geoloc_columns=['latitude', 'longitude'],
        vars_encode=['id'],
        min_cluster_size=5,
        max_cluster_size=15,
    )
    anim, fig = viz.animate_splits(interval=1000)
    plt.show()

    fig2, _ = viz.plot_final()
    plt.show()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple, Dict, Any
import sys
import os

try:
    from scipy.spatial import ConvexHull
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from .core import RecursiveClustering
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from core import RecursiveClustering


# ─── Paleta e constantes visuais ────────────────────────────────────────────

_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#A8DADC", "#6D6875", "#B5838D", "#52B788", "#48CAE4",
    "#7209B7", "#F77F00", "#06D6A0", "#118AB2", "#8338EC",
    "#FF6B9D", "#C77DFF", "#4CC9F0", "#F9C74F", "#43AA8B",
]

_COLOR_UNPROCESSED = "#DADDE1"
_COLOR_ACTIVE      = "#FFD166"
_SPLIT_A           = "#FF6B6B"
_SPLIT_B           = "#4ECDC4"

_PHASE_META: Dict[str, Dict] = {
    "init":         {"badge": "#4A90D9", "label": "INÍCIO"},
    "splitting":    {"badge": "#E67E22", "label": "DIVIDINDO"},
    "split_result": {"badge": "#E67E22", "label": "RESULTADO DA DIVISÃO"},
    "split_small":  {"badge": "#E74C3C", "label": "SUBDIVISÃO (pequeno)"},
    "accept_ok":    {"badge": "#27AE60", "label": "CLUSTER ACEITO ✓"},
    "accept_small": {"badge": "#F39C12", "label": "ACEITO (pequeno)"},
    "final":        {"badge": "#2C3E50", "label": "CONCLUÍDO ✓"},
}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _draw_hull(ax, pts: np.ndarray, color: str, alpha: float = 0.18, lw: float = 1.5):
    """Desenha o convex hull de um conjunto de pontos."""
    if len(pts) < 2:
        return
    if len(pts) == 2:
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=0.6)
        return
    if not _SCIPY_AVAILABLE or len(pts) < 3:
        return
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        verts = np.vstack([verts, verts[0]])
        ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=alpha, zorder=2)
        ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw, alpha=0.55, zorder=3)
    except Exception:
        pass


def _color_for(cluster_id: int, color_map: Dict[int, str]) -> str:
    return color_map.get(cluster_id, _COLOR_UNPROCESSED)


# ─── Subclasse instrumentada ─────────────────────────────────────────────────

class _InstrumentedClustering(RecursiveClustering):
    """
    Subclasse de RecursiveClustering que intercepta _recursive_split
    e registra snapshots do estado a cada decisão do algoritmo.
    Cada snapshot é uma fotografia completa dos rótulos de todos os pontos.
    """

    # -- fit ----------------------------------------------------------------

    def fit(self, X, flavor="pandas"):
        if flavor == "pyspark":
            X.cache().count()
            X = X.toPandas()

        coords = X[self.geoloc_columns].values
        n = len(coords)

        # Augmenta dados com índice original na coluna 2 para rastreamento
        self._aug = np.column_stack([coords, np.arange(n, dtype=float)])
        self._n   = n
        self._labels = np.full(n, -1, dtype=int)   # -1 = não atribuído
        self._snapshots: List[Dict[str, Any]] = []
        self._step = 0

        self._snap(
            active=np.arange(n),
            title=f"Início: {n} pontos em 1 único grupo",
            subtitle=f"Parâmetros → min={self.min_cluster_size}, max={self.max_cluster_size}",
            phase="init",
            depth=0,
        )

        self._recursive_split(self._aug, cluster_id=0)
        self.original_df = X

        self._snap(
            active=np.arange(n),
            title=f"Concluído: {len(self.clusters)} clusters finais",
            subtitle="Todos os pontos foram distribuídos com sucesso",
            phase="final",
            depth=0,
        )

    # -- _recursive_split (sobrescreve o pai) --------------------------------

    def _recursive_split(self, data, cluster_id, depth=0, subsplit_attempts_remaining=None):
        if subsplit_attempts_remaining is None:
            subsplit_attempts_remaining = self.max_subsplit_attempts

        idx = data[:, 2].astype(int)
        n   = len(data)

        # --- cluster menor que o mínimo ---
        if n < self.min_cluster_size:
            if subsplit_attempts_remaining > 0 and n > 1:
                for attempt in range(3):
                    km = KMeans(n_clusters=2, random_state=100 + attempt, n_init=10)
                    labs = km.fit_predict(data[:, :2])
                    c1, c2 = data[labs == 0], data[labs == 1]
                    if len(c1) == n or len(c2) == n:
                        continue
                    i1, i2 = c1[:, 2].astype(int), c2[:, 2].astype(int)
                    self._snap(
                        active=idx, split=(i1, i2),
                        title=f"Prof. {depth}: Tentando subdividir cluster pequeno ({n} pts)",
                        subtitle=f"→ {len(i1)} + {len(i2)} pontos",
                        phase="split_small", depth=depth,
                    )
                    self._recursive_split(c1, cluster_id * 2 + 1, depth + 1, subsplit_attempts_remaining - 1)
                    self._recursive_split(c2, cluster_id * 2 + 2, depth + 1, subsplit_attempts_remaining - 1)
                    return
            self._assign(idx, cluster_id)
            self.clusters.append((cluster_id, data))
            self._snap(
                active=idx,
                title=f"Prof. {depth}: Cluster aceito (pequeno, {n} pts)",
                subtitle=f"Abaixo do mínimo ({self.min_cluster_size}) — sem divisão possível",
                phase="accept_small", depth=depth,
            )
            return

        # --- cluster dentro do intervalo ideal ---
        if self.min_cluster_size <= n <= self.max_cluster_size:
            self._assign(idx, cluster_id)
            self.clusters.append((cluster_id, data))
            self._snap(
                active=idx,
                title=f"Prof. {depth}: Cluster aceito ({n} pts ✓)",
                subtitle=f"Dentro do intervalo [{self.min_cluster_size}, {self.max_cluster_size}]",
                phase="accept_ok", depth=depth,
            )
            return

        # --- cluster grande → dividir ---
        self._snap(
            active=idx,
            title=f"Prof. {depth}: Dividindo cluster grande ({n} pts)",
            subtitle=f"Acima do máximo ({self.max_cluster_size}) → KMeans(k=2)",
            phase="splitting", depth=depth,
        )

        for attempt in range(15):
            km   = KMeans(n_clusters=2, random_state=42 + attempt, n_init=10)
            labs = km.fit_predict(data[:, :2])
            c1, c2 = data[labs == 0], data[labs == 1]
            i1, i2 = c1[:, 2].astype(int), c2[:, 2].astype(int)

            split_ok = False

            if len(c1) >= self.min_cluster_size:
                self._snap(
                    active=idx, split=(i1, i2),
                    title=f"Prof. {depth}: Divisão → {len(i1)} + {len(i2)} pontos",
                    subtitle=f"Seed {42 + attempt}: testando subclusters",
                    phase="split_result", depth=depth,
                )
                if len(c1) > self.max_cluster_size:
                    self._recursive_split(c1, cluster_id * 2 + 1, depth + 1)
                else:
                    self._assign(i1, cluster_id * 2 + 1)
                    self.clusters.append((cluster_id * 2 + 1, c1))
                split_ok = True

            if len(c2) >= self.min_cluster_size:
                if len(c2) > self.max_cluster_size:
                    self._recursive_split(c2, cluster_id * 2 + 2, depth + 1)
                else:
                    self._assign(i2, cluster_id * 2 + 2)
                    self.clusters.append((cluster_id * 2 + 2, c2))
                split_ok = True

            if split_ok:
                return

        self._assign(idx, cluster_id)
        self.clusters.append((cluster_id, data))

    # -- get_cluster_dataframe (strip extra col) -----------------------------

    def get_cluster_dataframe(self):
        data_list, centroids = [], []
        for cid, cluster in self.clusters:
            cluster = np.array(cluster)
            centroids.append(cluster[:, :2].mean(axis=0))
            for pt in cluster:
                data_list.append([pt[0], pt[1], cid])
        df = pd.DataFrame(data_list, columns=self.geoloc_columns + ["cluster"])
        df = pd.concat([df, self.original_df.drop(columns=self.geoloc_columns)], axis=1)
        if self.encode:
            df = self._encode_cluster_labels(df)
        return df, np.array(centroids)

    # -- helpers internos ----------------------------------------------------

    def _assign(self, indices: np.ndarray, cluster_id: int):
        self._labels[indices] = cluster_id

    def _snap(self, active, title, subtitle, phase, depth, split=None):
        self._step += 1
        self._snapshots.append({
            "step":    self._step,
            "title":   title,
            "subtitle": subtitle,
            "phase":   phase,
            "depth":   depth,
            "labels":  self._labels.copy(),
            "active":  np.asarray(active),
            "split":   split,   # (idx_a, idx_b) ou None
        })


# ─── Visualizador principal ──────────────────────────────────────────────────

class RecursiveClusteringVisualizer:
    """
    Gera visualizações estáticas e animadas para um _InstrumentedClustering
    já ajustado (após chamar .fit()).

    Parâmetros
    ----------
    rc : _InstrumentedClustering
        Instância após o fit.
    """

    def __init__(self, rc: _InstrumentedClustering):
        self.rc        = rc
        self.coords    = rc._aug[:, :2]
        self.snapshots = rc._snapshots
        self.n         = rc._n
        geo            = rc.geoloc_columns
        self.xlabel    = geo[0]
        self.ylabel    = geo[1] if len(geo) > 1 else ""

        final_ids      = sorted({cid for cid, _ in rc.clusters})
        self.color_map = {cid: _PALETTE[i % len(_PALETTE)] for i, cid in enumerate(final_ids)}

    # ── Plot final estático ────────────────────────────────────────────────

    def plot_final(
        self,
        figsize: Tuple[int, int] = (11, 8),
        show_centroids: bool = True,
        show_hulls: bool = True,
        show_legend: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plota o estado final da clusterização com hulls, centroides e legenda.

        Retorna
        -------
        fig, ax
        """
        clustered_df, centroids = self.rc.get_cluster_dataframe()
        unique_ids = sorted(clustered_df["cluster"].unique())

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F0F4F8")
        fig.patch.set_facecolor("white")

        for i, cid in enumerate(unique_ids):
            mask  = clustered_df["cluster"] == cid
            pts   = clustered_df[mask][self.rc.geoloc_columns].values
            if len(pts) == 0:
                continue
            color = self.color_map.get(cid, _PALETTE[i % len(_PALETTE)])

            if show_hulls and len(pts) >= 3:
                _draw_hull(ax, pts, color, alpha=0.20)

            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=color, s=55, zorder=4,
                edgecolors="white", linewidths=0.6,
            )

            if show_centroids:
                cx, cy = pts.mean(axis=0)
                ax.scatter(cx, cy, marker="*", s=220, c=color,
                           edgecolors="black", linewidths=0.8, zorder=6)

        ax.set_xlabel(self.xlabel, fontsize=11)
        ax.set_ylabel(self.ylabel, fontsize=11)
        ax.set_title(
            f"Recursive KMeans — {len(unique_ids)} clusters finais\n"
            f"(min_size={self.rc.min_cluster_size}, max_size={self.rc.max_cluster_size}, "
            f"n={self.n} pontos)",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.grid(True, alpha=0.35, linestyle="--", color="white")

        if show_legend:
            handles = [
                mpatches.Patch(
                    color=self.color_map.get(cid, _PALETTE[i % len(_PALETTE)]),
                    label=f"Cluster {i+1}  ({(clustered_df['cluster'] == cid).sum()} pts)",
                )
                for i, cid in enumerate(unique_ids)
            ]
            ncol = max(1, len(handles) // 10)
            ax.legend(handles=handles, fontsize=8, framealpha=0.85,
                      loc="best", ncol=ncol)

        plt.tight_layout()
        return fig, ax

    # ── Animação dos splits ────────────────────────────────────────────────

    def animate_splits(
        self,
        figsize: Tuple[int, int] = (14, 8),
        interval: int = 1100,
        repeat: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 100,
    ) -> Tuple[animation.FuncAnimation, plt.Figure]:
        """
        Anima o processo de divisão recursiva passo a passo.

        Parâmetros
        ----------
        figsize   : tamanho da figura (largura, altura) em polegadas
        interval  : milissegundos entre frames
        repeat    : reiniciar ao terminar
        save_path : se fornecido, salva como GIF (ex: 'output.gif')
        dpi       : resolução para o arquivo salvo

        Retorna
        -------
        anim, fig
        """
        fig = plt.figure(figsize=figsize, facecolor="white")

        # Layout: área principal (esquerda) + painel de info (direita)
        ax_map  = fig.add_axes([0.03, 0.12, 0.62, 0.82])
        ax_info = fig.add_axes([0.68, 0.12, 0.30, 0.82])
        ax_prog = fig.add_axes([0.03, 0.03, 0.94, 0.05])

        x, y = self.coords[:, 0], self.coords[:, 1]
        xpad = (x.max() - x.min()) * 0.07 or 0.5
        ypad = (y.max() - y.min()) * 0.07 or 0.5
        xlim = (x.min() - xpad, x.max() + xpad)
        ylim = (y.min() - ypad, y.max() + ypad)

        n_snaps = len(self.snapshots)

        def _pt_colors(labels: np.ndarray) -> List[str]:
            return [
                _color_for(labels[i], self.color_map) if labels[i] >= 0
                else _COLOR_UNPROCESSED
                for i in range(self.n)
            ]

        def _draw_frame(snap_idx: int):
            for ax in (ax_map, ax_info, ax_prog):
                ax.cla()
                ax.axis("off")

            snap   = self.snapshots[snap_idx]
            labels = snap["labels"]
            active = snap["active"]
            phase  = snap["phase"]
            split  = snap["split"]       # (idx_a, idx_b) ou None
            depth  = snap["depth"]

            pt_colors = _pt_colors(labels)

            # ── Mapa ────────────────────────────────────────────────────────

            # 1. Pontos inativos (fora do cluster em análise)
            inactive = np.ones(self.n, dtype=bool)
            inactive[active] = False
            if inactive.any():
                ax_map.scatter(
                    x[inactive], y[inactive],
                    c=[pt_colors[i] for i in np.where(inactive)[0]],
                    s=22, alpha=0.45, zorder=2,
                    edgecolors="white", linewidths=0.3,
                )

            # 2. Clusters já finalizados — hull
            finalized_ids = set(labels[labels >= 0])
            active_set    = set(active.tolist())
            for cid in finalized_ids:
                pts_idx = np.where(labels == cid)[0]
                if active_set.isdisjoint(pts_idx):
                    _draw_hull(ax_map, self.coords[pts_idx],
                               _color_for(cid, self.color_map), alpha=0.13)

            # 3. Cluster ativo
            if split is not None:
                ia, ib = split
                ax_map.scatter(x[ia], y[ia], c=_SPLIT_A, s=65, zorder=5,
                               edgecolors="white", linewidths=0.6)
                ax_map.scatter(x[ib], y[ib], c=_SPLIT_B, s=65, zorder=5,
                               edgecolors="white", linewidths=0.6)
                _draw_hull(ax_map, self.coords[ia], _SPLIT_A, alpha=0.20)
                _draw_hull(ax_map, self.coords[ib], _SPLIT_B, alpha=0.20)
            else:
                cid_active = int(labels[active[0]]) if labels[active[0]] >= 0 else -1
                acolor = _color_for(cid_active, self.color_map) if cid_active >= 0 else _COLOR_ACTIVE
                border = "black" if phase in ("accept_ok", "accept_small") else "white"
                ax_map.scatter(
                    x[active], y[active],
                    c=acolor, s=65, zorder=5,
                    edgecolors=border, linewidths=0.9,
                )
                if phase in ("accept_ok", "accept_small"):
                    _draw_hull(ax_map, self.coords[active], acolor, alpha=0.28)

            ax_map.set_xlim(*xlim)
            ax_map.set_ylim(*ylim)
            ax_map.set_facecolor("#F0F4F8")
            ax_map.grid(True, alpha=0.35, linestyle="--", color="white")
            ax_map.set_xlabel(self.xlabel, fontsize=9, color="#555")
            ax_map.set_ylabel(self.ylabel, fontsize=9, color="#555")

            # ── Painel de informação ─────────────────────────────────────────

            ax_info.set_xlim(0, 1)
            ax_info.set_ylim(0, 1)
            ax_info.set_facecolor("#F8F9FA")

            meta    = _PHASE_META.get(phase, {"badge": "#888", "label": phase.upper()})
            badge_c = meta["badge"]

            # Badge de fase
            badge = mpatches.FancyBboxPatch(
                (0.05, 0.89), 0.90, 0.09,
                boxstyle="round,pad=0.01",
                facecolor=badge_c, edgecolor="none",
                transform=ax_info.transAxes, clip_on=False,
            )
            ax_info.add_patch(badge)
            ax_info.text(
                0.50, 0.935, meta["label"],
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", transform=ax_info.transAxes,
            )

            # Título e subtítulo
            ax_info.text(
                0.50, 0.80, snap["title"],
                ha="center", va="center", fontsize=8.5, color="#2C3E50",
                transform=ax_info.transAxes, multialignment="center",
                wrap=True,
            )
            ax_info.text(
                0.50, 0.70, snap["subtitle"],
                ha="center", va="center", fontsize=7.5, color="#7F8C8D",
                transform=ax_info.transAxes, multialignment="center",
                style="italic",
            )

            # Separador
            ax_info.axhline(0.64, xmin=0.05, xmax=0.95, color="#DDD", lw=0.8)

            # Estatísticas
            n_done    = int((labels >= 0).sum())
            n_pending = int((labels == -1).sum())
            n_cls     = len(set(labels[labels >= 0]))
            stats = [
                ("Pontos processados",  f"{n_done} / {self.n}"),
                ("Pontos pendentes",    f"{n_pending}"),
                ("Clusters aceitos",    f"{n_cls}"),
                ("Profundidade",        f"{depth}"),
                ("Passo",               f"{snap['step']} / {n_snaps}"),
            ]
            y_s = 0.58
            for lbl, val in stats:
                ax_info.text(0.08, y_s, lbl + ":", fontsize=8, color="#555",
                             transform=ax_info.transAxes, va="center")
                ax_info.text(0.92, y_s, val, fontsize=8, fontweight="bold",
                             color="#2C3E50", transform=ax_info.transAxes,
                             va="center", ha="right")
                y_s -= 0.083

            # Separador
            ax_info.axhline(0.20, xmin=0.05, xmax=0.95, color="#DDD", lw=0.8)

            # Legenda de cores
            legend_items = [
                (_COLOR_UNPROCESSED, "Não processado"),
                (_COLOR_ACTIVE,      "Em análise"),
                (_SPLIT_A,           "Subcluster A"),
                (_SPLIT_B,           "Subcluster B"),
                ("#27AE60",          "Cluster aceito"),
            ]
            handles = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=9, label=lbl)
                for c, lbl in legend_items
            ]
            ax_info.legend(
                handles=handles, loc="lower center", fontsize=7.5,
                framealpha=0.0, handletextpad=0.4, labelspacing=0.4,
            )

            # ── Barra de progresso ───────────────────────────────────────────

            ax_prog.set_xlim(0, 1)
            ax_prog.set_ylim(0, 1)
            ax_prog.set_facecolor("#F8F9FA")

            progress = (snap_idx + 1) / n_snaps
            bg = mpatches.FancyBboxPatch(
                (0, 0.15), 1.0, 0.70,
                boxstyle="round,pad=0.01",
                facecolor="#E0E3E7", edgecolor="none",
                transform=ax_prog.transAxes, clip_on=False,
            )
            ax_prog.add_patch(bg)
            if progress > 0.005:
                fill = mpatches.FancyBboxPatch(
                    (0, 0.15), progress, 0.70,
                    boxstyle="round,pad=0.01",
                    facecolor=badge_c, edgecolor="none",
                    transform=ax_prog.transAxes, clip_on=False,
                )
                ax_prog.add_patch(fill)
            ax_prog.text(
                0.50, 0.50,
                f"Passo {snap['step']} / {n_snaps}  —  {int(progress * 100)}%",
                ha="center", va="center", fontsize=8, color="#2C3E50",
                fontweight="bold", transform=ax_prog.transAxes,
            )

        def _update(frame: int):
            _draw_frame(frame)

        anim = animation.FuncAnimation(
            fig, _update,
            frames=n_snaps,
            interval=interval,
            repeat=repeat,
            blit=False,
        )

        if save_path:
            fps    = max(1, int(1000 / interval))
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=dpi)
            print(f"Animação salva em: {save_path}")

        return anim, fig


# ─── API pública ─────────────────────────────────────────────────────────────

def fit_and_visualize(
    df: pd.DataFrame,
    geoloc_columns: List[str],
    vars_encode: List[str] = None,
    min_cluster_size: int = 5,
    max_cluster_size: int = 15,
    encode: bool = False,
    max_subsplit_attempts: int = 15,
) -> RecursiveClusteringVisualizer:
    """
    Ajusta o RecursiveClustering instrumentado e retorna um visualizador pronto.

    Parâmetros
    ----------
    df               : DataFrame com os dados
    geoloc_columns   : colunas de coordenadas (ex: ['latitude', 'longitude'])
    vars_encode      : colunas auxiliares para encoding (padrão: [])
    min_cluster_size : tamanho mínimo dos clusters
    max_cluster_size : tamanho máximo dos clusters
    encode           : gerar cluster_encoded
    max_subsplit_attempts : tentativas para clusters pequenos

    Retorna
    -------
    RecursiveClusteringVisualizer

    Exemplo
    -------
    >>> viz = fit_and_visualize(df, ['lat', 'lon'], min_cluster_size=5, max_cluster_size=15)
    >>> anim, fig = viz.animate_splits(interval=900)
    >>> plt.show()
    """
    if vars_encode is None:
        vars_encode = []

    rc = _InstrumentedClustering(
        geoloc_columns=geoloc_columns,
        vars_encode=vars_encode,
        encode=encode,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        max_subsplit_attempts=max_subsplit_attempts,
    )
    rc.fit(df)
    return RecursiveClusteringVisualizer(rc)
