"""Propósito: Figuras de evaluación del Módulo 1 (error por horizonte, cobertura por ruta, pronóstico vs observado).
Responsable: Persona 2.

Usa la misma paleta y estilo que ``m1_demanda/eda.py`` (Persona 1). Las funciones reciben DataFrames ya
calculados (``final.py``) y devuelven la figura; ``guardar`` la escribe en ``reports/figures/m1/``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_demanda.data.utils import RAIZ, ruta_abs

AZUL, NARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRIS, GRIS_CLARO = "#0b0b0b", "#52514e", "#a3a29c", "#e8e7e3"
COLORES = {"ingenuo": GRIS, "glm": TINTA2, "mlp": AZUL, "rnn": NARANJA}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIS_CLARO, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def _color(modelo: str) -> str:
    return COLORES.get(modelo.split("_")[0], TINTA)


def guardar(fig, cfg_datos: dict, nombre: str):
    ruta = ruta_abs(cfg_datos, "figuras") / nombre
    fig.savefig(ruta)
    plt.close(fig)
    print(f"[fig] {ruta.relative_to(RAIZ)}")
    return ruta


def fig_error_por_horizonte(series: dict[str, pd.DataFrame], particion: str = "test"):
    """MASE por horizonte (1..30). ``series``: {modelo: DataFrame con horizonte y mase (media) y opc. mase_desv}."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for nombre, d in series.items():
        c = _color(nombre)
        ax.plot(d["horizonte"], d["mase"], color=c, label=nombre, lw=1.8 if nombre.startswith(("mlp", "rnn")) else 1.2)
        if "mase_desv" in d and d["mase_desv"].notna().any():
            ax.fill_between(d["horizonte"], d["mase"] - d["mase_desv"], d["mase"] + d["mase_desv"], color=c,
                            alpha=0.15, lw=0)
    ax.axhline(1.0, color=GRIS, lw=0.8, ls="--")
    ax.set_xlabel("Horizonte (días desde el origen)")
    ax.set_ylabel("MASE (1 = ingenuo estacional en entrenamiento)")
    ax.set_title(f"Error por horizonte — partición {particion} (banda = ±1 desv. entre semillas)")
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    return fig


def fig_cobertura_por_ruta(cobertura: dict[str, pd.Series], particion: str = "test", objetivo: float = 0.9):
    """Cobertura del P90 por ruta. ``cobertura``: {modelo: Serie indexada por ruta_id}."""
    modelos = list(cobertura)
    rutas = list(next(iter(cobertura.values())).index)
    x = np.arange(len(rutas))
    ancho = 0.8 / len(modelos)
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    for i, m in enumerate(modelos):
        ax.bar(x + (i - (len(modelos) - 1) / 2) * ancho, cobertura[m].reindex(rutas).to_numpy(), ancho,
               color=_color(m), label=m)
    ax.axhline(objetivo, color=TINTA, lw=1.0, ls="--")
    # La etiqueta va a la izquierda, donde las barras de la primera ruta quedan por debajo de la línea.
    ax.text(-0.45, objetivo + 0.012, f"objetivo {objetivo:.0%}", ha="left", va="bottom", color=TINTA, fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(rutas, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Cobertura del P90 (proporción de días con y ≤ q)")
    ax.set_ylim(0, 1.18)                                # espacio libre arriba para la leyenda, fuera de las barras
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(f"Cobertura del P90 por ruta — partición {particion}")
    ax.legend(ncol=len(modelos), loc="upper center", fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_pronostico_vs_observado(fechas, observado: np.ndarray, pronosticos: dict[str, tuple[np.ndarray, np.ndarray]],
                                titulo: str, ruta_nombre: str):
    """Pronóstico (línea) y P90 (banda) de varios modelos frente a lo observado, para un origen y una ruta."""
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.plot(fechas, observado, color=TINTA, lw=1.8, label="observado", zorder=5)
    for nombre, (mu, q) in pronosticos.items():
        c = _color(nombre)
        ax.plot(fechas, mu, color=c, label=f"{nombre} (media)")
        ax.fill_between(fechas, mu, q, color=c, alpha=0.12, lw=0)
    ax.set_ylabel("Pasajeros por día")
    ax.set_title(f"{titulo} — {ruta_nombre} (banda = media a P90)")
    ax.legend(ncol=3, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
