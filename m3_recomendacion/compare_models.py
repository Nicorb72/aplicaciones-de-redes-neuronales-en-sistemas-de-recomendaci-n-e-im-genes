
"""
Módulo 3: Comparación de estabilidad de modelos.
Responsable: Persona 4.

Genera una figura comparativa de NDCG@3
a partir de los experimentos existentes.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
FIGURES = ROOT / "reports" / "figures" / "m3"

FIGURES.mkdir(parents=True, exist_ok=True)


def main():

    archivos = {
        "Aleatorio": "random_baseline_100_seeds.csv",
        "Colaborativo": "stability_collaborative.csv",
        "Neuronal": "stability_neural.csv"
    }

    resultados = {}

    for nombre, archivo in archivos.items():
        datos = pd.read_csv(PROCESSED / archivo)

        resultados[nombre] = datos["NDCG"].dropna()

    # Diagrama de cajas
    fig, ax = plt.subplots(figsize=(9, 6))

    nombres = list(resultados.keys())
    valores = [resultados[n] for n in nombres]

    ax.boxplot(
        valores,
        tick_labels=nombres,
        patch_artist=True,
        boxprops={
            "facecolor": "#A8C9DD",
            "edgecolor": "#456B83"
        },
        medianprops={
            "color": "#253F52",
            "linewidth": 2
        },
        whiskerprops={"color": "#456B83"},
        capprops={"color": "#456B83"}
    )

    for i, valores_modelo in enumerate(valores, start=1):
        media = valores_modelo.mean()

        ax.scatter(
            i,
            media,
            color="#B4533D",
            marker="D",
            s=50,
            zorder=3
        )

        ax.text(
            i + 0.12,
            media,
            f"{media:.4f}",
            fontsize=10,
            va="center"
        )

    ax.set_title(
        "Variabilidad de NDCG@3 entre modelos",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("NDCG@3")
    ax.set_xlabel("Modelo")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    fig.tight_layout()

    ruta = FIGURES / "fig_m3_07_comparacion_estabilidad.png"

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    resumen = pd.DataFrame({
        nombre: [
            valores.mean(),
            valores.std(),
            valores.min(),
            valores.max(),
            len(valores)
        ]
        for nombre, valores in resultados.items()
    }, index=["Media", "DesvStd", "Min", "Max", "Corridas"])

    print("\n" + "=" * 60)
    print("COMPARACIÓN DE ESTABILIDAD ENTRE MODELOS")
    print("=" * 60)

    print(resumen.round(4).to_string())

    print("\nFigura guardada en:", ruta)


if __name__ == "__main__":
    main()
