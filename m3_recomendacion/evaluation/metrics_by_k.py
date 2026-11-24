
"""
Módulo 3: Comparación de métricas según K.

Modelos:
1. Popularidad.
2. Filtrado colaborativo.
3. Recomendador neuronal C5.

Métricas:
- Precision@K
- Recall@K
- NDCG@K
- Cobertura del catálogo

Valores de K: 1, 2, 3, 4 y 5.

Utiliza exclusivamente entrenamiento y validación.
No utiliza test.csv ni cold_test.csv.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from m3_recomendacion.models.popularity import PopularityRecommender
from m3_recomendacion.models.collaborative import CollaborativeRecommender
from m3_recomendacion.experiments.train import NeuralRecommendationSystem
from m3_recomendacion.evaluation.evaluate import evaluar_modelo


# ============================================================
# 1. RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = (
    ROOT / "data" / "m3_recomendacion" / "processed"
)

RAW = (
    ROOT / "data" / "m3_recomendacion" / "raw"
)

FIGURES = (
    ROOT / "reports" / "figures" / "m3"
)


# ============================================================
# 2. CARGA DE DATOS
# ============================================================

def cargar_datos():

    train = pd.read_csv(
        PROCESSED / "train.csv"
    )

    val = pd.read_csv(
        PROCESSED / "val.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    return train, val, catalogo, usuarios


# ============================================================
# 3. ENTRENAMIENTO DE LOS MODELOS
# ============================================================

def entrenar_modelos(train, val, catalogo, usuarios):

    print("\n" + "=" * 60)
    print("PREPARACIÓN DE LOS MODELOS")
    print("=" * 60)

    # --------------------------------------------------------
    # Modelo 1: Popularidad
    # --------------------------------------------------------

    print("\n[1/3] Modelo de popularidad")

    popularidad = PopularityRecommender()

    popularidad.fit(
        train,
        catalogo
    )

    print("Popularidad: preparado correctamente.")

    # --------------------------------------------------------
    # Modelo 2: Filtrado colaborativo
    # --------------------------------------------------------

    print("\n[2/3] Filtrado colaborativo")

    # Utiliza la configuración predeterminada del
    # módulo collaborative.py, establecida previamente:
    # factores=8, lr=0.01, weight_decay=0.001,
    # epochs=100 y seed=42.
    #
    # Si esos valores se modificaron en collaborative.py,
    # deben restablecerse antes de esta comparación.

    colaborativo = CollaborativeRecommender()

    colaborativo.fit(
        train,
        catalogo
    )

    print("Filtrado colaborativo: entrenado correctamente.")

    # --------------------------------------------------------
    # Modelo 3: Recomendador neuronal C5
    # --------------------------------------------------------

    print("\n[3/3] Recomendador neuronal C5")

    neuronal = NeuralRecommendationSystem(
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        lr=0.001,
        epochs=100,
        seed=42,
        patience=15,
        negative_ratio=None
    )

    neuronal.fit(
        train,
        catalogo,
        usuarios,
        val=val,
        k=3
    )

    print("Recomendador neuronal: entrenado correctamente.")

    # --------------------------------------------------------
    # Diccionario de modelos
    # --------------------------------------------------------

    modelos = {
        "Popularidad": popularidad,
        "Colaborativo": colaborativo,
        "Neuronal C5": neuronal
    }

    return modelos


# ============================================================
# 4. EVALUACIÓN PARA DISTINTOS VALORES DE K
# ============================================================

def evaluar_por_k(modelos, val, catalogo):

    resultados = []

    valores_k = [1, 2, 3, 4, 5]

    print("\n" + "=" * 60)
    print("EVALUACIÓN DE LOS MODELOS")
    print("=" * 60)

    for nombre, modelo in modelos.items():

        print(f"\nEvaluando: {nombre}")

        for k in valores_k:

            metricas, _ = evaluar_modelo(
                modelo,
                val,
                catalogo,
                k=k
            )

            resultados.append({
                "Modelo": nombre,
                "K": k,
                "Precision": metricas["Precision@K"],
                "Recall": metricas["Recall@K"],
                "NDCG": metricas["NDCG@K"],
                "Cobertura": metricas["Cobertura"],
                "Usuarios": metricas["Usuarios_evaluados"]
            })

            print(
                f"K={k}"
                f" | Precision={metricas['Precision@K']:.4f}"
                f" | Recall={metricas['Recall@K']:.4f}"
                f" | NDCG={metricas['NDCG@K']:.4f}"
            )

    tabla = pd.DataFrame(resultados)

    return tabla


# ============================================================
# 5. GENERACIÓN DE FIGURAS COMPARATIVAS
# ============================================================

def generar_figuras(tabla):

    FIGURES.mkdir(
        parents=True,
        exist_ok=True
    )

    configuracion = {
        "Precision": {
            "titulo": "Precision@K según K",
            "ylabel": "Precision@K",
            "archivo": "fig_m3_09_precision_por_k.png"
        },

        "Recall": {
            "titulo": "Recall@K según K",
            "ylabel": "Recall@K",
            "archivo": "fig_m3_10_recall_por_k.png"
        },

        "NDCG": {
            "titulo": "NDCG@K según K",
            "ylabel": "NDCG@K",
            "archivo": "fig_m3_11_ndcg_por_k.png"
        }
    }

    estilos = {
        "Popularidad": {
            "color": "#5B8FB9",
            "marker": "o",
            "linestyle": "-"
        },

        "Colaborativo": {
            "color": "#E6A23C",
            "marker": "s",
            "linestyle": "--"
        },

        "Neuronal C5": {
            "color": "#3C9D83",
            "marker": "^",
            "linestyle": "-"
        }
    }

    orden_modelos = [
        "Popularidad",
        "Colaborativo",
        "Neuronal C5"
    ]

    for columna, opciones in configuracion.items():

        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )

        for nombre in orden_modelos:

            grupo = tabla.loc[
                tabla["Modelo"] == nombre
            ].sort_values("K")

            if grupo.empty:
                continue

            estilo = estilos[nombre]

            ax.plot(
                grupo["K"],
                grupo[columna],
                label=nombre,
                color=estilo["color"],
                marker=estilo["marker"],
                linestyle=estilo["linestyle"],
                linewidth=2,
                markersize=7
            )

        ax.set_title(
            opciones["titulo"],
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Número de recomendaciones (K)",
            fontsize=11
        )

        ax.set_ylabel(
            opciones["ylabel"],
            fontsize=11
        )

        ax.set_xticks(
            [1, 2, 3, 4, 5]
        )

        ax.set_ylim(0, 1.05)

        ax.grid(
            alpha=0.25,
            linestyle="--"
        )

        ax.legend(
            title="Modelo",
            loc="best"
        )

        fig.tight_layout()

        ruta = (
            FIGURES / opciones["archivo"]
        )

        fig.savefig(
            ruta,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            "Figura guardada:",
            ruta
        )


# ============================================================
# 6. RESUMEN PARA K = 3
# ============================================================

def mostrar_resumen_k3(tabla):

    resumen = tabla.loc[
        tabla["K"] == 3
    ].copy()

    resumen = resumen.sort_values(
        "NDCG",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("COMPARACIÓN PRINCIPAL PARA K = 3")
    print("=" * 60)

    print(
        resumen.round(4).to_string(
            index=False
        )
    )

    mejor = resumen.iloc[0]

    print(
        "\nModelo con mayor NDCG@3 en esta ejecución:",
        mejor["Modelo"]
    )

    print(
        f"NDCG@3: {mejor['NDCG']:.4f}"
    )

    print(
        "\nNota: esta comparación utiliza una "
        "única semilla para los modelos entrenables."
    )

    print(
        "La estabilidad entre semillas se "
        "analiza en experimentos independientes."
    )


# ============================================================
# 7. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURES.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Cargar datos
    # --------------------------------------------------------

    train, val, catalogo, usuarios = cargar_datos()

    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RECOMENDADORES POR K")
    print("=" * 60)

    print(
        "Interacciones de entrenamiento:",
        len(train)
    )

    print(
        "Interacciones de validación:",
        len(val)
    )

    print(
        "Usuarios evaluados en validación:",
        val["UserID"].nunique()
    )

    print(
        "Destinos disponibles:",
        catalogo["ItemID"].nunique()
    )

    # --------------------------------------------------------
    # Entrenar modelos
    # --------------------------------------------------------

    modelos = entrenar_modelos(
        train,
        val,
        catalogo,
        usuarios
    )

    # --------------------------------------------------------
    # Evaluación por K
    # --------------------------------------------------------

    tabla = evaluar_por_k(
        modelos,
        val,
        catalogo
    )

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    ruta_csv = (
        PROCESSED / "metrics_by_k.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMPARACIÓN DE MÉTRICAS SEGÚN K")
    print("=" * 60)

    print(
        tabla.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Generar gráficas
    # --------------------------------------------------------

    generar_figuras(
        tabla
    )

    # --------------------------------------------------------
    # Resumen K=3
    # --------------------------------------------------------

    mostrar_resumen_k3(
        tabla
    )

    print("\n" + "=" * 60)
    print("ARCHIVOS GENERADOS")
    print("=" * 60)

    print(
        "CSV:",
        ruta_csv
    )

    print(
        "Figuras:",
        FIGURES
    )

    print("\nComparación completada correctamente.")


# ============================================================
# 8. INICIO DEL PROGRAMA
# ============================================================

if __name__ == "__main__":
    main()
