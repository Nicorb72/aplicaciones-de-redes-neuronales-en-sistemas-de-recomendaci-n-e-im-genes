
"""
Módulo 3: Concentración de recomendaciones.

Analiza la distribución de recomendaciones del modelo C5
para usuarios conocidos en entrenamiento.

No utiliza datos de prueba.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from m3_recomendacion.train import NeuralRecommendationSystem


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"
FIGURES = ROOT / "reports" / "figures" / "m3"


def main():

    FIGURES.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    # Arquitectura C5 y semilla reproducible.
    modelo = NeuralRecommendationSystem(
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        lr=0.001,
        epochs=100,
        seed=42,
        patience=15,
        negative_ratio=None
    )

    modelo.fit(
        train,
        catalogo,
        usuarios,
        val=val,
        k=3
    )

    nombres = dict(zip(
        catalogo["ItemID"].astype(int),
        catalogo["Name"]
    ))

    conteos = {
        int(item): 0
        for item in catalogo["ItemID"].unique()
    }

    primeras_posiciones = dict.fromkeys(conteos, 0)

    total_recomendaciones = 0
    usuarios_con_recomendaciones = 0

    for user_id in sorted(train["UserID"].unique()):

        ranking = modelo.recommend(
            int(user_id),
            k=3
        )

        if ranking:
            usuarios_con_recomendaciones += 1

        for posicion, (item, score) in enumerate(
            ranking, start=1
        ):

            conteos[int(item)] += 1
            total_recomendaciones += 1

            if posicion == 1:
                primeras_posiciones[int(item)] += 1

    filas = []

    for item in sorted(conteos):

        apariciones = conteos[item]

        porcentaje = (
            100 * apariciones / total_recomendaciones
            if total_recomendaciones > 0
            else 0
        )

        filas.append({
            "ItemID": item,
            "Destino": nombres[item],
            "Apariciones": apariciones,
            "Porcentaje": porcentaje,
            "PrimerasPosiciones": primeras_posiciones[item]
        })

    tabla = pd.DataFrame(filas)

    tabla = tabla.sort_values(
        "Apariciones",
        ascending=False
    ).reset_index(drop=True)

    total_destinos = len(tabla)

    destinos_recomendados = int(
        (tabla["Apariciones"] > 0).sum()
    )

    cobertura = (
        destinos_recomendados / total_destinos
    )

    participaciones = (
        tabla["Apariciones"] / total_recomendaciones
    )

    concentracion_top2 = (
        participaciones.head(2).sum()
    )

    # Índice Herfindahl-Hirschman:
    # 1/n representa una distribución uniforme.
    hhi = float(
        (participaciones ** 2).sum()
    )

    # Distribución uniforme de referencia.
    hhi_uniforme = 1 / total_destinos

    ruta_csv = (
        PROCESSED / "recommendation_concentration.csv"
    )

    tabla.to_csv(
        ruta_csv,
        index=False
    )

    # --------------------------------------------------------
    # Figura: participación por destino
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5.5))

    barras = ax.bar(
        tabla["Destino"],
        tabla["Porcentaje"],
        color="#A8C9DD",
        edgecolor="#456B83"
    )

    for barra, porcentaje in zip(
        barras, tabla["Porcentaje"]
    ):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.4,
            f"{porcentaje:.1f}%",
            ha="center",
            va="bottom"
        )

    ax.axhline(
        100 / total_destinos,
        color="#B4533D",
        linestyle="--",
        label="Distribución uniforme"
    )

    ax.set_title(
        "Concentración de recomendaciones por destino"
    )
    ax.set_xlabel("Destino turístico")
    ax.set_ylabel("Participación en recomendaciones (%)")

    ax.tick_params(
        axis="x",
        rotation=20
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    ruta_figura = (
        FIGURES
        / "fig_m3_08_concentracion_recomendaciones.png"
    )

    fig.savefig(
        ruta_figura,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONCENTRACIÓN DE RECOMENDACIONES")
    print("=" * 60)

    print(
        "Usuarios de entrenamiento:",
        train["UserID"].nunique()
    )

    print(
        "Usuarios con al menos una recomendación:",
        usuarios_con_recomendaciones
    )

    print(
        "Total de recomendaciones emitidas:",
        total_recomendaciones
    )

    print("\nRecomendaciones por destino:")
    print(
        tabla.round(4).to_string(index=False)
    )

    print("\nINDICADORES DE CONCENTRACIÓN")

    print(
        f"Cobertura del catálogo: {cobertura:.4f}"
    )

    print(
        f"Participación de los dos destinos principales: "
        f"{concentracion_top2:.4f}"
    )

    print(
        f"HHI de recomendaciones: {hhi:.4f}"
    )

    print(
        f"HHI uniforme de referencia: "
        f"{hhi_uniforme:.4f}"
    )

    print("\nCSV guardado en:", ruta_csv)
    print("Figura guardada en:", ruta_figura)


if __name__ == "__main__":
    main()
