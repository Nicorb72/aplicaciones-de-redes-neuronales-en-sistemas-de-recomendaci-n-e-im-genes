
"""
Módulo 3: Análisis exploratorio de recomendaciones.

Objetivo:
Analizar las características de los usuarios,
destinos e interacciones.

Las figuras se guardan en reports/figures/m3.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Rutas del proyecto
ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"

FIGURES = ROOT / "reports" / "figures" / "m3"
FIGURES.mkdir(parents=True, exist_ok=True)


# 2. Distribución de interacciones por usuario
def grafica_interacciones_usuario():

    # Usamos pares consolidados para evitar contar
    # varias veces el mismo destino de un usuario.
    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    usuarios = pd.read_csv(
        ROOT / "data" / "m3_recomendacion" / "raw"
        / "Final_Updated_Expanded_Users.csv"
    )

    pares = eventos[
        ["UserID", "ItemID"]
    ].drop_duplicates()

    # Incluimos también usuarios sin historial.
    actividad = (
        pares.groupby("UserID")
        .size()
        .reindex(usuarios["UserID"], fill_value=0)
    )

    distribucion = (
        actividad.value_counts()
        .sort_index()
    )

    # 3. Construcción de la gráfica
    fig, ax = plt.subplots(figsize=(9, 5))

    barras = ax.bar(
        distribucion.index,
        distribucion.values,
        color="#7CA6C4",
        edgecolor="#34546B"
    )

    for barra in barras:
        altura = barra.get_height()

        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura + 4,
            str(int(altura)),
            ha="center",
            fontsize=10
        )

    ax.set_title(
        "Distribución de destinos observados por usuario",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Número de destinos distintos por usuario"
    )

    ax.set_ylabel("Número de usuarios")

    ax.set_xticks(range(6))
    ax.set_ylim(0, max(distribucion.values) * 1.15)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # 4. Guardar figura
    ruta = FIGURES / "fig_m3_01_interacciones_usuario.png"

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nGRÁFICA GENERADA CORRECTAMENTE")
    print("Archivo:", ruta)

    print("\nDistribución de usuarios:")
    print(distribucion.to_string())

# 3. Distribución de interacciones por destino
def grafica_interacciones_destino():

    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    # Cada usuario-destino cuenta una sola vez
    pares = eventos[
        ["UserID", "ItemID"]
    ].drop_duplicates()

    # Contar usuarios que interactuaron con cada destino
    conteos = (
        pares.groupby("ItemID")
        .size()
        .reset_index(name="Interacciones")
    )

    conteos = conteos.merge(
        catalogo[["ItemID", "Name"]],
        on="ItemID",
        how="left"
    )

    conteos = conteos.sort_values(
        "Interacciones",
        ascending=False
    )

    # Porcentaje del total
    conteos["Porcentaje"] = (
        conteos["Interacciones"]
        / conteos["Interacciones"].sum() * 100
    )

    # Gráfica
    fig, ax = plt.subplots(figsize=(10, 5))

    barras = ax.bar(
        conteos["Name"],
        conteos["Interacciones"],
        color="#7CA6C4",
        edgecolor="#34546B"
    )

    for barra, porcentaje in zip(
        barras, conteos["Porcentaje"]
    ):
        altura = barra.get_height()

        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura + 5,
            f"{int(altura)} ({porcentaje:.1f}%)",
            ha="center",
            fontsize=10
        )

    ax.set_title(
        "Distribución de interacciones por destino",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Destino turístico")
    ax.set_ylabel("Pares usuario-destino observados")

    ax.set_ylim(
        0, conteos["Interacciones"].max() * 1.18
    )

    ax.tick_params(axis="x", rotation=15)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    ruta = FIGURES / "fig_m3_02_interacciones_destino.png"

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nGRÁFICA DE DESTINOS GENERADA CORRECTAMENTE")
    print("Archivo:", ruta)

    print("\nInteracciones por destino:")
    print(
        conteos[
            ["Name", "Interacciones", "Porcentaje"]
        ].to_string(index=False)
    )

    # Concentración de los dos destinos más populares
    top2 = conteos["Porcentaje"].head(2).sum()

    print(
        f"\nConcentración de los 2 destinos principales: "
        f"{top2:.2f}%"
    )


# 4. Calificación promedio por destino
def grafica_calificaciones_destino():

    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    # Una calificación promedio por usuario y destino
    pares = (
        eventos.groupby(
            ["UserID", "ItemID"],
            as_index=False
        )
        .agg(Rating=("Rating", "mean"))
    )

    # Calificación promedio de cada destino
    resumen = (
        pares.groupby("ItemID", as_index=False)
        .agg(
            CalificacionPromedio=("Rating", "mean"),
            Usuarios=("UserID", "nunique")
        )
    )

    resumen = resumen.merge(
        catalogo[["ItemID", "Name"]],
        on="ItemID",
        how="left"
    )

    resumen = resumen.sort_values(
        "CalificacionPromedio",
        ascending=False
    )

    # Crear gráfica
    fig, ax = plt.subplots(figsize=(10, 5))

    barras = ax.bar(
        resumen["Name"],
        resumen["CalificacionPromedio"],
        color="#7CA6C4",
        edgecolor="#34546B"
    )

    for barra in barras:
        valor = barra.get_height()

        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.05,
            f"{valor:.2f}",
            ha="center",
            fontsize=11
        )

    ax.set_title(
        "Calificación promedio por destino turístico",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Destino turístico")
    ax.set_ylabel("Calificación promedio (escala 1–5)")

    ax.set_ylim(0, 5.2)
    ax.tick_params(axis="x", rotation=15)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    ruta = FIGURES / "fig_m3_03_calificaciones_destino.png"

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nGRÁFICA DE CALIFICACIONES GENERADA")
    print("Archivo:", ruta)

    print("\nCalificaciones por destino:")
    print(
        resumen[
            ["Name", "CalificacionPromedio", "Usuarios"]
        ].to_string(index=False)
    )


# 5. Preferencias declaradas y calificaciones positivas
def grafica_preferencias_usuarios():

    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    usuarios = pd.read_csv(
        ROOT / "data" / "m3_recomendacion" / "raw"
        / "Final_Updated_Expanded_Users.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    # Una calificación por usuario y destino
    pares = (
        eventos.groupby(
            ["UserID", "ItemID"],
            as_index=False
        )
        .agg(Rating=("Rating", "mean"))
    )

    # Agregar preferencias y características del destino
    pares = pares.merge(
        usuarios[["UserID", "Preferences"]],
        on="UserID",
        how="left",
        validate="many_to_one"
    )

    pares = pares.merge(
        catalogo[["ItemID", "Name", "Type"]],
        on="ItemID",
        how="left",
        validate="many_to_one"
    )

    # Identificar calificaciones positivas
    pares["Positiva"] = pares["Rating"] >= 4

    # Porcentaje de calificaciones positivas
    # para cada perfil y destino
    resumen = (
        pares.groupby(["Preferences", "Name"])
        .agg(
            TasaPositiva=("Positiva", "mean"),
            Observaciones=("Positiva", "size")
        )
        .reset_index()
    )

    tabla = resumen.pivot(
        index="Preferences",
        columns="Name",
        values="TasaPositiva"
    )

    tabla = tabla.fillna(0) * 100

    # Construir mapa de calor
    fig, ax = plt.subplots(figsize=(11, 5))

    imagen = ax.imshow(
        tabla.values,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_xticks(range(len(tabla.columns)))
    ax.set_xticklabels(
        tabla.columns,
        rotation=20,
        ha="right"
    )

    ax.set_yticks(range(len(tabla.index)))
    ax.set_yticklabels(tabla.index)

    # Etiquetas dentro del mapa
    for i in range(len(tabla.index)):
        for j in range(len(tabla.columns)):
            valor = tabla.iloc[i, j]

            ax.text(
                j, i,
                f"{valor:.1f}%",
                ha="center",
                va="center",
                color="white" if valor > 50 else "black",
                fontsize=10
            )

    ax.set_title(
        "Calificaciones positivas según preferencias del usuario",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("Destino turístico")
    ax.set_ylabel("Preferencias declaradas")

    fig.colorbar(
        imagen,
        ax=ax,
        label="Calificaciones positivas (%)"
    )

    fig.tight_layout()

    ruta = FIGURES / "fig_m3_04_preferencias_usuarios.png"

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nGRÁFICA DE PREFERENCIAS GENERADA")
    print("Archivo:", ruta)

    print("\nPorcentaje de calificaciones positivas:")
    print(tabla.round(2).to_string())

    print("\nNúmero de observaciones por perfil y destino:")
    print(
        resumen.pivot(
            index="Preferences",
            columns="Name",
            values="Observaciones"
        ).fillna(0).to_string()
    )

# 6. Coincidencia entre preferencias y tipos de destino
def analizar_coincidencia_preferencias():

    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    usuarios = pd.read_csv(
        ROOT / "data" / "m3_recomendacion" / "raw"
        / "Final_Updated_Expanded_Users.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    # Consolidar las calificaciones por usuario-destino
    pares = (
        eventos.groupby(
            ["UserID", "ItemID"],
            as_index=False
        )
        .agg(Rating=("Rating", "mean"))
    )

    pares = pares.merge(
        usuarios[["UserID", "Preferences"]],
        on="UserID",
        validate="many_to_one"
    )

    pares = pares.merge(
        catalogo[["ItemID", "Type"]],
        on="ItemID",
        validate="many_to_one"
    )

    # Normalizar etiquetas equivalentes
    equivalencias = {
        "Beach": "Beaches",
        "City": "City",
        "Historical": "Historical",
        "Nature": "Nature",
        "Adventure": "Adventure"
    }

    def coincide(row):
        preferencias = [
            p.strip()
            for p in row["Preferences"].split(",")
        ]

        tipo = equivalencias.get(
            row["Type"], row["Type"]
        )

        return tipo in preferencias

    pares["Coincide"] = pares.apply(
        coincide, axis=1
    )

    pares["Positiva"] = pares["Rating"] >= 4

    resumen = (
        pares.groupby("Coincide")
        .agg(
            Pares=("Rating", "size"),
            CalificacionPromedio=("Rating", "mean"),
            TasaPositiva=("Positiva", "mean")
        )
    )

    resumen["TasaPositiva"] *= 100

    print("\n" + "=" * 60)
    print("COINCIDENCIA ENTRE PREFERENCIAS Y DESTINOS")
    print("=" * 60)

    print("\nResultados generales:")
    print(resumen.round(3).to_string())

    print("\nResultados por perfil:")
    por_perfil = (
        pares.groupby(["Preferences", "Coincide"])
        .agg(
            Pares=("Rating", "size"),
            TasaPositiva=("Positiva", "mean")
        )
    )

    print(por_perfil.round(3).to_string())

    if True in resumen.index and False in resumen.index:
        diferencia = (
            resumen.loc[True, "TasaPositiva"]
            - resumen.loc[False, "TasaPositiva"]
        )

        print(
            "\nDiferencia de tasa positiva "
            "(coincide - no coincide): "
            f"{diferencia:+.2f} puntos porcentuales"
        )


def main():
    grafica_interacciones_usuario()
    grafica_interacciones_destino()
    grafica_calificaciones_destino()
    grafica_preferencias_usuarios()                
    analizar_coincidencia_preferencias()

if __name__ == "__main__":
    main()

