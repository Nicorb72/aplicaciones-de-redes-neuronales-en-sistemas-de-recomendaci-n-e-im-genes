
"""
Módulo 3: Preparación inicial del sistema de recomendación.
Responsable: Persona 4.

Genera:
1. Catálogo consolidado de destinos.
2. Tabla unificada de eventos de interacción.

No realiza particiones ni combina calificaciones repetidas.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "m3_recomendacion" / "raw"
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"


def preparar_datos():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Leer los archivos originales
    destinos = pd.read_csv(RAW / "Expanded_Destinations.csv")
    historial = pd.read_csv(
        RAW / "Final_Updated_Expanded_UserHistory.csv"
    )
    resenas = pd.read_csv(
        RAW / "Final_Updated_Expanded_Reviews.csv"
    )

    # 2. Consolidar catálogo turístico
    catalogo = (
        destinos.groupby(
            ["Name", "State", "Type", "BestTimeToVisit"],
            as_index=False
        )
        .agg(
            Popularity=("Popularity", "mean"),
            OriginalIDs=("DestinationID", "nunique")
        )
        .sort_values("Name")
        .reset_index(drop=True)
    )

    catalogo.insert(
        0, "ItemID", range(1, len(catalogo) + 1)
    )

    # 3. Crear equivalencia de identificadores originales
    equivalencias = destinos.merge(
        catalogo[
            ["Name", "State", "Type",
             "BestTimeToVisit", "ItemID"]
        ],
        on=["Name", "State", "Type", "BestTimeToVisit"],
        how="left",
        validate="many_to_one"
    )[["DestinationID", "ItemID"]]

    # 4. Estandarizar el historial
    historial = historial.rename(
        columns={"ExperienceRating": "Rating"}
    )
    historial["Fuente"] = "historial"

    # 5. Estandarizar las reseñas
    resenas["Fuente"] = "resena"
    resenas["VisitDate"] = pd.NA

    # 6. Unir eventos sin perder información
    columnas = [
        "UserID", "DestinationID",
        "Rating", "VisitDate", "Fuente"
    ]

    eventos = pd.concat(
        [historial[columnas], resenas[columnas]],
        ignore_index=True
    )

    eventos = eventos.merge(
        equivalencias,
        on="DestinationID",
        how="left",
        validate="many_to_one"
    )

    # 7. Validaciones
    assert eventos["ItemID"].notna().all()
    assert eventos["Rating"].between(1, 5).all()
    assert len(eventos) == len(historial) + len(resenas)
    assert equivalencias["DestinationID"].is_unique

    # 8. Guardar archivos procesados
    catalogo.to_csv(
        PROCESSED / "catalogo_destinos.csv",
        index=False
    )

    equivalencias.to_csv(
        PROCESSED / "equivalencias_destinos.csv",
        index=False
    )

    eventos.to_csv(
        PROCESSED / "eventos_interaccion.csv",
        index=False
    )

    print("\nPREPARACIÓN INICIAL COMPLETADA")
    print("-" * 45)
    print("Destinos consolidados:", len(catalogo))
    print("Equivalencias de ID:", len(equivalencias))
    print("Eventos conservados:", len(eventos))
    print("Archivos guardados en:", PROCESSED)


def validar_datos_procesados():
    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    equivalencias = pd.read_csv(
        PROCESSED / "equivalencias_destinos.csv"
    )

    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    print("\n" + "=" * 55)
    print("VALIDACIÓN DE LOS DATOS PROCESADOS")
    print("=" * 55)

    print("\nDestinos consolidados:")
    print(
        catalogo[
            ["ItemID", "Name", "Type", "Popularity"]
        ].to_string(index=False)
    )

    print("\nEventos por fuente:")
    print(eventos["Fuente"].value_counts().to_string())

    print("\nValores faltantes en columnas esenciales:")
    print(
        eventos[
            ["UserID", "DestinationID", "ItemID", "Rating"]
        ].isnull().sum().to_string()
    )

    print("\nPares usuario-destino consolidados:")
    print(
        eventos[["UserID", "ItemID"]]
        .drop_duplicates()
        .shape[0]
    )

    print("\nUsuarios con interacciones:")
    print(eventos["UserID"].nunique())

    print("\nIdentificadores originales asociados:")
    print(equivalencias["DestinationID"].nunique())

    print("\nDistribución de calificaciones:")
    print(
        eventos["Rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    assert len(catalogo) == 5
    assert len(equivalencias) == 1000
    assert len(eventos) == 1998
    assert eventos["ItemID"].notna().all()

    print("\nVALIDACIONES COMPLETADAS CORRECTAMENTE")


if __name__ == "__main__":
    preparar_datos()
    validar_datos_procesados()
