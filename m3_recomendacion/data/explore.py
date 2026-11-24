
"""
Módulo 3: Sistema de recomendación de destinos turísticos.

Objetivo:
Cargar y explorar los datos originales del Travel Recommendation Dataset.

En esta etapa se verifican:
- Dimensiones de cada archivo.
- Variables y tipos de datos.
- Valores faltantes.
- Registros duplicados.
- Cantidad de usuarios y destinos únicos.

Los datos originales no se modifican.
"""

from pathlib import Path
import pandas as pd


# 1. Ubicación de los archivos
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "m3_recomendacion" / "raw"


# 2. Nombres de los archivos
FILES = {
    "usuarios": "Final_Updated_Expanded_Users.csv",
    "destinos": "Expanded_Destinations.csv",
    "historial": "Final_Updated_Expanded_UserHistory.csv",
    "resenas": "Final_Updated_Expanded_Reviews.csv",
}


# 3. Cargar los datos
def cargar_datos():
    datos = {}

    for nombre, archivo in FILES.items():
        ruta = DATA_DIR / archivo

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {ruta}"
            )

        datos[nombre] = pd.read_csv(ruta)

    return datos


# 4. Explorar cada archivo
def explorar_datos(datos):
    for nombre, df in datos.items():
        print("\n" + "=" * 60)
        print(f"ARCHIVO: {nombre.upper()}")
        print("=" * 60)

        print(f"Filas: {df.shape[0]}")
        print(f"Columnas: {df.shape[1]}")

        print("\nVariables y tipos de datos:")
        print(df.dtypes.to_string())

        print("\nValores faltantes:")
        print(df.isnull().sum().to_string())

        print("\nFilas completamente duplicadas:")
        print(df.duplicated().sum())

        print("\nPrimeros 5 registros:")
        print(df.head().to_string(index=False))


# 5. Conteo de usuarios y destinos
def resumen_general(datos):
    usuarios = datos["usuarios"]
    destinos = datos["destinos"]
    historial = datos["historial"]
    resenas = datos["resenas"]

    print("\n" + "=" * 60)
    print("RESUMEN GENERAL DEL DATASET")
    print("=" * 60)

    print("Usuarios únicos:",
          usuarios["UserID"].nunique())

    print("Destinos únicos por ID:",
          destinos["DestinationID"].nunique())

    print("Nombres de destinos distintos:",
          destinos["Name"].nunique())

    print("Registros del historial:",
          len(historial))

    print("Registros de reseñas:",
          len(resenas))

    print("Total de registros de interacción:",
          len(historial) + len(resenas))
          

# 7. Analizar las características de los destinos
def analizar_destinos(datos):
    destinos = datos["destinos"]

    print("\n" + "=" * 60)
    print("ANÁLISIS DE LOS DESTINOS")
    print("=" * 60)

    # Cantidad de registros por destino
    print("\nCantidad de registros por nombre:")
    print(destinos["Name"].value_counts().to_string())

    # Características de cada destino
    print("\nCaracterísticas de los destinos:")
    caracteristicas = (
        destinos.groupby("Name")
        .agg(
            cantidad_ids=("DestinationID", "nunique"),
            estados_distintos=("State", "nunique"),
            tipos_distintos=("Type", "nunique"),
            temporadas_distintas=("BestTimeToVisit", "nunique"),
            popularidad_minima=("Popularity", "min"),
            popularidad_maxima=("Popularity", "max"),
            popularidad_promedio=("Popularity", "mean")
        )
        .round(3)
    )

    print(caracteristicas.to_string())

    # Distribución de las calificaciones
    print("\nCalificaciones del historial:")
    print(
        datos["historial"]["ExperienceRating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCalificaciones de las reseñas:")
    print(
        datos["resenas"]["Rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Preferencias de usuarios
    print("\nPreferencias declaradas por los usuarios:")
    print(
        datos["usuarios"]["Preferences"]
        .value_counts()
        .to_string()
    )

# 8. Analizar relaciones entre usuarios, destinos e interacciones
def analizar_interacciones(datos):
    destinos = datos["destinos"]
    historial = datos["historial"]
    resenas = datos["resenas"]

    print("\n" + "=" * 60)
    print("ANÁLISIS DE INTERACCIONES USUARIO-DESTINO")
    print("=" * 60)

    # Asociar cada interacción con el nombre del destino
    catalogo = destinos[["DestinationID", "Name"]]

    h = historial.merge(
        catalogo,
        on="DestinationID",
        how="left",
        validate="many_to_one"
    )

    r = resenas.merge(
        catalogo,
        on="DestinationID",
        how="left",
        validate="many_to_one"
    )

    # Unificar la estructura para analizar ambas fuentes
    h = h[["UserID", "DestinationID", "Name",
           "ExperienceRating"]].rename(
        columns={"ExperienceRating": "Rating"}
    )
    h["Fuente"] = "Historial"

    r = r[["UserID", "DestinationID", "Name",
           "Rating"]].copy()
    r["Fuente"] = "Resena"

    interacciones = pd.concat(
        [h, r], ignore_index=True
    )

    print("\nTotal de registros:",
          len(interacciones))

    print("Usuarios con interacciones:",
          interacciones["UserID"].nunique())

    print("Destinos originales utilizados:",
          interacciones["DestinationID"].nunique())

    print("\nPares usuario-ID de destino distintos:",
          interacciones[
              ["UserID", "DestinationID"]
          ].drop_duplicates().shape[0])

    print("\nPares usuario-nombre de destino distintos:",
          interacciones[
              ["UserID", "Name"]
          ].drop_duplicates().shape[0])

    # Repeticiones por destino consolidado
    pares = (
        interacciones
        .groupby(["UserID", "Name"])
        .agg(
            registros=("Rating", "size"),
            calificaciones_distintas=("Rating", "nunique")
        )
        .reset_index()
    )

    print("\nPares con múltiples interacciones:",
          (pares["registros"] > 1).sum())

    print("Pares con calificaciones diferentes:",
          (pares["calificaciones_distintas"] > 1).sum())

    # Actividad por usuario
    actividad = (
        interacciones
        .groupby("UserID")
        .agg(
            total_interacciones=("Name", "size"),
            destinos_diferentes=("Name", "nunique")
        )
    )

    print("\nDistribución de destinos diferentes por usuario:")
    print(
        actividad["destinos_diferentes"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Distribución de interacciones por destino
    print("\nInteracciones por destino:")
    print(
        interacciones["Name"]
        .value_counts()
        .to_string()
    )


# 6. Ejecución
def main():
    datos = cargar_datos()
    explorar_datos(datos)
    resumen_general(datos)
    analizar_destinos(datos)
    analizar_interacciones(datos)


if __name__ == "__main__":
    main()
