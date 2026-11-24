
"""
Módulo 3: Evaluación de recomendadores por historial.
Responsable: Persona 4.

Objetivo:
Comparar el desempeño de tres recomendadores según
la cantidad de interacciones observadas del usuario.

Grupos:
- Pocas_1: una interacción observada en train.
- Mas_2_o_mas: dos o más interacciones en train.

Modelos:
1. Popularidad.
2. Filtrado colaborativo.
3. Red neuronal C5.

Métricas:
- Precision@K.
- Recall@K.
- NDCG@K.
- Cobertura del catálogo.

Valores de K: 1, 2 y 3.

Se utiliza exclusivamente entrenamiento y validación.
No se utilizan test.csv ni cold_test.csv.
"""

from pathlib import Path

import pandas as pd

from m3_recomendacion.popularity import PopularityRecommender
from m3_recomendacion.collaborative import CollaborativeRecommender
from m3_recomendacion.train import NeuralRecommendationSystem
from m3_recomendacion.evaluate import evaluar_modelo


# ============================================================
# 1. RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = (
    ROOT / "data" / "m3_recomendacion" / "processed"
)

RAW = (
    ROOT / "data" / "m3_recomendacion" / "raw"
)


# ============================================================
# 2. CARGAR DATOS
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
# 3. CLASIFICAR USUARIOS SEGÚN SU HISTORIAL
# ============================================================

def clasificar_usuarios(train, val):

    # Cantidad de destinos distintos observados
    # por usuario en el conjunto de entrenamiento.
    conteos = (
        train.groupby("UserID")["ItemID"]
        .nunique()
    )

    val_grupos = val.copy()

    val_grupos["InteraccionesTrain"] = (
        val_grupos["UserID"]
        .map(conteos)
        .fillna(0)
        .astype(int)
    )

    def asignar_grupo(n):

        if n == 1:
            return "Pocas_1"

        if n >= 2:
            return "Mas_2_o_mas"

        return "Sin_historial"

    val_grupos["Grupo"] = (
        val_grupos["InteraccionesTrain"]
        .apply(asignar_grupo)
    )

    return val_grupos


# ============================================================
# 4. MOSTRAR DISTRIBUCIÓN DE USUARIOS
# ============================================================

def mostrar_distribucion(val_grupos):

    distribucion = (
        val_grupos.groupby("Grupo")
        .agg(
            Usuarios=("UserID", "nunique"),
            InteraccionesVal=("ItemID", "size"),
            HistorialMedio=("InteraccionesTrain", "mean")
        )
    )

    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN DE USUARIOS POR HISTORIAL")
    print("=" * 60)

    print(
        distribucion.round(4).to_string()
    )

    return distribucion


# ============================================================
# 5. ENTRENAR LOS TRES MODELOS
# ============================================================

def entrenar_modelos(train, val, catalogo, usuarios):

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DE LOS MODELOS")
    print("=" * 60)

    # --------------------------------------------------------
    # Modelo 1: Popularidad
    # --------------------------------------------------------

    print("\n[1/3] Popularidad")

    popularidad = PopularityRecommender()

    popularidad.fit(
        train,
        catalogo
    )

    # --------------------------------------------------------
    # Modelo 2: Filtrado colaborativo
    # --------------------------------------------------------

    print("\n[2/3] Filtrado colaborativo")

    # Se utiliza la configuración predeterminada
    # previamente evaluada:
    # factores=8, lr=0.01,
    # regularización=0.001,
    # epochs=100, seed=42.
    #
    # Estos valores deben mantenerse en collaborative.py.

    colaborativo = CollaborativeRecommender()

    colaborativo.fit(
        train,
        catalogo
    )

    # --------------------------------------------------------
    # Modelo 3: Red neuronal C5
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

    # El modelo neuronal selecciona los mejores
    # pesos utilizando NDCG@3 en validación.
    neuronal.fit(
        train,
        catalogo,
        usuarios,
        val=val,
        k=3
    )

    modelos = {
        "Popularidad": popularidad,
        "Colaborativo": colaborativo,
        "Neuronal_C5": neuronal
    }

    return modelos


# ============================================================
# 6. EVALUAR POR GRUPO Y POR K
# ============================================================

def evaluar_por_grupo(modelos, val_grupos, catalogo):

    resultados = []

    valores_k = [1, 2, 3]

    for grupo, datos_grupo in val_grupos.groupby("Grupo"):

        if grupo == "Sin_historial":

            print(
                "\nSe omite el grupo Sin_historial "
                "de la evaluación warm-start."
            )

            continue

        print("\n" + "-" * 60)
        print("Evaluando grupo:", grupo)
        print(
            "Usuarios:",
            datos_grupo["UserID"].nunique()
        )
        print("-" * 60)

        for nombre, modelo in modelos.items():

            for k in valores_k:

                metricas, _ = evaluar_modelo(
                    modelo,
                    datos_grupo,
                    catalogo,
                    k=k
                )

                resultados.append({
                    "Grupo": grupo,
                    "Modelo": nombre,
                    "K": k,
                    "Usuarios": metricas["Usuarios_evaluados"],
                    "Precision": metricas["Precision@K"],
                    "Recall": metricas["Recall@K"],
                    "NDCG": metricas["NDCG@K"],
                    "Cobertura": metricas["Cobertura"]
                })

    return pd.DataFrame(resultados)


# ============================================================
# 7. MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(tabla):

    print("\n" + "=" * 60)
    print("RESULTADOS POR GRUPO DE USUARIOS")
    print("=" * 60)

    tabla_ordenada = tabla.sort_values(
        ["Grupo", "K", "Modelo"]
    )

    print(
        tabla_ordenada.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Comparación principal para K = 1
    # --------------------------------------------------------

    k1 = tabla.loc[
        tabla["K"] == 1
    ].copy()

    print("\n" + "=" * 60)
    print("COMPARACIÓN PRINCIPAL PARA K = 1")
    print("=" * 60)

    print(
        k1.sort_values(
            ["Grupo", "NDCG"],
            ascending=[True, False]
        ).round(4).to_string(index=False)
    )

    print(
        "\nNota: NDCG@1 equivale a Recall@1 "
        "cuando cada usuario tiene un único "
        "destino positivo retenido."
    )

    # --------------------------------------------------------
    # Comparación para K = 3
    # --------------------------------------------------------

    k3 = tabla.loc[
        tabla["K"] == 3
    ].copy()

    print("\n" + "=" * 60)
    print("COMPARACIÓN PARA K = 3")
    print("=" * 60)

    print(
        k3.sort_values(
            ["Grupo", "NDCG"],
            ascending=[True, False]
        ).round(4).to_string(index=False)
    )


# ============================================================
# 8. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # Cargar datos
    # --------------------------------------------------------

    train, val, catalogo, usuarios = cargar_datos()

    # --------------------------------------------------------
    # Clasificar usuarios usando solo train
    # --------------------------------------------------------

    val_grupos = clasificar_usuarios(
        train,
        val
    )

    # --------------------------------------------------------
    # Mostrar distribución
    # --------------------------------------------------------

    mostrar_distribucion(
        val_grupos
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
    # Evaluar modelos
    # --------------------------------------------------------

    tabla = evaluar_por_grupo(
        modelos,
        val_grupos,
        catalogo
    )

    if tabla.empty:
        raise ValueError(
            "No se obtuvieron resultados de evaluación."
        )

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta = (
        PROCESSED / "metrics_by_user_group.csv"
    )

    tabla.to_csv(
        ruta,
        index=False
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    mostrar_resultados(
        tabla
    )

    print("\n" + "=" * 60)
    print("ARCHIVO GENERADO")
    print("=" * 60)

    print(
        "CSV guardado en:",
        ruta
    )


# ============================================================
# 9. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
