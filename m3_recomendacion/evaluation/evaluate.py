
"""
Módulo 3: Evaluación de recomendadores.

Métricas:
- Precision@K
- Recall@K
- NDCG@K
- Cobertura del catálogo

Los modelos se evalúan con las mismas particiones.
"""

from pathlib import Path
import argparse
import math

import pandas as pd

from m3_recomendacion.models.popularity import PopularityRecommender


# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = (
    ROOT / "data" / "m3_recomendacion" / "processed"
)


# ============================================================
# 2. MÉTRICAS DE EVALUACIÓN
# ============================================================

def precision_at_k(recomendados, relevantes, k):

    if k <= 0:
        raise ValueError("K debe ser positivo.")

    recomendados = recomendados[:k]

    aciertos = len(
        set(recomendados) & set(relevantes)
    )

    return aciertos / k


def recall_at_k(recomendados, relevantes, k):

    relevantes = set(relevantes)

    if not relevantes:
        return 0.0

    recomendados = recomendados[:k]

    aciertos = len(
        set(recomendados) & relevantes
    )

    return aciertos / len(relevantes)


def ndcg_at_k(recomendados, relevantes, k):

    relevantes = set(relevantes)

    if not relevantes:
        return 0.0

    recomendados = recomendados[:k]

    # Discounted Cumulative Gain
    dcg = sum(
        1 / math.log2(posicion + 2)
        for posicion, item in enumerate(recomendados)
        if item in relevantes
    )

    # Ideal Discounted Cumulative Gain
    n_ideales = min(
        len(relevantes), k
    )

    idcg = sum(
        1 / math.log2(posicion + 2)
        for posicion in range(n_ideales)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# 3. EVALUACIÓN GENERAL DEL MODELO
# ============================================================

def evaluar_modelo(modelo, evaluacion, catalogo, k=3):

    resultados = []
    destinos_recomendados = set()

    for user_id, grupo in evaluacion.groupby("UserID"):

        relevantes = set(
            grupo.loc[
                grupo["Positiva"], "ItemID"
            ].astype(int)
        )

        # Solo usuarios con interacciones positivas
        if not relevantes:
            continue

        # Generar recomendaciones
        recomendaciones = modelo.recommend(
            user_id=int(user_id),
            k=k
        )

        recomendados = [
            int(item_id)
            for item_id, _ in recomendaciones
        ]

        destinos_recomendados.update(
            recomendados
        )

        # Calcular métricas individuales
        precision = precision_at_k(
            recomendados, relevantes, k
        )

        recall = recall_at_k(
            recomendados, relevantes, k
        )

        ndcg = ndcg_at_k(
            recomendados, relevantes, k
        )

        resultados.append({
            "UserID": int(user_id),
            "Precision": precision,
            "Recall": recall,
            "NDCG": ndcg
        })

    if not resultados:
        raise ValueError(
            "No hay usuarios evaluables "
            "con interacciones positivas."
        )

    df = pd.DataFrame(resultados)

    # Promedio de métricas entre usuarios
    metricas = {
        "Precision@K": df["Precision"].mean(),
        "Recall@K": df["Recall"].mean(),
        "NDCG@K": df["NDCG"].mean(),
        "Cobertura": (
            len(destinos_recomendados)
            / len(catalogo)
        ),
        "Usuarios_evaluados": len(df)
    }

    return metricas, df


# ============================================================
# 4. EVALUACIÓN PARA DIFERENTES VALORES DE K
# ============================================================

def evaluar_varios_k(modelo, evaluacion, catalogo):

    resultados = []

    # Evaluar desde K=1 hasta el tamaño del catálogo
    for k in range(1, len(catalogo) + 1):

        metricas, _ = evaluar_modelo(
            modelo,
            evaluacion,
            catalogo,
            k=k
        )

        resultados.append({
            "K": k,
            "Precision": metricas["Precision@K"],
            "Recall": metricas["Recall@K"],
            "NDCG": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"]
        })

    tabla = pd.DataFrame(resultados)

    print("\n" + "=" * 55)
    print("EVALUACIÓN PARA DIFERENTES VALORES DE K")
    print("=" * 55)

    print(
        tabla.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return tabla


# ============================================================
# 5. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluación del modelo de popularidad"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Número de destinos recomendados"
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Conjunto utilizado para evaluación"
    )

    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k debe ser positivo.")

    # Cargar los datos procesados
    train = pd.read_csv(
        PROCESSED / "train.csv"
    )

    evaluacion = pd.read_csv(
        PROCESSED / f"{args.split}.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    # Ajustar modelo usando solo entrenamiento
    modelo = PopularityRecommender()

    modelo.fit(
        train,
        catalogo
    )

    # Evaluación para el K seleccionado
    metricas, detalle = evaluar_modelo(
        modelo,
        evaluacion,
        catalogo,
        k=args.k
    )

    print("\n" + "=" * 55)
    print("EVALUACIÓN DEL MODELO DE POPULARIDAD")
    print("=" * 55)

    print("Conjunto:", args.split)
    print("K:", args.k)

    print("\nMétricas:")

    for nombre, valor in metricas.items():

        if nombre == "Usuarios_evaluados":
            print(f"{nombre}: {valor}")

        else:
            print(f"{nombre}: {valor:.4f}")

    # Resultados individuales
    print("\nPrimeros cinco usuarios evaluados:")

    print(
        detalle.head().to_string(index=False)
    )

    # Evaluación de K=1 hasta K=5
    tabla_k = evaluar_varios_k(
        modelo,
        evaluacion,
        catalogo
    )


# ============================================================
# 6. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
