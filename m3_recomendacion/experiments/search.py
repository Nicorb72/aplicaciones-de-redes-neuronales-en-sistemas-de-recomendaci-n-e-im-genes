
"""
Módulo 3: Búsqueda de hiperparámetros.

Optimización del filtrado colaborativo con PyTorch.
Selección por NDCG@3 en validación.
"""

from pathlib import Path
from itertools import product
from time import perf_counter
import argparse

import pandas as pd

from m3_recomendacion.models.collaborative import (
    CollaborativeRecommender
)
from m3_recomendacion.evaluation.evaluate import evaluar_modelo


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.k < 1 or args.epochs < 1:
        parser.error("K y epochs deben ser positivos.")

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    # Configuraciones a evaluar
    factores = [4, 8]
    learning_rates = [0.005, 0.01]
    regularizaciones = [0.0001, 0.001]

    configuraciones = list(product(
        factores,
        learning_rates,
        regularizaciones
    ))

    resultados = []
    inicio_total = perf_counter()

    print("\n" + "=" * 60)
    print("BÚSQUEDA DE HIPERPARÁMETROS - COLABORATIVO")
    print("=" * 60)
    print("Configuraciones:", len(configuraciones))
    print("Conjunto de selección: val")
    print("Métrica principal: NDCG@K")
    print("K:", args.k)

    for numero, (factors, lr, wd) in enumerate(
        configuraciones, start=1
    ):

        print("\n" + "-" * 50)
        print(
            f"Configuración {numero}/{len(configuraciones)}"
        )
        print(
            f"Factores={factors} | LR={lr} | WD={wd}"
        )

        modelo = CollaborativeRecommender(
            factors=factors,
            lr=lr,
            epochs=args.epochs,
            weight_decay=wd,
            seed=args.seed
        )

        inicio = perf_counter()
        modelo.fit(train, catalogo)
        tiempo = perf_counter() - inicio

        metricas, _ = evaluar_modelo(
            modelo,
            val,
            catalogo,
            k=args.k
        )

        parametros = sum(
            p.numel()
            for p in modelo.model.parameters()
            if p.requires_grad
        )

        resultados.append({
            "Configuracion": numero,
            "Factors": factors,
            "LearningRate": lr,
            "WeightDecay": wd,
            "Epochs": args.epochs,
            "Seed": args.seed,
            "Precision": metricas["Precision@K"],
            "Recall": metricas["Recall@K"],
            "NDCG": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"],
            "LossFinal": modelo.losses[-1],
            "Parametros": parametros,
            "TiempoSegundos": tiempo
        })

        print(
            f"NDCG@{args.k}: "
            f"{metricas['NDCG@K']:.4f}"
        )
        print(f"Tiempo: {tiempo:.2f} segundos")

    tabla = pd.DataFrame(resultados)

    # Mejor NDCG; desempate por menor complejidad
    tabla = tabla.sort_values(
        ["NDCG", "Parametros", "Configuracion"],
        ascending=[False, True, True]
    )

    ruta = PROCESSED / "search_collaborative.csv"
    tabla.to_csv(ruta, index=False)

    print("\n" + "=" * 60)
    print("RESULTADOS DE LA BÚSQUEDA")
    print("=" * 60)

    print(
        tabla[
            ["Configuracion", "Factors", "LearningRate",
             "WeightDecay", "Precision", "Recall",
             "NDCG", "TiempoSegundos"]
        ].round(4).to_string(index=False)
    )

    mejor = tabla.iloc[0]

    print("\nMEJOR CONFIGURACIÓN")
    print(mejor.to_string())

    print(
        "\nTiempo total de búsqueda:",
        round(perf_counter() - inicio_total, 2),
        "segundos"
    )

    print("\nResultados guardados en:", ruta)


if __name__ == "__main__":
    main()
