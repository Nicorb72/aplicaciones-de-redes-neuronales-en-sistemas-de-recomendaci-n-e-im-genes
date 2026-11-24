
"""
Módulo 3: Estabilidad del filtrado colaborativo.

Evalúa la configuración seleccionada con 30 semillas.
No utiliza el conjunto de prueba.
"""

from pathlib import Path
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

    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--weight-decay", type=float, default=0.001
    )
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    if args.seeds < 1 or args.k < 1:
        parser.error("Seeds y K deben ser positivos.")

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    resultados = []
    inicio_total = perf_counter()

    print("\n" + "=" * 60)
    print("ESTABILIDAD DEL FILTRADO COLABORATIVO")
    print("=" * 60)

    for seed in range(args.seeds):

        modelo = CollaborativeRecommender(
            factors=args.factors,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            seed=seed
        )

        inicio = perf_counter()
        modelo.fit(train, catalogo)

        metricas, _ = evaluar_modelo(
            modelo, val, catalogo, k=args.k
        )

        tiempo = perf_counter() - inicio

        resultados.append({
            "Seed": seed,
            "Precision": metricas["Precision@K"],
            "Recall": metricas["Recall@K"],
            "NDCG": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"],
            "LossFinal": modelo.losses[-1],
            "TiempoSegundos": tiempo
        })

        print(
            f"Semilla {seed + 1}/{args.seeds} "
            f"| NDCG@{args.k}: "
            f"{metricas['NDCG@K']:.4f}"
        )

    df = pd.DataFrame(resultados)

    resumen = df[
        ["Precision", "Recall", "NDCG", "Cobertura"]
    ].agg(["mean", "std", "min", "max"])

    print("\n" + "=" * 60)
    print("RESUMEN DE ESTABILIDAD")
    print("=" * 60)
    print(resumen.round(4).to_string())

    print(
        "\nTiempo total:",
        round(perf_counter() - inicio_total, 2),
        "segundos"
    )

    ruta = PROCESSED / "stability_collaborative.csv"
    df.to_csv(ruta, index=False)

    print("Resultados guardados en:", ruta)


if __name__ == "__main__":
    main()
