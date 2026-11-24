
"""
Módulo 3: Estabilidad del recomendador neuronal.

Evalúa 30 inicializaciones independientes.
Usa entrenamiento y validación.
No utiliza el conjunto de prueba.
"""

from pathlib import Path
from time import perf_counter
import argparse

import pandas as pd

from m3_recomendacion.experiments.train import NeuralRecommendationSystem
from m3_recomendacion.evaluation.evaluate import evaluar_modelo


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = (
    ROOT / "data" / "m3_recomendacion" / "processed"
)

RAW = ROOT / "data" / "m3_recomendacion" / "raw"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--k", type=int, default=3)

    args = parser.parse_args()

    if (
        args.seeds < 1
        or args.embedding_dim < 1
        or args.hidden_dim < 2
        or args.epochs < 1
        or args.patience < 1
        or args.k < 1
        or args.lr <= 0
        or not 0 <= args.dropout < 1
    ):
        parser.error("Parámetros inválidos.")

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    resultados = []
    inicio_total = perf_counter()

    print("\n" + "=" * 60)
    print("ESTABILIDAD DEL RECOMENDADOR NEURONAL")
    print("=" * 60)

    print("Número de semillas:", args.seeds)
    print("Embeddings:", args.embedding_dim)
    print("Hidden:", args.hidden_dim)
    print("Dropout:", args.dropout)
    print("Learning rate:", args.lr)
    print("K:", args.k)

    for seed in range(args.seeds):

        sistema = NeuralRecommendationSystem(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            seed=seed,
            patience=args.patience
        )

        inicio = perf_counter()

        sistema.fit(
            train,
            catalogo,
            usuarios,
            val=val,
            k=args.k
        )

        tiempo = perf_counter() - inicio

        metricas, _ = evaluar_modelo(
            sistema,
            val,
            catalogo,
            k=args.k
        )

        parametros = sum(
            p.numel()
            for p in sistema.model.parameters()
            if p.requires_grad
        )

        resultados.append({
            "Seed": seed,
            "BestEpoch": sistema.best_epoch,
            "Precision": metricas["Precision@K"],
            "Recall": metricas["Recall@K"],
            "NDCG": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"],
            "Parametros": parametros,
            "TiempoSegundos": tiempo
        })

        print(
            f"\nSEMILLA {seed + 1}/{args.seeds}"
            f" | Mejor época: {sistema.best_epoch}"
            f" | NDCG@{args.k}: "
            f"{metricas['NDCG@K']:.4f}"
        )

    df = pd.DataFrame(resultados)

    ruta = PROCESSED / "stability_neural.csv"
    df.to_csv(ruta, index=False)

    resumen = df[
        ["Precision", "Recall", "NDCG", "Cobertura"]
    ].agg(["mean", "std", "min", "max"])

    print("\n" + "=" * 60)
    print("RESUMEN DE ESTABILIDAD NEURONAL")
    print("=" * 60)

    print(resumen.round(4).to_string())

    print("\nMejor NDCG:")
    print(
        df.loc[df["NDCG"].idxmax(),
               ["Seed", "BestEpoch", "NDCG"]].to_string()
    )

    print("\nPeor NDCG:")
    print(
        df.loc[df["NDCG"].idxmin(),
               ["Seed", "BestEpoch", "NDCG"]].to_string()
    )

    print(
        "\nTiempo total:",
        round(perf_counter() - inicio_total, 2),
        "segundos"
    )

    print("Resultados guardados en:", ruta)


if __name__ == "__main__":
    main()
