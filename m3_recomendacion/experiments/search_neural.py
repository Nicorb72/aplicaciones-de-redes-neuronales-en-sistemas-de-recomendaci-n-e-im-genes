
"""
Módulo 3: Búsqueda de hiperparámetros del recomendador neuronal.

Búsqueda inicial de 8 configuraciones.
Selección con NDCG@3 en validación.
No utiliza el conjunto de prueba.
"""

from pathlib import Path
from itertools import product
from time import perf_counter
import argparse

import pandas as pd
import torch

from m3_recomendacion.experiments.train import NeuralRecommendationSystem
from m3_recomendacion.evaluation.evaluate import evaluar_modelo


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    if (
        args.k < 1
        or args.epochs < 1
        or args.patience < 1
        or args.lr <= 0
    ):
        parser.error("Parámetros de búsqueda inválidos.")

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )
    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    configuraciones = list(product(
        [4, 8],       # Embeddings
        [16, 32],     # Neuronas ocultas
        [0.1, 0.3]    # Dropout
    ))

    resultados = []
    inicio_total = perf_counter()

    print("\n" + "=" * 60)
    print("BÚSQUEDA DE HIPERPARÁMETROS NEURONALES")
    print("=" * 60)
    print("Configuraciones:", len(configuraciones))
    print("Semilla:", args.seed)
    print("Criterio de selección: NDCG@K")
    print("K:", args.k)

    for numero, (emb, hidden, dropout) in enumerate(
        configuraciones, start=1
    ):

        print("\n" + "-" * 55)
        print(
            f"Configuración {numero}/{len(configuraciones)}"
        )
        print(
            f"Embeddings={emb} | Hidden={hidden} "
            f"| Dropout={dropout}"
        )

        sistema = NeuralRecommendationSystem(
            embedding_dim=emb,
            hidden_dim=hidden,
            dropout=dropout,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
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
            "Configuracion": numero,
            "EmbeddingDim": emb,
            "HiddenDim": hidden,
            "Dropout": dropout,
            "LearningRate": args.lr,
            "Seed": args.seed,
            "BestEpoch": sistema.best_epoch,
            "Precision": metricas["Precision@K"],
            "Recall": metricas["Recall@K"],
            "NDCG": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"],
            "Parametros": parametros,
            "TiempoSegundos": tiempo
        })

        print(
            f"NDCG@{args.k}: "
            f"{metricas['NDCG@K']:.4f}"
        )
        print(
            "Mejor época:",
            sistema.best_epoch
        )
        print(f"Tiempo: {tiempo:.2f} segundos")

    tabla = pd.DataFrame(resultados)

    tabla = tabla.sort_values(
        ["NDCG", "Parametros", "Configuracion"],
        ascending=[False, True, True]
    )

    ruta = PROCESSED / "search_neural.csv"
    tabla.to_csv(ruta, index=False)

    print("\n" + "=" * 60)
    print("RESULTADOS DE LA BÚSQUEDA NEURONAL")
    print("=" * 60)

    print(
        tabla[
            [
                "Configuracion",
                "EmbeddingDim",
                "HiddenDim",
                "Dropout",
                "BestEpoch",
                "NDCG",
                "Parametros",
                "TiempoSegundos"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nMEJOR CONFIGURACIÓN NEURONAL")
    print(tabla.iloc[0].to_string())

    print(
        "\nTiempo total de búsqueda:",
        round(perf_counter() - inicio_total, 2),
        "segundos"
    )

    print("\nResultados guardados en:", ruta)


if __name__ == "__main__":
    main()
