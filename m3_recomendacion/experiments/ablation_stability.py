
"""
Módulo 3: Estabilidad del estudio de ablación.

Compara dos variantes con las mismas semillas:
1. IDs + tipo de destino.
2. Modelo completo.

Se utiliza exclusivamente entrenamiento y validación.
"""

from pathlib import Path
from time import perf_counter
import argparse

import pandas as pd

from m3_recomendacion.ablation_neural import (
    AblationNeuralSystem
)
from m3_recomendacion.evaluate import evaluar_modelo


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)

    args = parser.parse_args()

    if (
        args.seeds < 1
        or args.k < 1
        or args.epochs < 1
        or args.patience < 1
        or args.lr <= 0
        or args.embedding_dim < 1
        or args.hidden_dim < 2
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

    variantes = [
        ("IDs_Types", False, True),
        ("Completo", True, True)
    ]

    resultados = []
    inicio_total = perf_counter()

    print("\n" + "=" * 60)
    print("ESTABILIDAD DE LA ABLACIÓN NEURONAL")
    print("=" * 60)

    for seed in range(args.seeds):
        for nombre, use_pref, use_type in variantes:

            sistema = AblationNeuralSystem(
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                epochs=args.epochs,
                seed=seed,
                patience=args.patience,
                use_preferences=use_pref,
                use_types=use_type
            )

            inicio = perf_counter()

            sistema.fit(
                train, catalogo, usuarios,
                val=val, k=args.k
            )

            tiempo = perf_counter() - inicio

            metricas, _ = evaluar_modelo(
                sistema, val, catalogo, k=args.k
            )

            resultados.append({
                "Seed": seed,
                "Variante": nombre,
                "Precision": metricas["Precision@K"],
                "Recall": metricas["Recall@K"],
                "NDCG": metricas["NDCG@K"],
                "Cobertura": metricas["Cobertura"],
                "BestEpoch": sistema.best_epoch,
                "TiempoSegundos": tiempo
            })

            print(
                f"Semilla {seed + 1}/{args.seeds}"
                f" | {nombre}"
                f" | NDCG@{args.k}: "
                f"{metricas['NDCG@K']:.4f}"
            )

    df = pd.DataFrame(resultados)

    ruta = PROCESSED / "ablation_stability.csv"
    df.to_csv(ruta, index=False)

    resumen = (
        df.groupby("Variante")
        .agg(
            NDCG_Media=("NDCG", "mean"),
            NDCG_DesvStd=("NDCG", "std"),
            NDCG_Min=("NDCG", "min"),
            NDCG_Max=("NDCG", "max"),
            Precision_Media=("Precision", "mean"),
            Recall_Media=("Recall", "mean")
        )
    )

    # Diferencia emparejada: misma semilla
    tabla_pares = df.pivot(
        index="Seed",
        columns="Variante",
        values="NDCG"
    )

    diferencias = (
        tabla_pares["Completo"]
        - tabla_pares["IDs_Types"]
    )

    print("\n" + "=" * 60)
    print("RESUMEN DE ESTABILIDAD DE ABLACIÓN")
    print("=" * 60)

    print(resumen.round(4).to_string())

    print("\nDiferencia Completo - IDs_Types:")
    print("Media:", round(diferencias.mean(), 4))
    print("Desviación:", round(diferencias.std(), 4))
    print(
        "Semillas en que mejora el completo:",
        int((diferencias > 0).sum()),
        "de",
        len(diferencias)
    )

    print(
        "\nTiempo total:",
        round(perf_counter() - inicio_total, 2),
        "segundos"
    )

    print("Resultados guardados en:", ruta)


if __name__ == "__main__":
    main()
