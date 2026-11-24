
"""
Módulo 3: Referencia aleatoria.

Objetivo:
Evaluar cuánto desempeño puede alcanzarse
recomendando destinos al azar.

Se utiliza el mismo conjunto de entrenamiento,
validación y métricas que en los demás modelos.
"""

from pathlib import Path
import argparse

import pandas as pd
import torch

from m3_recomendacion.evaluate import (
    evaluar_modelo,
    evaluar_varios_k
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"


class RandomRecommender:

    def __init__(self, seed=42):
        self.seed = seed
        self.item_ids = []
        self.historial = {}

    def fit(self, train, catalogo):

        self.item_ids = sorted(
            catalogo["ItemID"].astype(int).tolist()
        )

        self.historial = (
            train.groupby("UserID")["ItemID"]
            .apply(lambda x: set(x.astype(int)))
            .to_dict()
        )

        return self

    def recommend(self, user_id, k=3):

        user_id = int(user_id)

        vistos = self.historial.get(user_id, set())

        candidatos = [
            item_id
            for item_id in self.item_ids
            if item_id not in vistos
        ]

        if not candidatos:
            return []

        # Generador reproducible para cada usuario.
        # El ranking se conserva al cambiar K.
        generador = torch.Generator()
        generador.manual_seed(self.seed + user_id)

        indices = torch.randperm(
            len(candidatos),
            generator=generador
        ).tolist()

        ranking = [
            candidatos[i] for i in indices
        ]

        # La puntuación indica solamente el orden
        # aleatorio, no una probabilidad estimada.
        return [
            (item_id, float(len(ranking) - posicion))
            for posicion, item_id in enumerate(ranking[:k])
        ]


def evaluar_semillas_aleatorias(train, val, catalogo, k=3,
                               n_semillas=100):

    resultados = []

    for seed in range(n_semillas):
        modelo = RandomRecommender(seed=seed)
        modelo.fit(train, catalogo)

        metricas, _ = evaluar_modelo(
            modelo, val, catalogo, k=k
        )

        resultados.append({
            "Semilla": seed,
            "Precision": metricas["Precision@K"],
            "Recall": metricas["Recall@K"],
            "NDCG": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"]
        })

    df = pd.DataFrame(resultados)

    print("\n" + "=" * 55)
    print("REFERENCIA ALEATORIA: 100 SEMILLAS")
    print("=" * 55)

    resumen = df[
        ["Precision", "Recall", "NDCG", "Cobertura"]
    ].agg(["mean", "std", "min", "max"])

    print(resumen.round(4).to_string())

    # Comparación con los valores obtenidos
    # previamente en validación para K=3.
    if k == 3:
        print("\nComparación de NDCG@3:")
        print(
            "Promedio aleatorio:",
            round(df["NDCG"].mean(), 4)
        )
        print("Popularidad: 0.6438")
        print("Colaborativo inicial: 0.7474")

        proporcion = (df["NDCG"] >= 0.7474).mean()

        print(
            "Proporción de semillas aleatorias con "
            "NDCG >= colaborativo:",
            f"{proporcion:.2%}"
        )

    # Guardar resultados reproducibles
    ruta = PROCESSED / "random_baseline_100_seeds.csv"
    df.to_csv(ruta, index=False)

    print("\nResultados guardados en:", ruta)

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k", type=int, default=3
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    if args.k < 1:
        parser.error("K debe ser positivo.")

    train = pd.read_csv(
        PROCESSED / "train.csv"
    )

    val = pd.read_csv(
        PROCESSED / "val.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    modelo = RandomRecommender(
        seed=args.seed
    )

    modelo.fit(train, catalogo)

    metricas, detalle = evaluar_modelo(
        modelo,
        val,
        catalogo,
        k=args.k
    )

    print("\n" + "=" * 55)
    print("EVALUACIÓN DE REFERENCIA ALEATORIA")
    print("=" * 55)

    print("Conjunto: val")
    print("Semilla:", args.seed)
    print("K:", args.k)

    print("\nMétricas:")

    for nombre, valor in metricas.items():
        if nombre == "Usuarios_evaluados":
            print(f"{nombre}: {valor}")
        else:
            print(f"{nombre}: {valor:.4f}")

    # Evaluar K de 1 hasta el tamaño del catálogo.
    evaluar_varios_k(
        modelo,
        val,
        catalogo
    )

    evaluar_semillas_aleatorias(
        train,
        val,
        catalogo,
        k=args.k,
        n_semillas=100
    )    

if __name__ == "__main__":
    main()
