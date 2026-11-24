
"""
Módulo 3: Recomendación basada en popularidad.

Modelo de referencia implementado con PyTorch.

La popularidad se calcula exclusivamente con train.csv.
Se recomiendan destinos que el usuario no haya
observado durante el entrenamiento.
"""

from pathlib import Path
import argparse

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"


class PopularityRecommender:

    def __init__(self):
        self.scores = None
        self.item_ids = None
        self.historial = {}

    def fit(self, train, catalogo):

        # Los destinos se ordenan por identificador.
        self.item_ids = sorted(
            catalogo["ItemID"].astype(int).tolist()
        )

        id_a_indice = {
            item_id: i
            for i, item_id in enumerate(self.item_ids)
        }

        # PyTorch: contar interacciones positivas.
        positivos = train[train["Rating"] >= 4].copy()

        indices = torch.tensor(
            positivos["ItemID"]
            .map(id_a_indice)
            .to_numpy(),
            dtype=torch.long
        )

        self.scores = torch.bincount(
            indices,
            minlength=len(self.item_ids)
        ).float()

        # Historial observado de cada usuario.
        self.historial = (
            train.groupby("UserID")["ItemID"]
            .apply(lambda x: set(x.astype(int)))
            .to_dict()
        )

        return self

    def recommend(self, user_id, k=3):

        if self.scores is None:
            raise RuntimeError(
                "Primero debes ajustar el modelo."
            )

        # Excluir destinos ya observados.
        vistos = self.historial.get(int(user_id), set())

        candidatos = [
            (item_id, float(self.scores[i].item()))
            for i, item_id in enumerate(self.item_ids)
            if item_id not in vistos
        ]

        # Desempate determinista por ItemID.
        candidatos.sort(
            key=lambda x: (-x[1], x[0])
        )

        return candidatos[:k]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k", type=int, default=3
    )

    parser.add_argument(
        "--user-id", type=int, default=None
    )

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k debe ser mayor o igual a 1")

    train = pd.read_csv(
        PROCESSED / "train.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    modelo = PopularityRecommender()
    modelo.fit(train, catalogo)

    nombres = dict(zip(
        catalogo["ItemID"],
        catalogo["Name"]
    ))

    print("\n" + "=" * 55)
    print("MODELO DE REFERENCIA: POPULARIDAD")
    print("=" * 55)

    print("\nInteracciones positivas por destino:")

    for item_id, score in sorted(
        zip(modelo.item_ids, modelo.scores.tolist()),
        key=lambda x: (-x[1], x[0])
    ):
        print(f"{nombres[item_id]}: {int(score)}")

    # Si no se especifica usuario, mostrar 5 de test.
    if args.user_id is None:

        test = pd.read_csv(
            PROCESSED / "test.csv"
        )

        usuarios = sorted(
            test["UserID"].unique()
        )[:5]

    else:
        usuarios = [args.user_id]

    for user_id in usuarios:

        print(f"\nUSUARIO {user_id}")

        vistos = sorted(
            modelo.historial.get(user_id, set())
        )

        print(
            "Historial de entrenamiento:",
            [nombres[i] for i in vistos]
        )

        recomendaciones = modelo.recommend(
            user_id, k=args.k
        )

        print("Destinos recomendados:")

        for posicion, (item_id, score) in enumerate(
            recomendaciones, start=1
        ):
            print(
                f" {posicion}. {nombres[item_id]} "
                f"(positivos en train: {int(score)})"
            )

        if len(recomendaciones) < args.k:
            print(
                "Hay menos destinos disponibles "
                "que el K solicitado."
            )


if __name__ == "__main__":
    main()

