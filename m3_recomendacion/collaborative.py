
"""
Módulo 3: Filtrado colaborativo clásico.

Factorización de matrices implementada con PyTorch.
Entrenamiento sobre interacciones observadas.
"""

from m3_recomendacion.evaluate import (
    evaluar_modelo,
    evaluar_varios_k
)
from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"


class MatrixFactorization(nn.Module):

    def __init__(self, n_users, n_items, factors=8):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, factors)
        self.item_embedding = nn.Embedding(n_items, factors)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):

        p = self.user_embedding(users)
        q = self.item_embedding(items)

        interaccion = (p * q).sum(dim=1)

        sesgo_usuario = self.user_bias(users).squeeze(-1)
        sesgo_destino = self.item_bias(items).squeeze(-1)

        return (
            interaccion
            + sesgo_usuario
            + sesgo_destino
            + self.global_bias
        )


class CollaborativeRecommender:

    def __init__(
        self,
        factors=8,
        lr=0.01,
        epochs=100,
        weight_decay=0.001,
        seed=42
    ):
        self.factors = factors
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed

        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.item_ids = []
        self.historial = {}
        self.losses = []

    def fit(self, train, catalogo):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        user_ids = sorted(train["UserID"].unique())
        self.item_ids = sorted(catalogo["ItemID"].unique())

        self.user_map = {
            int(u): idx for idx, u in enumerate(user_ids)
        }

        self.item_map = {
            int(i): idx for idx, i in enumerate(self.item_ids)
        }

        self.historial = (
            train.groupby("UserID")["ItemID"]
            .apply(lambda x: set(x.astype(int)))
            .to_dict()
        )

        users = torch.tensor(
            train["UserID"].map(self.user_map).to_numpy(),
            dtype=torch.long
        )

        items = torch.tensor(
            train["ItemID"].map(self.item_map).to_numpy(),
            dtype=torch.long
        )

        labels = torch.tensor(
            (train["Rating"] >= 4).astype(float).to_numpy(),
            dtype=torch.float32
        )

        self.model = MatrixFactorization(
            n_users=len(user_ids),
            n_items=len(self.item_ids),
            factors=self.factors
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        criterion = nn.BCEWithLogitsLoss()

        self.losses = []
        self.model.train()

        for epoch in range(self.epochs):

            optimizer.zero_grad()

            logits = self.model(users, items)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            self.losses.append(float(loss.item()))

            if (epoch + 1) % 20 == 0:
                print(
                    f"Época {epoch + 1:3d}/{self.epochs} "
                    f"| Pérdida: {loss.item():.4f}"
                )

        self.model.eval()
        return self

    def recommend(self, user_id, k=3):

        if self.model is None:
            raise RuntimeError("Primero debes entrenar el modelo.")

        user_id = int(user_id)
        vistos = self.historial.get(user_id, set())

        candidatos = [
            int(item)
            for item in self.item_ids
            if int(item) not in vistos
        ]

        if not candidatos:
            return []

        # Para un usuario desconocido no hay embedding entrenado.
        if user_id not in self.user_map:
            raise ValueError(
                "Usuario desconocido. Requiere estrategia cold-start."
            )

        user_idx = self.user_map[user_id]

        users = torch.full(
            (len(candidatos),),
            user_idx,
            dtype=torch.long
        )

        items = torch.tensor(
            [self.item_map[i] for i in candidatos],
            dtype=torch.long
        )

        with torch.no_grad():
            logits = self.model(users, items)
            scores = torch.sigmoid(logits).tolist()

        ranking = list(zip(candidatos, scores))
        ranking.sort(key=lambda x: (-x[1], x[0]))

        return ranking[:k]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--weight-decay", type=float, default=0.001
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.k < 1 or args.factors < 1 or args.epochs < 1:
        parser.error("K, factors y epochs deben ser positivos.")

    train = pd.read_csv(PROCESSED / "train.csv")
    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )
    val = pd.read_csv(PROCESSED / "val.csv")

    modelo = CollaborativeRecommender(
        factors=args.factors,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        seed=args.seed
    )

    print("\n" + "=" * 55)
    print("MODELO DE FACTORIZACIÓN DE MATRICES")
    print("=" * 55)

    modelo.fit(train, catalogo)

    parametros = sum(
        p.numel()
        for p in modelo.model.parameters()
        if p.requires_grad
    )

    print("\nParámetros entrenables:", parametros)

    nombres = dict(zip(
        catalogo["ItemID"], catalogo["Name"]
    ))

    print("\nRECOMENDACIONES DE VALIDACIÓN")

    for user_id in sorted(val["UserID"].unique())[:5]:

        vistos = modelo.historial.get(int(user_id), set())

        print(f"\nUsuario {user_id}")
        print("Historial:", [nombres[i] for i in sorted(vistos)])

        recomendaciones = modelo.recommend(
            user_id=int(user_id),
            k=args.k
        )

        for posicion, (item_id, score) in enumerate(
            recomendaciones, start=1
        ):
            print(
                f"{posicion}. {nombres[item_id]} "
                f"| Puntuación: {score:.4f}"
            )

    # Evaluación del filtrado colaborativo
    metricas, detalle = evaluar_modelo(
        modelo,
        val,
        catalogo,
        k=args.k
    )

    print("\n" + "=" * 55)
    print("EVALUACIÓN DEL FILTRADO COLABORATIVO")
    print("=" * 55)

    print("Conjunto: val")
    print("K:", args.k)

    for nombre, valor in metricas.items():
        if nombre == "Usuarios_evaluados":
            print(f"{nombre}: {valor}")
        else:
            print(f"{nombre}: {valor:.4f}")

    # Evaluación para diferentes valores de K
    tabla_k = evaluar_varios_k(
        modelo,
        val,
        catalogo
    )


if __name__ == "__main__":
    main()
