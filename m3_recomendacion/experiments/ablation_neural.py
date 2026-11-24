"""
Módulo 3: Ablación del recomendador neuronal.

Objetivo:
Comparar variantes del modelo neuronal para medir
el aporte de las variables de contenido.

Variantes:
1. Solo IDs
2. IDs + preferencias
3. IDs + tipo
4. Completo
"""

from pathlib import Path
from copy import deepcopy
import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from m3_recomendacion.evaluation.evaluate import evaluar_modelo

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"


class FlexibleNeuralRecommender(nn.Module):

    def __init__(
        self,
        n_users,
        n_items,
        n_preferences,
        n_types,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        use_preferences=True,
        use_types=True
    ):
        super().__init__()

        self.use_preferences = use_preferences
        self.use_types = use_types

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        input_dim = 2 * embedding_dim

        if self.use_preferences:
            input_dim += n_preferences

        if self.use_types:
            input_dim += n_types

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        user_ids,
        item_ids,
        preferences,
        item_types
    ):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        features = [user_emb, item_emb]

        if self.use_preferences:
            features.append(preferences)

        if self.use_types:
            features.append(item_types)

        x = torch.cat(features, dim=1)
        return self.mlp(x).squeeze(-1)


class AblationNeuralSystem:

    def __init__(
        self,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        lr=0.001,
        epochs=100,
        seed=42,
        patience=15,
        use_preferences=True,
        use_types=True
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.patience = patience
        self.use_preferences = use_preferences
        self.use_types = use_types

        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.historial = {}
        self.item_ids = []
        self.user_preferences = {}
        self.item_types = {}
        self.pref_map = {}
        self.type_map = {}
        self.losses = []
        self.val_history = []
        self.best_epoch = None
        self.best_ndcg = None

    def preparar(self, train, catalogo, usuarios):

        self.user_map = {
            int(u): i
            for i, u in enumerate(sorted(train["UserID"].unique()))
        }

        self.item_ids = sorted(catalogo["ItemID"].astype(int))

        self.item_map = {
            item: i for i, item in enumerate(self.item_ids)
        }

        preferencias = sorted(usuarios["Preferences"].unique())
        tipos = sorted(catalogo["Type"].unique())

        self.pref_map = {p: i for i, p in enumerate(preferencias)}
        self.type_map = {t: i for i, t in enumerate(tipos)}

        self.user_preferences = dict(zip(
            usuarios["UserID"].astype(int),
            usuarios["Preferences"]
        ))

        self.item_types = dict(zip(
            catalogo["ItemID"].astype(int),
            catalogo["Type"]
        ))

        self.historial = (
            train.groupby("UserID")["ItemID"]
            .apply(lambda x: set(x.astype(int)))
            .to_dict()
        )

        self.model = FlexibleNeuralRecommender(
            n_users=len(self.user_map),
            n_items=len(self.item_map),
            n_preferences=len(self.pref_map),
            n_types=len(self.type_map),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            use_preferences=self.use_preferences,
            use_types=self.use_types
        )

    def codificar(self, user_ids, item_ids):

        user_ids = [int(u) for u in user_ids]
        item_ids = [int(i) for i in item_ids]

        users = torch.tensor(
            [self.user_map[u] for u in user_ids],
            dtype=torch.long
        )

        items = torch.tensor(
            [self.item_map[i] for i in item_ids],
            dtype=torch.long
        )

        prefs = torch.zeros(
            len(user_ids),
            len(self.pref_map),
            dtype=torch.float32
        )

        types = torch.zeros(
            len(item_ids),
            len(self.type_map),
            dtype=torch.float32
        )

        for j, user_id in enumerate(user_ids):
            pref = self.user_preferences[user_id]
            prefs[j, self.pref_map[pref]] = 1.0

        for j, item_id in enumerate(item_ids):
            tipo = self.item_types[item_id]
            types[j, self.type_map[tipo]] = 1.0

        return users, items, prefs, types

    def fit(self, train, catalogo, usuarios, val=None, k=3):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.preparar(train, catalogo, usuarios)

        entradas = self.codificar(
            train["UserID"].tolist(),
            train["ItemID"].tolist()
        )

        etiquetas = torch.tensor(
            train["Positiva"].astype(float).to_numpy(),
            dtype=torch.float32
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        criterion = nn.BCEWithLogitsLoss()

        mejor_ndcg = float("-inf")
        mejor_epoca = 0
        mejores_pesos = None
        epocas_sin_mejora = 0
        tolerancia = 1e-4

        self.losses = []
        self.val_history = []

        for epoch in range(self.epochs):

            self.model.train()
            optimizer.zero_grad()

            logits = self.model(*entradas)
            loss = criterion(logits, etiquetas)

            loss.backward()
            optimizer.step()

            self.losses.append(float(loss.item()))

            if val is not None:
                self.model.eval()

                with torch.no_grad():
                    metricas, _ = evaluar_modelo(
                        self, val, catalogo, k=k
                    )

                ndcg = float(metricas["NDCG@K"])
                self.val_history.append(ndcg)

                if ndcg > mejor_ndcg + tolerancia:
                    mejor_ndcg = ndcg
                    mejor_epoca = epoch + 1
                    mejores_pesos = deepcopy(
                        self.model.state_dict()
                    )
                    epocas_sin_mejora = 0
                else:
                    epocas_sin_mejora += 1

                if epocas_sin_mejora >= self.patience:
                    break

        if mejores_pesos is not None:
            self.model.load_state_dict(mejores_pesos)

        self.best_epoch = mejor_epoca
        self.best_ndcg = mejor_ndcg
        self.model.eval()

        return self

    def recommend(self, user_id, k=3):

        if self.model is None:
            raise RuntimeError("Primero debes entrenar el modelo.")

        user_id = int(user_id)

        if user_id not in self.user_map:
            raise ValueError("Usuario desconocido.")

        vistos = self.historial.get(user_id, set())

        candidatos = [
            item for item in self.item_ids
            if item not in vistos
        ]

        if not candidatos:
            return []

        entradas = self.codificar(
            [user_id] * len(candidatos),
            candidatos
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(*entradas)
            scores = torch.sigmoid(logits).tolist()

        ranking = list(zip(candidatos, scores))
        ranking.sort(key=lambda x: (-x[1], x[0]))

        return ranking[:k]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=3)

    args = parser.parse_args()

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    catalogo = pd.read_csv(PROCESSED / "catalogo_destinos.csv")
    usuarios = pd.read_csv(RAW / "Final_Updated_Expanded_Users.csv")

    variantes = [
        ("Solo_IDs", False, False),
        ("IDs_Preferences", True, False),
        ("IDs_Types", False, True),
        ("Completo", True, True),
    ]

    resultados = []

    print("\n" + "=" * 60)
    print("ABLACIÓN DEL RECOMENDADOR NEURONAL")
    print("=" * 60)

    for nombre, use_pref, use_type in variantes:

        print("\n" + "-" * 50)
        print(f"Variante: {nombre}")
        print(
            f"use_preferences={use_pref} | use_types={use_type}"
        )

        sistema = AblationNeuralSystem(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            patience=args.patience,
            use_preferences=use_pref,
            use_types=use_type
        )

        sistema.fit(
            train, catalogo, usuarios, val=val, k=args.k
        )

        metricas, _ = evaluar_modelo(
            sistema, val, catalogo, k=args.k
        )

        parametros = sum(
            p.numel()
            for p in sistema.model.parameters()
            if p.requires_grad
        )

        resultados.append({
            "Variante": nombre,
            "UsePreferences": use_pref,
            "UseTypes": use_type,
            "BestEpoch": sistema.best_epoch,
            "Precision": metricas["Precision@K"],
            "Recall": metricas["Recall@K"],
            "NDCG": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"],
            "Parametros": parametros
        })

        print(f"Mejor época: {sistema.best_epoch}")
        print(f"NDCG@{args.k}: {metricas['NDCG@K']:.4f}")

    tabla = pd.DataFrame(resultados)
    tabla = tabla.sort_values(
        ["NDCG", "Parametros"],
        ascending=[False, True]
    )

    ruta = PROCESSED / "ablation_neural.csv"
    tabla.to_csv(ruta, index=False)

    print("\n" + "=" * 60)
    print("RESULTADOS DE ABLACIÓN")
    print("=" * 60)
    print(tabla.round(4).to_string(index=False))

    print("\nMejor variante:")
    print(tabla.iloc[0].to_string())

    print("\nResultados guardados en:", ruta)


if __name__ == "__main__":
    main()