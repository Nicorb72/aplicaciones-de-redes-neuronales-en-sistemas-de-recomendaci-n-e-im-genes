
"""
Módulo 3: Recomendador neuronal.
Responsable: Persona 4.

Arquitectura:
- Embeddings de usuarios y destinos.
- Preferencias declaradas del usuario.
- Tipo turístico del destino.
- Perceptrón multicapa (MLP).

Paso inicial: verificar la arquitectura.
"""

from pathlib import Path
import argparse

import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"

RAW = ROOT / "data" / "m3_recomendacion" / "raw"


class NeuralRecommender(nn.Module):

    def __init__(
        self,
        n_users,
        n_items,
        n_preferences,
        n_types,
        embedding_dim=8,
        hidden_dim=32,
        dropout=0.2
    ):
        super().__init__()

        # Factores latentes aprendidos
        self.user_embedding = nn.Embedding(
            n_users, embedding_dim
        )

        self.item_embedding = nn.Embedding(
            n_items, embedding_dim
        )

        # Dimensión total de entrada
        input_dim = (
            2 * embedding_dim
            + n_preferences
            + n_types
        )

        # Perceptrón multicapa
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

        # Concatenar todas las características
        x = torch.cat(
            [
                user_emb,
                item_emb,
                preferences,
                item_types
            ],
            dim=1
        )

        # Logits para BCEWithLogitsLoss
        return self.mlp(x).squeeze(-1)


def verificar_arquitectura(
    embedding_dim=8,
    hidden_dim=32,
    dropout=0.2
):

    train = pd.read_csv(
        PROCESSED / "train.csv"
    )

    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    # Identificadores disponibles en entrenamiento
    user_ids = sorted(train["UserID"].unique())
    item_ids = sorted(catalogo["ItemID"].unique())

    preferencias = sorted(
        usuarios["Preferences"].unique()
    )

    tipos = sorted(
        catalogo["Type"].unique()
    )

    user_map = {
        int(u): idx for idx, u in enumerate(user_ids)
    }

    item_map = {
        int(i): idx for idx, i in enumerate(item_ids)
    }

    # Instanciar red neuronal
    modelo = NeuralRecommender(
        n_users=len(user_ids),
        n_items=len(item_ids),
        n_preferences=len(preferencias),
        n_types=len(tipos),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    # Seleccionar cinco interacciones de prueba interna
    muestra = train.head(5).copy()

    muestra = muestra.merge(
        usuarios[["UserID", "Preferences"]],
        on="UserID",
        validate="many_to_one"
    )

    muestra = muestra.merge(
        catalogo[["ItemID", "Type"]],
        on="ItemID",
        validate="many_to_one"
    )

    user_tensor = torch.tensor(
        muestra["UserID"].map(user_map).to_numpy(),
        dtype=torch.long
    )

    item_tensor = torch.tensor(
        muestra["ItemID"].map(item_map).to_numpy(),
        dtype=torch.long
    )

    # Codificación categórica one-hot
    preferencias_onehot = pd.get_dummies(
        pd.Categorical(
            muestra["Preferences"],
            categories=preferencias
        )
    ).to_numpy(dtype="float32")

    tipos_onehot = pd.get_dummies(
        pd.Categorical(
            muestra["Type"],
            categories=tipos
        )
    ).to_numpy(dtype="float32")

    pref_tensor = torch.tensor(
        preferencias_onehot,
        dtype=torch.float32
    )

    tipo_tensor = torch.tensor(
        tipos_onehot,
        dtype=torch.float32
    )

    # Verificar funcionamiento
    modelo.eval()

    with torch.no_grad():
        logits = modelo(
            user_tensor,
            item_tensor,
            pref_tensor,
            tipo_tensor
        )

        probabilidades = torch.sigmoid(logits)

    parametros = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 60)
    print("VERIFICACIÓN DEL RECOMENDADOR NEURONAL")
    print("=" * 60)

    print("Usuarios de entrenamiento:", len(user_ids))
    print("Destinos:", len(item_ids))
    print("Perfiles de preferencias:", len(preferencias))
    print("Tipos turísticos:", len(tipos))
    print("Dimensión embeddings:", embedding_dim)
    print("Neuronas primera capa:", hidden_dim)
    print("Dropout:", dropout)
    print("Parámetros entrenables:", parametros)

    print("\nDimensiones de entrada:")
    print("Usuarios:", tuple(user_tensor.shape))
    print("Destinos:", tuple(item_tensor.shape))
    print("Preferencias:", tuple(pref_tensor.shape))
    print("Tipos:", tuple(tipo_tensor.shape))

    print("\nDimensión de salida:")
    print(tuple(logits.shape))

    print("\nPuntuaciones iniciales:")
    print(probabilidades.tolist())

    print("\nARQUITECTURA VERIFICADA CORRECTAMENTE")
    print("\nNota: la red todavía no está entrenada.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding-dim", type=int, default=8
    )

    parser.add_argument(
        "--hidden-dim", type=int, default=32
    )

    parser.add_argument(
        "--dropout", type=float, default=0.2
    )

    args = parser.parse_args()

    if args.embedding_dim < 1 or args.hidden_dim < 2:
        parser.error(
            "embedding-dim debe ser >= 1 "
            "y hidden-dim debe ser >= 2."
        )

    if not 0 <= args.dropout < 1:
        parser.error("dropout debe estar entre 0 y 1.")

    verificar_arquitectura(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )


if __name__ == "__main__":
    main()
