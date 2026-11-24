
"""
Módulo 3: Visualización de embeddings.
Responsable: Persona 4.

- t-SNE de embeddings de usuarios.
- PCA de embeddings de destinos.

Modelo neuronal C5, semilla 42.
Entrenamiento con train y selección de pesos con val.
No utiliza test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from m3_recomendacion.train import NeuralRecommendationSystem


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"
FIGURES = ROOT / "reports" / "figures" / "m3"


def pca_2d(matriz):
    """PCA bidimensional mediante descomposición SVD."""

    matriz = np.asarray(matriz, dtype=float)
    centrada = matriz - matriz.mean(axis=0)

    _, _, vt = np.linalg.svd(
        centrada,
        full_matrices=False
    )

    return centrada @ vt[:2].T


def main():

    FIGURES.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )
    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    sistema = NeuralRecommendationSystem(
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        lr=0.001,
        epochs=100,
        seed=42,
        patience=15,
        negative_ratio=None
    )

    sistema.fit(
        train,
        catalogo,
        usuarios,
        val=val,
        k=3
    )

    # --------------------------------------------------------
    # 1. Embeddings de usuarios
    # --------------------------------------------------------

    user_embeddings = (
        sistema.model.user_embedding.weight
        .detach()
        .cpu()
        .numpy()
    )

    user_ids = sorted(
        sistema.user_map,
        key=lambda u: sistema.user_map[u]
    )

    print("\n" + "=" * 60)
    print("VISUALIZACIÓN DE EMBEDDINGS")
    print("=" * 60)

    print(
        "Matriz embeddings usuarios:",
        user_embeddings.shape
    )

    tsne = TSNE(
        n_components=2,
        perplexity=25,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    user_coords = tsne.fit_transform(user_embeddings)

    df_users = pd.DataFrame({
        "UserID": user_ids,
        "X": user_coords[:, 0],
        "Y": user_coords[:, 1]
    })

    df_users["Preferences"] = (
        df_users["UserID"]
        .map(sistema.user_preferences)
    )

    df_users.to_csv(
        PROCESSED / "user_embeddings_2d.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for preferencia, grupo in df_users.groupby(
        "Preferences"
    ):
        ax.scatter(
            grupo["X"],
            grupo["Y"],
            s=19,
            alpha=0.65,
            label=preferencia
        )

    ax.set_title(
        "Proyección t-SNE de embeddings de usuarios"
    )
    ax.set_xlabel("Dimensión t-SNE 1")
    ax.set_ylabel("Dimensión t-SNE 2")
    ax.legend(title="Preferencias", fontsize=9)
    ax.grid(alpha=0.15)

    fig.tight_layout()

    ruta_usuarios = (
        FIGURES / "fig_m3_12_embeddings_usuarios_tsne.png"
    )

    fig.savefig(
        ruta_usuarios,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 2. Embeddings de destinos
    # --------------------------------------------------------

    item_embeddings = (
        sistema.model.item_embedding.weight
        .detach()
        .cpu()
        .numpy()
    )

    item_ids = sorted(
        sistema.item_map,
        key=lambda i: sistema.item_map[i]
    )

    nombres = dict(zip(
        catalogo["ItemID"].astype(int),
        catalogo["Name"]
    ))

    tipos = dict(zip(
        catalogo["ItemID"].astype(int),
        catalogo["Type"]
    ))

    item_coords = pca_2d(item_embeddings)

    df_items = pd.DataFrame({
        "ItemID": item_ids,
        "Destino": [nombres[i] for i in item_ids],
        "Type": [tipos[i] for i in item_ids],
        "X": item_coords[:, 0],
        "Y": item_coords[:, 1]
    })

    df_items.to_csv(
        PROCESSED / "item_embeddings_2d.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        df_items["X"],
        df_items["Y"],
        s=110,
        color="#447B9B"
    )

    for _, fila in df_items.iterrows():
        ax.annotate(
            fila["Destino"],
            (fila["X"], fila["Y"]),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=10
        )

    ax.set_title(
        "Proyección PCA de embeddings de destinos"
    )
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    ruta_destinos = (
        FIGURES / "fig_m3_13_embeddings_destinos_pca.png"
    )

    fig.savefig(
        ruta_destinos,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Matriz embeddings destinos:",
        item_embeddings.shape
    )

    print("\nDestinos proyectados:")
    print(df_items.round(4).to_string(index=False))

    print("\nFIGURAS GENERADAS")
    print("Usuarios:", ruta_usuarios)
    print("Destinos:", ruta_destinos)


if __name__ == "__main__":
    main()
