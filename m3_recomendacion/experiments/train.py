
"""
Módulo 3: Entrenamiento del recomendador neuronal.

Incluye:
- Embeddings de usuarios y destinos.
- Preferencias del usuario y tipos turísticos.
- Entrenamiento con PyTorch.
- Función de pérdida BCEWithLogitsLoss.
- Muestreo opcional de negativos observados.
- Early stopping basado en NDCG@K.
- Recuperación de los mejores pesos.
- Evaluación en validación.
- Exportación del historial de entrenamiento.
- Gráficas de pérdida y NDCG.

No utiliza el conjunto de prueba para entrenamiento.
"""

from pathlib import Path
from copy import deepcopy
import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

from m3_recomendacion.neural_recommender import NeuralRecommender
from m3_recomendacion.evaluate import evaluar_modelo


# ============================================================
# 1. RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = (
    ROOT / "data" / "m3_recomendacion" / "processed"
)

RAW = (
    ROOT / "data" / "m3_recomendacion" / "raw"
)

FIGURES = (
    ROOT / "reports" / "figures" / "m3"
)


# ============================================================
# 2. SISTEMA DE RECOMENDACIÓN NEURONAL
# ============================================================

class NeuralRecommendationSystem:

    def __init__(
        self,
        embedding_dim=8,
        hidden_dim=32,
        dropout=0.2,
        lr=0.001,
        epochs=100,
        seed=42,
        patience=15,
        negative_ratio=None
    ):

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.patience = patience
        self.negative_ratio = negative_ratio

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

        self.n_train_positivos = None
        self.n_train_negativos = None


    # ========================================================
    # 3. PREPARAR USUARIOS, DESTINOS Y MODELO
    # ========================================================

    def preparar(self, train, catalogo, usuarios):

        self.user_map = {
            int(u): i
            for i, u in enumerate(
                sorted(train["UserID"].unique())
            )
        }

        self.item_ids = sorted(
            catalogo["ItemID"].astype(int).tolist()
        )

        self.item_map = {
            item: i
            for i, item in enumerate(self.item_ids)
        }

        preferencias = sorted(
            usuarios["Preferences"].unique()
        )

        tipos = sorted(
            catalogo["Type"].unique()
        )

        self.pref_map = {
            pref: i
            for i, pref in enumerate(preferencias)
        }

        self.type_map = {
            tipo: i
            for i, tipo in enumerate(tipos)
        }

        self.user_preferences = dict(
            zip(
                usuarios["UserID"].astype(int),
                usuarios["Preferences"]
            )
        )

        self.item_types = dict(
            zip(
                catalogo["ItemID"].astype(int),
                catalogo["Type"]
            )
        )

        # IMPORTANTE:
        # El historial se obtiene del train COMPLETO.
        # No se construye sobre la muestra utilizada
        # para calcular la función de pérdida.
        self.historial = (
            train.groupby("UserID")["ItemID"]
            .apply(
                lambda x: set(x.astype(int))
            )
            .to_dict()
        )

        self.model = NeuralRecommender(
            n_users=len(self.user_map),
            n_items=len(self.item_map),
            n_preferences=len(self.pref_map),
            n_types=len(self.type_map),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )


    # ========================================================
    # 4. CODIFICAR ENTRADAS
    # ========================================================

    def codificar(self, user_ids, item_ids):

        user_ids = [
            int(u) for u in user_ids
        ]

        item_ids = [
            int(i) for i in item_ids
        ]

        users = torch.tensor(
            [
                self.user_map[u]
                for u in user_ids
            ],
            dtype=torch.long
        )

        items = torch.tensor(
            [
                self.item_map[i]
                for i in item_ids
            ],
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

            prefs[
                j,
                self.pref_map[pref]
            ] = 1.0

        for j, item_id in enumerate(item_ids):

            tipo = self.item_types[item_id]

            types[
                j,
                self.type_map[tipo]
            ] = 1.0

        return users, items, prefs, types


    # ========================================================
    # 5. MUESTREO DE NEGATIVOS OBSERVADOS
    # ========================================================

    def preparar_muestra_entrenamiento(self, train):

        # Si no se solicita muestreo,
        # se conservan todas las interacciones observadas.
        if self.negative_ratio is None:

            train_loss = train.copy()

        else:

            if self.negative_ratio <= 0:
                raise ValueError(
                    "negative_ratio debe ser mayor que cero."
                )

            positivos = train.loc[
                train["Positiva"].astype(int) == 1
            ]

            negativos = train.loc[
                train["Positiva"].astype(int) == 0
            ]

            cantidad_negativos = min(
                len(negativos),
                round(
                    len(positivos) * self.negative_ratio
                )
            )

            if cantidad_negativos < 1:
                raise ValueError(
                    "La proporción seleccionada no permite "
                    "muestrear ejemplos negativos."
                )

            # Selección reproducible de negativos
            # entre los realmente observados.
            negativos_muestra = negativos.sample(
                n=cantidad_negativos,
                replace=False,
                random_state=self.seed
            )

            train_loss = pd.concat(
                [
                    positivos,
                    negativos_muestra
                ],
                ignore_index=True
            )

            train_loss = train_loss.sample(
                frac=1,
                random_state=self.seed
            ).reset_index(drop=True)

        self.n_train_positivos = int(
            train_loss["Positiva"].astype(int).sum()
        )

        self.n_train_negativos = (
            len(train_loss) - self.n_train_positivos
        )

        print("\nMUESTRA UTILIZADA PARA LA PÉRDIDA")
        print(
            "Positivos observados:",
            self.n_train_positivos
        )
        print(
            "Negativos observados:",
            self.n_train_negativos
        )
        print(
            "Total:",
            len(train_loss)
        )

        if self.negative_ratio is None:
            print("Muestreo: desactivado")
        else:
            print(
                "Relación solicitada:",
                self.negative_ratio
            )

        return train_loss


    # ========================================================
    # 6. ENTRENAMIENTO CON EARLY STOPPING
    # ========================================================

    def fit(
        self,
        train,
        catalogo,
        usuarios,
        val=None,
        k=3
    ):

        # Reproducibilidad.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Construir modelo e historial con train completo.
        self.preparar(
            train,
            catalogo,
            usuarios
        )

        # Aplicar muestreo exclusivamente sobre
        # los ejemplos que intervienen en la pérdida.
        train_loss = self.preparar_muestra_entrenamiento(
            train
        )

        entradas = self.codificar(
            train_loss["UserID"].tolist(),
            train_loss["ItemID"].tolist()
        )

        etiquetas = torch.tensor(
            train_loss["Positiva"]
            .astype(float)
            .to_numpy(),
            dtype=torch.float32
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        criterion = nn.BCEWithLogitsLoss()

        tolerancia = 1e-4

        mejor_ndcg = float("-inf")
        mejor_epoca = 0
        mejores_pesos = None

        epocas_sin_mejora = 0

        self.losses = []
        self.val_history = []

        self.best_epoch = None
        self.best_ndcg = None

        print("\n" + "=" * 60)
        print("ENTRENAMIENTO DEL RECOMENDADOR NEURONAL")
        print("=" * 60)

        print("Semilla:", self.seed)
        print("Épocas máximas:", self.epochs)
        print("Paciencia:", self.patience)

        # ----------------------------------------------------
        # Ciclo principal
        # ----------------------------------------------------

        for epoch in range(self.epochs):

            self.model.train()

            optimizer.zero_grad()

            logits = self.model(*entradas)

            loss = criterion(
                logits,
                etiquetas
            )

            loss.backward()

            optimizer.step()

            perdida = float(loss.item())

            self.losses.append(perdida)

            # ------------------------------------------------
            # Evaluación en validación
            # ------------------------------------------------

            if val is not None:

                self.model.eval()

                with torch.no_grad():

                    metricas, _ = evaluar_modelo(
                        self,
                        val,
                        catalogo,
                        k=k
                    )

                ndcg = float(
                    metricas["NDCG@K"]
                )

                self.val_history.append(ndcg)

                # Guardar los mejores pesos
                # según la métrica de ranking.
                if ndcg > mejor_ndcg + tolerancia:

                    mejor_ndcg = ndcg
                    mejor_epoca = epoch + 1

                    mejores_pesos = deepcopy(
                        self.model.state_dict()
                    )

                    epocas_sin_mejora = 0

                else:

                    epocas_sin_mejora += 1

            # ------------------------------------------------
            # Mostrar progreso
            # ------------------------------------------------

            if (epoch + 1) % 10 == 0 or epoch == 0:

                mensaje = (
                    f"Época {epoch + 1:3d}/{self.epochs}"
                    f" | Pérdida train: {perdida:.4f}"
                )

                if val is not None:

                    mensaje += (
                        f" | NDCG@{k} val: "
                        f"{ndcg:.4f}"
                    )

                print(mensaje)

            # ------------------------------------------------
            # Parada temprana
            # ------------------------------------------------

            if (
                val is not None
                and epocas_sin_mejora >= self.patience
            ):

                print(
                    f"\nParada temprana en época "
                    f"{epoch + 1}"
                )

                break

        # ----------------------------------------------------
        # Recuperar mejores pesos
        # ----------------------------------------------------

        if mejores_pesos is not None:

            self.model.load_state_dict(
                mejores_pesos
            )

            self.best_epoch = mejor_epoca
            self.best_ndcg = mejor_ndcg

            print("\nMEJOR MODELO RECUPERADO")
            print(
                "Mejor época:",
                self.best_epoch
            )
            print(
                f"Mejor NDCG@{k}: "
                f"{self.best_ndcg:.4f}"
            )

        self.model.eval()

        return self


    # ========================================================
    # 7. RECOMENDACIONES
    # ========================================================

    def recommend(self, user_id, k=3):

        if self.model is None:
            raise RuntimeError(
                "Primero debes entrenar el modelo."
            )

        user_id = int(user_id)

        if user_id not in self.user_map:

            raise ValueError(
                "Usuario desconocido: falta implementar "
                "la estrategia cold-start."
            )

        # Historial original, antes del muestreo.
        vistos = self.historial.get(
            user_id,
            set()
        )

        candidatos = [
            item
            for item in self.item_ids
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

            scores = torch.sigmoid(
                logits
            ).tolist()

        ranking = list(
            zip(
                candidatos,
                scores
            )
        )

        ranking.sort(
            key=lambda x: (-x[1], x[0])
        )

        return ranking[:k]


# ============================================================
# 8. GRÁFICAS E HISTORIAL
# ============================================================

def guardar_resultados_entrenamiento(
    sistema,
    k,
    negative_ratio=None
):

    FIGURES.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    # Nombres diferentes para evitar sobrescribir
    # los resultados del entrenamiento original.
    if negative_ratio is None:
        sufijo = "original"
    else:
        sufijo = (
            f"negativos_{negative_ratio:g}"
            .replace(".", "p")
        )

    epocas = list(
        range(
            1,
            len(sistema.losses) + 1
        )
    )

    historial = pd.DataFrame({
        "Epoch": epocas,
        "TrainLoss": sistema.losses,
        "ValNDCG": sistema.val_history
        if sistema.val_history
        else [float("nan")] * len(epocas)
    })

    ruta_historial = (
        PROCESSED
        / f"neural_training_history_{sufijo}.csv"
    )

    historial.to_csv(
        ruta_historial,
        index=False
    )

    # --------------------------------------------------------
    # Figura 1: Pérdida de entrenamiento
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        epocas,
        sistema.losses,
        linewidth=2,
        label="Pérdida de entrenamiento"
    )

    if sistema.best_epoch is not None:

        ax.axvline(
            sistema.best_epoch,
            color="green",
            linestyle="--",
            label=(
                f"Mejor época: "
                f"{sistema.best_epoch}"
            )
        )

    ax.set_title(
        "Pérdida de entrenamiento del recomendador neuronal"
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("Entropía cruzada binaria")

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    ruta_loss = (
        FIGURES
        / f"fig_m3_05_neural_train_loss_{sufijo}.png"
    )

    fig.savefig(
        ruta_loss,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Figura 2: NDCG de validación
    # --------------------------------------------------------

    if sistema.val_history:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            epocas,
            sistema.val_history,
            linewidth=2,
            label=f"NDCG@{k} de validación"
        )

        if sistema.best_epoch is not None:

            ax.axvline(
                sistema.best_epoch,
                color="green",
                linestyle="--",
                label=(
                    f"Mejor época: "
                    f"{sistema.best_epoch}"
                )
            )

        ax.axvline(
            len(epocas),
            color="red",
            linestyle=":",
            label=(
                f"Final del entrenamiento: "
                f"{len(epocas)}"
            )
        )

        ax.set_title(
            "Evolución de NDCG en validación"
        )

        ax.set_xlabel("Época")
        ax.set_ylabel(f"NDCG@{k}")

        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        ruta_ndcg = (
            FIGURES
            / f"fig_m3_06_neural_val_ndcg_{sufijo}.png"
        )

        fig.savefig(
            ruta_ndcg,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    print("\nHISTORIAL Y FIGURAS GUARDADOS")
    print("Historial:", ruta_historial)
    print("Figura pérdida:", ruta_loss)

    if sistema.val_history:
        print("Figura NDCG:", ruta_ndcg)


# ============================================================
# 9. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento neuronal con early stopping "
            "y muestreo opcional de negativos observados."
        )
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=8
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15
    )

    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=None,
        help=(
            "Cantidad de negativos observados por "
            "cada positivo. Por defecto utiliza "
            "todas las interacciones observadas."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validación de argumentos
    # --------------------------------------------------------

    if (
        args.embedding_dim < 1
        or args.hidden_dim < 2
        or args.epochs < 1
        or args.patience < 1
        or args.k < 1
        or args.lr <= 0
        or not 0 <= args.dropout < 1
    ):

        parser.error(
            "Parámetros de entrenamiento inválidos."
        )

    if (
        args.negative_ratio is not None
        and (
            not np.isfinite(args.negative_ratio)
            or args.negative_ratio <= 0
        )
    ):

        parser.error(
            "--negative-ratio debe ser un número "
            "finito mayor que cero."
        )

    # --------------------------------------------------------
    # Carga de datos
    # --------------------------------------------------------

    train = pd.read_csv(
        PROCESSED / "train.csv"
    )

    val = pd.read_csv(
        PROCESSED / "val.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    # --------------------------------------------------------
    # Construcción del sistema
    # --------------------------------------------------------

    sistema = NeuralRecommendationSystem(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        patience=args.patience,
        negative_ratio=args.negative_ratio
    )

    # --------------------------------------------------------
    # Entrenar
    # --------------------------------------------------------

    sistema.fit(
        train,
        catalogo,
        usuarios,
        val=val,
        k=args.k
    )

    # --------------------------------------------------------
    # Parámetros entrenables
    # --------------------------------------------------------

    parametros = sum(
        p.numel()
        for p in sistema.model.parameters()
        if p.requires_grad
    )

    # --------------------------------------------------------
    # Evaluar en validación
    # --------------------------------------------------------

    metricas, detalle = evaluar_modelo(
        sistema,
        val,
        catalogo,
        k=args.k
    )

    print("\n" + "=" * 60)
    print("RESULTADOS DEL RECOMENDADOR NEURONAL")
    print("=" * 60)

    print(
        "Parámetros entrenables:",
        parametros
    )

    print(
        "Semilla:",
        args.seed
    )

    print(
        "K:",
        args.k
    )

    print(
        "Relación de negativos:",
        (
            "Todos los observados"
            if args.negative_ratio is None
            else args.negative_ratio
        )
    )

    print(
        "Mejor época:",
        sistema.best_epoch
    )

    for nombre, valor in metricas.items():

        if nombre == "Usuarios_evaluados":

            print(
                f"{nombre}: {valor}"
            )

        else:

            print(
                f"{nombre}: {valor:.4f}"
            )

    print(
        "\nPrimeros cinco usuarios evaluados:"
    )

    print(
        detalle.head().to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Guardar historial y figuras
    # --------------------------------------------------------

    guardar_resultados_entrenamiento(
        sistema,
        k=args.k,
        negative_ratio=args.negative_ratio
    )


# ============================================================
# 10. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
