"""Guardar y cargar el recomendador neuronal C5 sin reentrenarlo.

Ejecución desde la raíz del repositorio:
    python -m m3_recomendacion.models.model_io

El checkpoint contiene únicamente tensores y estructuras de datos básicas.
No se serializa ningún objeto Python arbitrario.
"""

from pathlib import Path

import pandas as pd
import torch

from m3_recomendacion.experiments.train import NeuralRecommendationSystem
from m3_recomendacion.models.neural_recommender import NeuralRecommender

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"
CHECKPOINT = ROOT / "artifacts" / "m3" / "neural_c5.pt"


def guardar_modelo(sistema: NeuralRecommendationSystem, ruta=CHECKPOINT):
    """Exporta los pesos, índices, variables categóricas e historial visto."""
    if sistema.model is None:
        raise RuntimeError("El modelo debe estar entrenado antes de guardarse.")

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    paquete = {
        "format_version": 1,
        "architecture": "NeuralRecommender_C5",
        "config": {
            "embedding_dim": int(sistema.embedding_dim),
            "hidden_dim": int(sistema.hidden_dim),
            "dropout": float(sistema.dropout),
            "lr": float(sistema.lr),
            "epochs": int(sistema.epochs),
            "seed": int(sistema.seed),
            "patience": int(sistema.patience),
            "negative_ratio": sistema.negative_ratio,
        },
        "state_dict": {k: v.detach().cpu() for k, v in sistema.model.state_dict().items()},
        "user_map": {int(k): int(v) for k, v in sistema.user_map.items()},
        "item_map": {int(k): int(v) for k, v in sistema.item_map.items()},
        "item_ids": [int(i) for i in sistema.item_ids],
        "pref_map": dict(sistema.pref_map),
        "type_map": dict(sistema.type_map),
        "user_preferences": {int(k): str(v) for k, v in sistema.user_preferences.items()},
        "item_types": {int(k): str(v) for k, v in sistema.item_types.items()},
        "historial": {int(k): sorted(int(i) for i in vistos) for k, vistos in sistema.historial.items()},
        "best_epoch": sistema.best_epoch,
        "best_ndcg": sistema.best_ndcg,
    }
    torch.save(paquete, ruta)
    return ruta


def cargar_modelo(ruta=CHECKPOINT):
    """Reconstruye un sistema utilizable mediante sistema.recommend(user_id, k)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {ruta}")

    paquete = torch.load(ruta, map_location="cpu", weights_only=True)
    if paquete.get("format_version") != 1 or paquete.get("architecture") != "NeuralRecommender_C5":
        raise ValueError("Formato de checkpoint no compatible.")

    config = paquete["config"]
    sistema = NeuralRecommendationSystem(**config)
    sistema.user_map = paquete["user_map"]
    sistema.item_map = paquete["item_map"]
    sistema.item_ids = paquete["item_ids"]
    sistema.pref_map = paquete["pref_map"]
    sistema.type_map = paquete["type_map"]
    sistema.user_preferences = paquete["user_preferences"]
    sistema.item_types = paquete["item_types"]
    sistema.historial = {int(k): set(v) for k, v in paquete["historial"].items()}
    sistema.best_epoch = paquete["best_epoch"]
    sistema.best_ndcg = paquete["best_ndcg"]

    sistema.model = NeuralRecommender(
        n_users=len(sistema.user_map),
        n_items=len(sistema.item_map),
        n_preferences=len(sistema.pref_map),
        n_types=len(sistema.type_map),
        embedding_dim=sistema.embedding_dim,
        hidden_dim=sistema.hidden_dim,
        dropout=sistema.dropout,
    )
    sistema.model.load_state_dict(paquete["state_dict"], strict=True)
    sistema.model.eval()
    return sistema


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    catalogo = pd.read_csv(PROCESSED / "catalogo_destinos.csv")
    usuarios = pd.read_csv(RAW / "Final_Updated_Expanded_Users.csv")

    sistema = NeuralRecommendationSystem(
        embedding_dim=8, hidden_dim=16, dropout=0.1,
        lr=0.001, epochs=100, seed=42, patience=15,
        negative_ratio=None,
    )
    sistema.fit(train, catalogo, usuarios, val=val, k=3)

    ruta = guardar_modelo(sistema)
    restaurado = cargar_modelo(ruta)

    ejemplo = next((u for u in sorted(sistema.user_map) if sistema.recommend(u, k=3)), None)
    if ejemplo is None:
        raise RuntimeError("No se encontró ningún usuario con destinos por recomendar.")

    antes = sistema.recommend(ejemplo, k=3)
    despues = restaurado.recommend(ejemplo, k=3)
    if [i for i, _ in antes] != [i for i, _ in despues]:
        raise AssertionError("El ranking cambió después de cargar el modelo.")
    for (_, a), (_, b) in zip(antes, despues):
        if abs(a - b) > 1e-7:
            raise AssertionError("Los puntajes cambiaron después de cargar el modelo.")

    print("\nMODELO GUARDADO Y RECUPERADO CORRECTAMENTE")
    print("Archivo:", ruta)
    print("Mejor época:", restaurado.best_epoch)
    print("Mejor NDCG@3 en validación:", f"{restaurado.best_ndcg:.4f}")
    print("Usuario de prueba:", ejemplo)
    print("Recomendaciones:", despues)
    print("Se verificó que el ranking y los puntajes son idénticos.")


if __name__ == "__main__":
    main()
