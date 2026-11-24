"""Predicción local desde el checkpoint del experimento."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from m2_conduccion.alert_policy import risk_indices, risk_score
from m2_conduccion.data.explore import CLASSES, load_rgb
from m2_conduccion.gradcam import explain
from m2_conduccion.train import build, preprocess

CHECKPOINT = Path(__file__).resolve().parents[1] / "models/m2_conduccion/best.pt"


@lru_cache(maxsize=2)
def _load(path, modified):
    metadata = torch.load(path, map_location="cpu", weights_only=True)
    model = build(metadata["family"], metadata["parameters"])
    model.load_state_dict(metadata.pop("state_dict"))
    return model.eval(), metadata


def load_checkpoint(path=None):
    path = Path(path) if path is not None else CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}. Ejecuta python -m m2_conduccion.run --mode train.")
    return _load(path.resolve(), path.stat().st_mtime_ns)


def classify(image, threshold=None, *, with_gradcam=True, checkpoint=None):
    model, metadata = load_checkpoint(checkpoint)
    threshold = float(metadata["threshold"] if threshold is None else threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    size = metadata["image_size"]
    rgb = (ImageOps.exif_transpose(image).convert("RGB") if isinstance(image, Image.Image)
           else load_rgb(image, size))
    pixels = np.array(rgb.resize((size, size), Image.Resampling.BILINEAR))
    inputs = preprocess(torch.from_numpy(pixels).permute(2, 0, 1)[None])
    with torch.inference_mode():
        probabilities = (model(inputs)[0] / metadata["temperature"]).softmax(dim=0).numpy()
    label = int(probabilities.argmax())
    score = float(risk_score(probabilities.astype(float), risk_indices(metadata["class_grouping"])))
    return {"clase": CLASSES[label], "probabilidades": dict(zip(CLASSES, map(float, probabilities))),
            "riesgo": score, "umbral": threshold, "alerta": score >= threshold,
            "gradcam": explain(model, inputs, label) if with_gradcam else None}
