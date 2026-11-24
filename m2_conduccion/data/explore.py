"""Inventario del ZIP y vistas reducidas reproducibles para el experimento."""
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from m2_conduccion.data.duplicates_check import fingerprints

CLASSES = ("segura", "giro", "escribir", "hablar", "otros")
DISPLAY = ("Segura", "Giro", "Escribir", "Hablar", "Otros")
ALIASES = (
    {"safedriving", "safe", "normaldriving"},
    {"turning", "turn"},
    {"texting", "textingphone", "textingonthephone"},
    {"talking", "talkingphone", "talkingphones", "talkingonthephone"},
    {"other", "others", "otheractivities", "otherdistractions", "otherdistractingbehaviors"},
)


def camera_model(image):
    return str(image.getexif().get(272, "")).strip() or "Sin EXIF de modelo"


def class_id(name):
    for directory in reversed(PurePosixPath(name).parts[:-1]):
        key = re.sub(r"[^a-z]", "", directory.lower())
        for label, aliases in enumerate(ALIASES):
            if key in aliases:
                return label
    raise ValueError(f"No se reconoce la carpeta de clase: {name}")


def load_rgb(source, size=None):
    with Image.open(source) as opened:
        if size is not None:
            opened.draft("RGB", (size, size))
        return ImageOps.exif_transpose(opened).convert("RGB")


def inventory(archive, output, size):
    with ZipFile(archive) as reader:
        members = sorted(name for name in reader.namelist()
                         if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                         and "__MACOSX" not in PurePosixPath(name).parts
                         and not PurePosixPath(name).name.startswith("."))
        if not members or len(set(members)) != len(members):
            raise ValueError("El ZIP debe contener imágenes con rutas únicas.")
        labels = [class_id(name) for name in members]
        cache = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8,
                                          shape=(len(members), size, size, 3))
        rows, hashes, failures = [], [], []
        for i, (name, label) in enumerate(zip(members, labels)):
            data = reader.read(name)
            try:
                with Image.open(BytesIO(data)) as opened:
                    width, height = opened.size
                    camera = camera_model(opened)
                    opened.draft("RGB", (size, size))
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except (UnidentifiedImageError, OSError) as error:
                failures.append((name, label, type(error).__name__))
                continue
            thumbnail = image.resize((size, size), Image.Resampling.BILINEAR)
            cache[len(rows)] = thumbnail
            luminance = np.asarray(thumbnail.convert("L"))
            rows.append((name, label, width, height, camera, sha256(data).hexdigest(),
                         float(luminance.mean()), float(luminance.std())))
            hashes.append(fingerprints(thumbnail))
            if (i + 1) % 1000 == 0:
                print(f"Inventario: {i + 1}/{len(members)}", flush=True)
        cache.flush()
    columns = ["image", "label", "width", "height", "camera", "sha256", "luminance", "contrast"]
    return (pd.DataFrame(rows, columns=columns), np.asarray(hashes, dtype=np.uint64),
            pd.DataFrame(failures, columns=["image", "label", "error"]))
