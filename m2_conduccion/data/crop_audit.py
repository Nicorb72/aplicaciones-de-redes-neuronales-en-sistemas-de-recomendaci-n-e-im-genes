"""Búsqueda complementaria de recortes descentrados con verificación de forma."""
import numpy as np
import pandas as pd
from PIL import Image

from m2_conduccion.data.duplicates_check import hamming, perceptual_hash
from m2_conduccion.eda import TABLES

SCALES = (0.80, 0.82, 0.84, 0.86, 0.88, 0.90)
ANCHORS = ((0, 0), (0, 1), (1, 0), (1, 1),
           (0.125, 0.125), (0.125, 0.875), (0.875, 0.125), (0.875, 0.875))


def crop(image, scale, anchor):
    width, height = image.size
    w, h = round(width * scale), round(height * scale)
    x, y = round((width - w) * anchor[0]), round((height - h) * anchor[1])
    return image.crop((x, y, x + w, y + h))


def descriptor(image):
    pixels = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    vector = pixels.ravel() - pixels.mean()
    return perceptual_hash(pixels), vector / max(np.linalg.norm(vector), 1)


def descriptors(images):
    values = [descriptor(image) for image in images]
    return np.array([value[0] for value in values], dtype=np.uint64), np.stack(
        [value[1] for value in values])


def cross_partition_pairs(images, splits, distance, min_correlation):
    base_hash, base_vectors = descriptors(images)
    result = {}
    partitions = [(np.flatnonzero(splits == split), np.flatnonzero(splits != split))
                  for split in np.unique(splits)]
    for scale in SCALES:
        for anchor in ANCHORS:
            hashes, vectors = descriptors(crop(image, scale, anchor) for image in images)
            for source, other in partitions:
                for start in range(0, len(source), 128):
                    block = source[start:start + 128]
                    distances = hamming(hashes[block, None] ^ base_hash[None, other])
                    i, j = np.where(distances <= distance)
                    a_rows, b_rows = block[i], other[j]
                    scores = np.einsum("ij,ij->i", vectors[a_rows], base_vectors[b_rows])
                    for a, b, d, score in zip(a_rows, b_rows, distances[i, j], scores):
                        if score >= min_correlation:
                            key = tuple(sorted((int(a), int(b))))
                            if key not in result or score > result[key][-1]:
                                result[key] = (int(a), int(b), scale, *anchor, int(d), float(score))
    return list(result.values())


def probe(images, seed, distance, min_correlation):
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(images), min(120, len(images)), replace=False)
    rows = []
    for name, scale, anchor in (("esquina_80pct", 0.80, (0, 0)),
                                ("esquina_83pct", 0.83, (1, 1)),
                                ("desplazado_83pct", 0.83, (0.10, 0.90))):
        hits = 0
        for i in sample:
            original = images[i]
            target_hash, target_vector = descriptor(crop(original, scale, anchor))
            hashes, vectors = descriptors(crop(original, s, a) for s in SCALES for a in ANCHORS)
            hits += int(((hamming(hashes ^ np.uint64(target_hash)) <= distance) &
                         (vectors @ target_vector >= min_correlation)).any())
        rows.append((name, hits, len(sample)))
    return rows


def run_crop_audit(frame, cache_path, config):
    cache = np.load(cache_path, mmap_mode="r")
    images = [Image.fromarray(cache[i]).convert("L") for i in frame.index]
    distance, correlation = config["crop_hash_distance"], config["crop_min_correlation"]
    pairs = cross_partition_pairs(images, frame["split"].to_numpy(), distance, correlation)
    pd.DataFrame(pairs, columns=["crop_row", "whole_row", "scale", "anchor_x", "anchor_y",
                                 "distance", "correlation"]).to_csv(
        TABLES / "m2_recortes_entre_particiones.csv", index=False)
    pd.DataFrame(probe(images, config["seed"], distance, correlation),
                 columns=["transform", "hits", "pairs"]).to_csv(
        TABLES / "m2_sensibilidad_recortes.csv", index=False)
    print(f"Pares de recortes entre particiones: {len(pairs)}", flush=True)
