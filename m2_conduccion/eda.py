"""Auditoría de fuga y figuras descriptivas del conjunto publicado."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m2_conduccion.data.duplicates_check import components, perceptual_pairs, probe_recall
from m2_conduccion.data.explore import CLASSES, DISPLAY, inventory
from m2_conduccion.data.split_no_leakage import assign_groups

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/m2_conduccion/processed"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures/m2"


def audit(frame, hashes, config):
    distances = perceptual_pairs(hashes, config["hash_distance"])
    exact = [(int(indices[0]), int(i)) for indices in frame.groupby("sha256").indices.values()
             for i in indices[1:]]
    frame["group"] = components(len(frame), [*distances, *exact])
    frame["split"] = assign_groups(frame, config["split"], config["seed"])
    edges = sorted(set(distances) | set(exact))
    cross = sum(frame.at[i, "split"] != frame.at[j, "split"] for i, j in edges)
    if cross or frame.groupby("sha256")["split"].nunique().max() != 1:
        raise AssertionError("Se detectaron coincidencias que atraviesan particiones.")
    groups = frame.groupby("group")
    summary = {
        "images": len(frame), "exact_redundant_files": len(exact),
        "perceptual_pairs": len(distances), "hash_distance": config["hash_distance"],
        "groups": groups.ngroups, "largest_group": int(groups.size().max()),
        "groups_with_conflicting_labels": int((groups["label"].nunique() > 1).sum()),
        "cross_split_matches": cross,
    }
    pairs = pd.DataFrame([(i, j, distances.get((i, j)), frame.at[i, "sha256"] == frame.at[j, "sha256"])
                          for i, j in edges], columns=["row_a", "row_b", "distance", "identical_file"])
    return summary, pairs


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def show_matches(frame, cache, pairs, seed):
    candidates = pairs[pairs["distance"].notna()]
    if candidates.empty:
        return
    blocks = [subset.sample(n=min(3, len(subset)), random_state=seed)
              for _, subset in candidates.groupby(candidates["distance"] == 0)]
    chosen = pd.concat(blocks).sort_values("distance", kind="stable")
    fig, axes = plt.subplots(len(chosen), 2, figsize=(5, 1.85*len(chosen)), squeeze=False)
    for (left, right), match in zip(axes, chosen.itertuples(index=False)):
        for ax, i in ((left, match.row_a), (right, match.row_b)):
            ax.imshow(cache[i])
            ax.set_title(f"{DISPLAY[frame.at[i, 'label']]} · imagen {i}", fontsize=9)
            ax.axis("off")
        right.set_title(f"{right.get_title()} · Hamming {int(match.distance)}", fontsize=9)
    save_figure(fig, "coincidencias")


def describe_cameras(frame):
    dominant = (frame["width"] == 640) & (frame["height"] == 480)
    frame.assign(is_640x480=dominant).groupby(["is_640x480", "camera"]).size().rename(
        "images").to_csv(TABLES / "m2_camaras.csv")


def describe(frame, cache):
    distribution = pd.crosstab(frame["label"], frame["split"]).reindex(
        index=range(len(CLASSES)), columns=["train", "val", "test"], fill_value=0)
    distribution.index = CLASSES
    distribution.assign(total=distribution.sum(axis=1)).to_csv(TABLES / "m2_distribucion.csv")
    frame.assign(label=frame["label"].map(dict(enumerate(CLASSES)))).groupby(
        ["label", "width", "height"]).size().rename("images").to_csv(TABLES / "m2_resoluciones.csv")
    describe_cameras(frame)
    frame.assign(label=frame["label"].map(dict(enumerate(CLASSES)))).groupby("label")[
        ["luminance", "contrast"]].quantile([0.1, 0.5, 0.9]).to_csv(TABLES / "m2_iluminacion.csv")
    ax = distribution.set_axis(DISPLAY).plot.bar(stacked=True, rot=0, figsize=(8, 4),
                                               color=["#29557a", "#2b8d7d", "#e6a34e"])
    ax.set(ylabel="Imágenes", xlabel="Clase")
    ax.legend(["Entrenamiento", "Validación", "Prueba"])
    save_figure(ax.figure, "distribucion")
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for label, name in enumerate(DISPLAY):
        pool = frame[frame["label"] == label].sort_values("luminance").drop_duplicates("group")
        for row, fraction in enumerate((0.10, 0.50, 0.90)):
            image_id = pool.index[round(fraction * (len(pool) - 1))]
            ax = axes[row, label]
            ax.imshow(cache[image_id])
            ax.set_title(f"{name} · P{fraction:.0%}".replace("%", ""), fontsize=9)
            ax.axis("off")
    save_figure(fig, "ejemplos_clases")


def prepare(archive, config, reuse=False):
    for directory in (PROCESSED, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    cache_path = PROCESSED / f"imagenes_{config['image_size']}.npy"
    manifest = PROCESSED / "manifest.csv"
    if reuse:
        if not (manifest.is_file() and cache_path.is_file()):
            raise FileNotFoundError("Faltan datos procesados. Ejecuta --mode data antes de --reuse-data.")
        frame = pd.read_csv(manifest)
        if not frame["split"].equals(assign_groups(frame, config["split"], config["seed"])):
            raise ValueError("La partición guardada no corresponde al reparto o semilla de la configuración.")
        audit_config = json.loads((TABLES / "m2_auditoria.json").read_text(encoding="utf-8"))
        if audit_config["hash_distance"] != config["hash_distance"]:
            raise ValueError("Ejecuta --mode data para auditar con el nuevo umbral perceptual.")
        return frame, cache_path
    with TemporaryDirectory(dir=PROCESSED) as temporary:
        staging = Path(temporary)
        draft_cache = staging / cache_path.name
        frame, hashes, failures = inventory(archive, draft_cache, config["image_size"])
        print("Auditando similitud entre imágenes", flush=True)
        summary, pairs = audit(frame, hashes, config)
        summary["unreadable_images"] = len(failures)
        failures.assign(label=failures["label"].map(dict(enumerate(CLASSES)))).to_csv(
            TABLES / "m2_archivos_ilegibles.csv", index=False)
        pairs.to_csv(staging / "coincidencias.csv", index=False)
        cache = np.load(draft_cache, mmap_mode="r")
        describe(frame, cache)
        show_matches(frame, cache, pairs, config["seed"])
        probe = probe_recall(cache, config["seed"], config["hash_distance"], len(frame))
        pd.DataFrame(probe, columns=["transform", "hits", "pairs"]).to_csv(
            TABLES / "m2_sensibilidad_huellas.csv", index=False)
        del cache
        frame.to_csv(staging / manifest.name, index=False)
        draft_cache.replace(cache_path)
        (staging / "coincidencias.csv").replace(PROCESSED / "coincidencias.csv")
        (staging / manifest.name).replace(manifest)
        (TABLES / "m2_auditoria.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)
    return frame, cache_path
