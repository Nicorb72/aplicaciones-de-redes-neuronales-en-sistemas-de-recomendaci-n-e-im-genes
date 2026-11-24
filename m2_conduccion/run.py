"""Entradas del módulo para datos, entrenamiento y demostración local."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from common.config_loader import cargar_yaml
from m2_conduccion.data.download import ARCHIVE, download
from m2_conduccion.eda import ROOT, TABLES, prepare

MODELS = ROOT / "models/m2_conduccion"


def train(args, config, frame, cache):
    import pandas as pd
    import torch

    from common.seeds import elegir_dispositivo, fijar_semillas
    from m2_conduccion.baseline_logreg import fit_baseline
    from m2_conduccion.evaluate import evaluate
    from m2_conduccion.search import search
    from m2_conduccion.train import make_loaders

    config.update({key: value for key, value in vars(args).items()
                   if key in {"epochs", "search_trials", "threads"} and value is not None})
    torch.set_num_threads(config["threads"])
    device = elegir_dispositivo(args.device)
    fijar_semillas(config["seed"])
    MODELS.mkdir(parents=True, exist_ok=True)
    loaders = make_loaders(frame, cache, config)
    models, history, durations = {}, [], []
    for family in ("logreg", "scratch", "transfer"):
        start = perf_counter()
        if family == "logreg":
            model, params, score, results = fit_baseline(loaders, device, MODELS, config["seed"])
        else:
            model, params, score, results = search(family, loaders, config, device, MODELS)
        models[family] = model, params, score
        history.extend(results)
        durations.append({"family": family, "configurations": len(results),
                          "seconds": perf_counter() - start, "device": str(device),
                          "seed": config["seed"], "epochs": config["epochs"]})
    pd.DataFrame(durations).to_csv(TABLES / "m2_tiempos_busqueda.csv", index=False)
    if args.mode == "search-timing":
        return
    winner = max(models, key=lambda family: models[family][2])
    (TABLES / "m2_busqueda.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    threshold, temp = evaluate(models, winner, loaders, frame, cache, config, device)
    model, parameters, _ = models[winner]
    checkpoint = {"family": winner, "parameters": parameters, "threshold": threshold,
                  "temperature": temp, "class_grouping": config["class_grouping"],
                  "image_size": config["image_size"], "state_dict": model.state_dict()}
    path = MODELS / "best.pt"
    partial = path.with_suffix(".tmp")
    torch.save(checkpoint, partial)
    partial.replace(path)
    print(f"Modelo seleccionado: {winner}, umbral {threshold:.4f}", flush=True)


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("data", "train", "demo", "stability", "audit-crops",
                                           "video", "search-timing", "alert-review", "figures"),
                        default="train")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--search-trials", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--sound", action="store_true")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    for name in ("epochs", "search_trials", "threads", "seeds"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} debe ser positivo")
    config = cargar_yaml("configs/m2_conduccion.yaml")
    if args.mode == "video":
        import torch

        from m2_conduccion.video import analyze_video

        if args.video is None:
            parser.error("--mode video requiere --video")
        torch.set_num_threads(args.threads or config["threads"])
        print(json.dumps(analyze_video(args.video, config["temporal"], args.threshold, args.sound)))
        return
    if args.mode == "demo":
        from m2_conduccion.predict import classify

        sample = args.image
        if sample is None:
            import numpy as np
            from PIL import Image

            frame, cache_path = prepare(ARCHIVE, config, reuse=True)
            row = frame.index[frame["split"] == "test"][0]
            sample = Image.fromarray(np.load(cache_path, mmap_mode="r")[row])
        result = classify(sample, args.threshold, with_gradcam=False)
        print(json.dumps(result, ensure_ascii=False))
        return
    archive = download() if args.download else args.archive
    if not args.reuse_data and not archive.is_file():
        parser.error(f"Falta {archive}. Usa --download o indica --archive.")
    data = prepare(archive, config, reuse=args.reuse_data)
    if args.mode in {"train", "search-timing"}:
        train(args, config, *data)
    if args.mode == "alert-review":
        from m2_conduccion.alert_policy import review_policies

        review_policies(data[0], config, TABLES)
    if args.mode == "figures":
        import torch

        from m2_conduccion.evaluate import regenerate_figures

        torch.set_num_threads(args.threads or config["threads"])
        regenerate_figures(*data, config["seed"])
    if args.mode == "audit-crops":
        from m2_conduccion.data.crop_audit import run_crop_audit

        run_crop_audit(*data, config)
    if args.mode == "stability":
        import torch

        from common.seeds import elegir_dispositivo
        from m2_conduccion.stability import run_stability

        config["threads"] = args.threads or config["threads"]
        torch.set_num_threads(config["threads"])
        run_stability(*data, config, elegir_dispositivo(args.device), MODELS,
                      args.seeds or config["stability_runs"])


if __name__ == "__main__":
    main()
