"""Entradas del módulo para datos, entrenamiento y demostración local."""
import argparse
import json
from pathlib import Path

from common.config_loader import cargar_yaml
from m2_conduccion.data.download import ARCHIVE, download
from m2_conduccion.eda import ROOT, TABLES, prepare

MODELS = ROOT / "models/m2_conduccion"


def train(args, config, frame, cache):
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
    models, history = {}, []
    for family in ("logreg", "scratch", "transfer"):
        if family == "logreg":
            model, params, score, results = fit_baseline(loaders, device, MODELS, config["seed"])
        else:
            model, params, score, results = search(family, loaders, config, device, MODELS)
        models[family] = model, params, score
        history.extend(results)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("data", "train", "demo"), default="train")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--search-trials", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    for name in ("epochs", "search_trials", "threads"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} debe ser positivo")
    config = cargar_yaml("configs/m2_conduccion.yaml")
    if args.mode == "demo":
        from m2_conduccion.predict import classify, load_checkpoint
        if args.image is None:
            model, metadata = load_checkpoint()
            print({key: metadata[key] for key in ("family", "image_size", "threshold", "temperature")})
        else:
            result = classify(args.image, args.threshold, with_gradcam=False)
            print(json.dumps(result, ensure_ascii=False))
        return
    archive = download() if args.download else args.archive
    if not args.reuse_data and not archive.is_file():
        parser.error(f"Falta {archive}. Usa --download o indica --archive.")
    data = prepare(archive, config, reuse=args.reuse_data)
    if args.mode == "train":
        train(args, config, *data)


if __name__ == "__main__":
    main()
