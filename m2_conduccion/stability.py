"""Reentrena las configuraciones elegidas con semillas independientes."""
from hashlib import file_digest, sha256
import json
from time import perf_counter

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from common.multiseed import resumir
from common.seeds import fijar_semillas, lista_semillas
from m2_conduccion.data.explore import CLASSES
from m2_conduccion.eda import FIGURES, PROCESSED, TABLES, plt, save_figure
from m2_conduccion.train import build, fit, infer, make_loaders, preprocess
from m2_conduccion.transfer_learning import WEIGHTS_NAME


@torch.no_grad()
def prefix_loaders(model, loaders, device):
    model.to(device=device, memory_format=torch.channels_last).eval()
    result = {}
    for split, loader in loaders.items():
        features, targets = [], []
        for images, labels in loader:
            features.append(model.through_layer3(preprocess(images.to(device))).cpu())
            targets.append(labels)
        dataset = TensorDataset(torch.cat(features), torch.cat(targets))
        result[split] = DataLoader(dataset, batch_size=loader.batch_size)
    return result


def selected_networks(search_path):
    history = json.loads(search_path.read_text(encoding="utf-8"))
    winners = {}
    for family in ("scratch", "transfer"):
        trial = max((row for row in history if row["family"] == family),
                    key=lambda row: row["val_f1"])
        winners[family] = trial["parameters"], len(trial["history"])
    return winners


def score_run(labels, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=list(range(len(CLASSES))), zero_division=0)
    result = {"accuracy": float(accuracy_score(labels, predicted)),
              "macro_f1": float(f1.mean())}
    for name, values in zip(("precision", "recall", "f1"), (precision, recall, f1)):
        result.update({f"{name}_{label}": float(value) for label, value in zip(CLASSES, values)})
    return result


def signature(config, frame, cache, models_dir, device, winners):
    files = {}
    weights = models_dir / WEIGHTS_NAME
    for path in (cache, weights):
        if path == weights and not path.is_file():
            files[WEIGHTS_NAME] = "torchvision:ResNet18_Weights.IMAGENET1K_V1"
            continue
        with path.open("rb") as source:
            files[path.name] = file_digest(source, "sha256").hexdigest()
    metadata = {"partition": sha256(frame.to_csv(index=False).encode()).hexdigest(),
                "files": files, "selected": winners, "batch": config["batch_size"],
                "device": str(device), "threads": config["threads"], "torch": str(torch.__version__)}
    return sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def write_results(rows, expected):
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "m2_semillas.csv", index=False)
    counts = frame.groupby("family")["semilla"].nunique()
    if counts.reindex(["scratch", "transfer"], fill_value=0).eq(expected).all():
        summaries = []
        for family, entries in frame.groupby("family"):
            data = entries.drop(columns="family")
            maximize = [name for name in data if name not in {"semilla", "seconds"}]
            summary = resumir(data, maximize).rename_axis("metric").reset_index()
            summaries.append(summary.assign(family=family, runs=len(entries)))
        pd.concat(summaries, ignore_index=True).to_csv(
            TABLES / "m2_resumen_semillas.csv", index=False)
        plot_stability(frame)


def plot_stability(frame):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([frame.loc[frame.family == name, "macro_f1"] for name in ("scratch", "transfer")])
    reference = pd.read_csv(TABLES / "m2_modelos.csv").set_index("family").at["logreg", "macro_f1"]
    ax.axhline(reference, color="#555555", linestyle="--", label="Regresión logística")
    ax.set(xticks=[1, 2], xticklabels=["CNN propia", "ResNet-18 ajustada"],
           ylabel="F1 macro en prueba")
    ax.legend()
    save_figure(fig, "estabilidad_semillas")


def run_stability(frame, cache, config, device, models_dir, runs):
    loaders = make_loaders(frame, cache, config)
    winners = selected_networks(TABLES / "m2_busqueda.json")
    digest = signature(config, frame, cache, models_dir, device, winners)
    seeds = lista_semillas(config["seed"], runs)
    progress = PROCESSED / f"stability_{digest}.jsonl"
    previous = [json.loads(line) for line in progress.read_text().splitlines()] if progress.exists() else []
    completed = {(row["family"], row["semilla"]): row for row in previous}
    rows, complexity, timings = [], [], []
    for family, (parameters, epochs) in winners.items():
        prototype = build(family, parameters)
        complexity.append({"family": family, "trainable": sum(
            p.numel() for p in prototype.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in prototype.parameters())})
        del prototype
        missing = any((family, seed) not in completed for seed in seeds)
        cached = family == "transfer" and parameters["unfreeze"] == ["layer4"] and parameters["augmentation"] == "none"
        family_loaders = loaders
        prefix_path = progress.with_name(f"{progress.stem}.prefix.json")
        if missing and cached:
            print("Precalculando el prefijo congelado de ResNet-18.", flush=True)
            start = perf_counter()
            prefix_model = build(family, parameters, pretrained=True, weights_dir=models_dir)
            family_loaders = prefix_loaders(prefix_model, loaders, device)
            del prefix_model
            elapsed = perf_counter() - start
            previous_elapsed = json.loads(prefix_path.read_text()) if prefix_path.exists() else 0.0
            prefix_path.write_text(json.dumps(previous_elapsed + elapsed))
        for position, seed in enumerate(seeds, 1):
            entry = completed.get((family, seed))
            if entry is None:
                print(f"{family} · semilla {seed} ({position}/{runs})", flush=True)
                start = perf_counter()
                fijar_semillas(seed)
                model = build(family, parameters, pretrained=True, weights_dir=models_dir)
                run_config = dict(config, seed=seed, epochs=epochs)
                model, val_f1, _ = fit(model, family_loaders["train"], family_loaders["val"],
                                       parameters, run_config, device, prefix_features=cached)
                logits, labels = infer(model, family_loaders["test"], device, prefix_features=cached)
                entry = {"family": family, "semilla": seed, "val_f1": val_f1,
                         **score_run(labels, logits.argmax(dim=1).numpy()),
                         "seconds": perf_counter() - start}
                with progress.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry) + "\n")
            rows.append(entry)
        prefix_seconds = json.loads(prefix_path.read_text()) if cached and prefix_path.exists() else 0.0
        run_seconds = sum(row["seconds"] for row in rows if row["family"] == family)
        timings.append({"family": family, "runs": runs, "prefix_seconds": prefix_seconds,
                        "run_seconds": run_seconds, "total_seconds": run_seconds + prefix_seconds,
                        "device": str(device)})
    reference = build("logreg", {"dropout": 0.0, "unfreeze": []})
    complexity.insert(0, {"family": "logreg", "trainable": reference.classifier[-1].weight.numel()
                          + reference.classifier[-1].bias.numel(),
                          "total": sum(p.numel() for p in reference.parameters())})
    pd.DataFrame(complexity).to_csv(TABLES / "m2_parametros.csv", index=False)
    pd.DataFrame(timings).to_csv(TABLES / "m2_tiempos_estabilidad.csv", index=False)
    write_results(rows, runs)
