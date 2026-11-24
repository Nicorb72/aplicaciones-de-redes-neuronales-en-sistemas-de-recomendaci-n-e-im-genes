"""Búsqueda aleatoria reproducible con presupuesto uniforme por familia."""
import random

from common.seeds import fijar_semillas
from m2_conduccion.train import build, fit


def candidates(family, seed, trials):
    rng = random.Random(seed)
    for i in range(trials):
        optimizer = rng.choice(("adamw", "sgd")) if i else "adamw"
        rates = (0.0003, 0.001, 0.003) if optimizer == "adamw" else (0.003, 0.01, 0.03)
        parameters = {
            "optimizer": optimizer,
            "lr": rng.choice(rates) if i else 0.001,
            "decay": rng.choice((0.00001, 0.0001, 0.001)) if i else 0.0001,
            "dropout": rng.choice((0.1, 0.25, 0.4)) if i else 0.25,
            "augmentation": rng.choice(("none", "mild", "medium")) if i else "mild",
        }
        if family == "scratch":
            parameters.update(width=rng.choice((16, 24, 32)) if i else 24,
                              depth=rng.choice((3, 4)) if i else 3)
        else:
            parameters["unfreeze"] = rng.choice((["layer4"], ["layer3", "layer4"])) if i else ["layer4"]
        yield parameters


def search(family, loaders, config, device, weights_dir):
    best = None
    results = []
    for i, parameters in enumerate(candidates(family, config["seed"], config["search_trials"]), 1):
        fijar_semillas(config["seed"])
        print(f"{family} {i}/{config['search_trials']} {parameters}", flush=True)
        model = build(family, parameters, pretrained=True, weights_dir=weights_dir)
        model, score, history = fit(model, loaders["train"], loaders["val"], parameters, config, device)
        if best is None or score > best[2]:
            best = model, parameters, score
        results.append({"family": family, "trial": i, "parameters": parameters,
                        "val_f1": score, "history": history})
    return (*best, results)
