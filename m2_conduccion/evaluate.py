"""Evaluación final, ejemplos de prueba y comparación de la política de alerta."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch
from time import perf_counter

from m2_conduccion.alert_policy import choose_threshold, temperature, threshold_table
from m2_conduccion.data.explore import CLASSES, DISPLAY
from m2_conduccion.eda import TABLES, plt, save_figure
from m2_conduccion.gradcam import explain
from m2_conduccion.train import infer, preprocess


def examples(frame, cache, probabilities, predictions, correct, seed, model, device):
    sample = frame[frame["correct"] == correct].sample(frac=1, random_state=seed)
    sample = sample.drop_duplicates("group")
    # Prioriza grupos distintos y cubre ambos formatos dentro de cada clase.
    sample["dominant_resolution"] = (sample["width"] == 640) & (sample["height"] == 480)
    sample["order"] = sample.groupby(["label", "dominant_resolution"]).cumcount()
    chosen = sample.sort_values(["order", "label", "dominant_resolution"]).head(10)
    if len(chosen) < 10:
        others = frame[frame["correct"] == correct].drop(index=chosen.index)
        chosen = pd.concat([chosen, others.head(10 - len(chosen))])
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    cam_fig, cam_axes = plt.subplots(2, 5, figsize=(12, 6))
    for ax, cam_ax in zip(axes.flat, cam_axes.flat):
        ax.axis("off")
        cam_ax.axis("off")
    for ax, cam_ax, (row, entry) in zip(axes.flat, cam_axes.flat, chosen.iterrows()):
        image = cache[row]
        position = int(entry["position"])
        predicted = predictions[position]
        title = f"{DISPLAY[entry['label']]} → {DISPLAY[predicted]}\nP = {probabilities[position, predicted]:.2f}"
        inputs = preprocess(torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(device))
        cam = explain(model, inputs, int(predicted))
        for panel in (ax, cam_ax):
            panel.imshow(image)
            panel.set_title(title, fontsize=9)
        cam_ax.imshow(cam, cmap="inferno", alpha=0.58 * cam, vmin=0, vmax=1)
    name = "aciertos" if correct else "errores"
    save_figure(fig, name)
    save_figure(cam_fig, f"gradcam_{name}")
    return chosen.assign(example=np.arange(1, len(chosen)+1))


def evaluate(models, winner, loaders, frame, cache_path, config, device):
    labels = np.arange(len(CLASSES))
    summaries, details = [], []
    for family, (model, _, val_f1) in models.items():
        logits, y = infer(model, loaders["test"], device)
        predicted = logits.argmax(dim=1).numpy()
        pr, re, f1, support = precision_recall_fscore_support(
            y, predicted, labels=labels, zero_division=0)
        details.extend((family, name, p, r, f, n)
                       for name, p, r, f, n in zip(CLASSES, pr, re, f1, support))
        summaries.append((family, val_f1, accuracy_score(y, predicted),
                          f1_score(y, predicted, labels=labels, average="macro", zero_division=0)))
        cm = confusion_matrix(y, predicted, labels=labels)
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(TABLES / f"m2_confusion_{family}.csv")
        if family == winner:
            selected = model, logits, y, predicted, cm
    pd.DataFrame(summaries, columns=["family", "val_f1", "accuracy", "macro_f1"]).to_csv(
        TABLES / "m2_modelos.csv", index=False)
    pd.DataFrame(details, columns=["family", "class", "precision", "recall", "f1", "support"]).to_csv(
        TABLES / "m2_metricas_clase.csv", index=False)
    model, logits, y, predicted, cm = selected
    val_logits, val_labels = infer(model, loaders["val"], device)
    temp = temperature(val_logits, val_labels)
    val_alert = threshold_table(val_logits, val_labels, temp, config)
    threshold = choose_threshold(val_alert)
    test_alert = threshold_table(logits, y, temp, config, threshold)
    for name, table in (("val", val_alert), ("test", test_alert)):
        table.assign(selected=np.isclose(table["threshold"], threshold)).to_csv(
            TABLES / f"m2_alertas_{name}.csv", index=False)
    probabilities = softmax(logits.numpy() / temp, axis=1)
    test = frame[frame["split"] == "test"].copy()
    test["position"] = np.arange(len(test))
    test["prediction"] = predicted
    test["correct"] = y == predicted
    test = test.join(pd.DataFrame(probabilities, index=test.index, columns=[f"p_{c}" for c in CLASSES]))
    cache = np.load(cache_path, mmap_mode="r")
    selected_examples = [examples(test, cache, probabilities, predicted, state, config["seed"], model, device)
                         for state in (True, False)]
    pd.concat(selected_examples).to_csv(TABLES / "m2_ejemplos.csv", index=False)
    test.to_csv(TABLES / "m2_predicciones.csv", index=False)
    plot_results(cm, test_alert, threshold)
    if device.type == "cpu":
        sample = torch.from_numpy(cache[test.index[0]].copy()).permute(2, 0, 1)[None]
        with torch.inference_mode():
            for _ in range(8):
                model(preprocess(sample))
            times = []
            for _ in range(50):
                start = perf_counter()
                model(preprocess(sample)).softmax(dim=1)
                times.append(1000 * (perf_counter() - start))
        pd.DataFrame([{"family": winner, "device": "cpu", "threads": torch.get_num_threads(),
                       "p50_ms": np.quantile(times, 0.50), "p95_ms": np.quantile(times, 0.95)}]).to_csv(
            TABLES / "m2_latencia.csv", index=False)
    return threshold, temp


def plot_results(cm, test_alert, threshold):
    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.imshow(cm, cmap="Blues")
    fig.colorbar(heat, ax=ax, label="Imágenes")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=DISPLAY, yticklabels=DISPLAY,
           xlabel="Predicción", ylabel="Clase real")
    for row, col in np.ndindex(cm.shape):
        ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                color="white" if cm[row, col] > cm.max()/2 else "black")
    save_figure(fig, "confusion")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(test_alert["threshold"], test_alert["fp_per_1000"], label="Falsas alarmas", color="#29557a")
    ax.plot(test_alert["threshold"], test_alert["fn_per_1000"], label="Distracciones omitidas", color="#c96328")
    ax.axvline(threshold, color="#555555", linestyle="--", label=f"Elegido en validación ({threshold:.2f})")
    ax.set(xlabel="Umbral de alerta", ylabel="Errores por 1.000 imágenes de prueba")
    ax.legend()
    save_figure(fig, "alertas")


def regenerate_figures(frame, cache_path, seed):
    """Reconstruye figuras a partir de los resultados y el checkpoint publicados."""
    from m2_conduccion.eda import PROCESSED, plot_inventory, show_matches
    from m2_conduccion.predict import load_checkpoint
    from m2_conduccion.stability import plot_stability

    model, metadata = load_checkpoint()
    cache = np.load(cache_path, mmap_mode="r")
    plot_inventory(frame, cache)
    show_matches(frame, cache, pd.read_csv(PROCESSED / "coincidencias.csv"), seed)
    predictions = pd.read_csv(TABLES / "m2_predicciones.csv")
    test = frame[frame["split"] == "test"].copy()
    if not np.array_equal(test["image"], predictions["image"]):
        raise ValueError("Las predicciones guardadas corresponden a otra partición.")
    test["position"] = predictions["position"].to_numpy(dtype=int)
    test["correct"] = predictions["correct"].to_numpy(dtype=bool)
    probabilities = predictions[[f"p_{label}" for label in CLASSES]].to_numpy()
    predicted = predictions["prediction"].to_numpy()
    for correct in (True, False):
        examples(test, cache, probabilities, predicted, correct, seed, model, torch.device("cpu"))
    matrix = pd.read_csv(TABLES / f"m2_confusion_{metadata['family']}.csv", index_col=0).to_numpy()
    alert_table = pd.read_csv(TABLES / "m2_alertas_test.csv")
    plot_results(matrix, alert_table, metadata["threshold"])
    plot_stability(pd.read_csv(TABLES / "m2_semillas.csv"))
