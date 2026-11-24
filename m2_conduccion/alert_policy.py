"""Agrupación de probabilidades y selección de umbral por costo."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from m2_conduccion.data.explore import CLASSES


def risk_indices(grouping):
    risk = set(grouping["con_alerta"])
    safe = set(grouping["sin_alerta"])
    if risk & safe or risk | safe != set(CLASSES) or not risk or not safe:
        raise ValueError("La agrupación debe cubrir las cinco clases con dos grupos disjuntos.")
    return [i for i, label in enumerate(CLASSES) if label in risk]


def risk_score(probabilities, indices):
    return np.asarray(probabilities)[..., indices].sum(axis=-1)


def temperature(logits, labels):
    def loss(log_t):
        scaled = np.asarray(logits, dtype=np.float64) / np.exp(log_t)
        return (logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]).mean()
    result = minimize_scalar(loss, bounds=(-2.3, 2.3), method="bounded")
    return float(np.exp(result.x))


def threshold_table(logits, labels, temp, config, chosen=None):
    risky = risk_indices(config["class_grouping"])
    truth = np.isin(labels, risky)
    score = risk_score(softmax(np.asarray(logits, dtype=np.float64) / temp, axis=1), risky)
    grid = set(np.linspace(0.05, 0.95, 19).round(2)) | {config["default_threshold"]}
    if chosen is not None:
        grid.add(chosen)
    rows = []
    for threshold in sorted(grid):
        alert = score >= threshold
        fp = int((~truth & alert).sum())
        fn = int((truth & ~alert).sum())
        factor = 1000 / len(truth)
        cost = config["false_alarm_cost"] * fp + config["miss_cost"] * fn
        rows.append((threshold, fp, fn, fp * factor, fn * factor, cost * factor))
    return pd.DataFrame(rows, columns=["threshold", "fp", "fn", "fp_per_1000",
                                      "fn_per_1000", "cost_per_1000"])


def choose_threshold(table):
    return float(table.sort_values(["cost_per_1000", "fn", "fp", "threshold"]).iloc[0]["threshold"])
