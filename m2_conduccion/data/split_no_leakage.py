"""Partición reproducible de grupos visuales completos."""
import numpy as np
import pandas as pd


def assign_groups(frame, ratios, seed):
    counts = pd.crosstab(frame["group"], frame["label"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    values = counts.to_numpy()
    order = sorted(order, key=lambda i: -values[i].sum())
    targets = np.outer(ratios, values.sum(axis=0))
    totals = np.zeros_like(targets)
    mapping = {}
    names = ("train", "val", "test")
    for i in order:
        per_class = values[i]
        delta = ((totals + per_class - targets)**2 - (totals - targets)**2) / targets
        p = int(delta.sum(axis=1).argmin())
        totals[p] += per_class
        mapping[counts.index[i]] = names[p]
    splits = frame["group"].map(mapping)
    if (pd.crosstab(splits, frame["label"]).reindex(names, fill_value=0) == 0).any().any():
        raise ValueError("La partición agrupada dejó una clase ausente en alguna partición.")
    return splits
