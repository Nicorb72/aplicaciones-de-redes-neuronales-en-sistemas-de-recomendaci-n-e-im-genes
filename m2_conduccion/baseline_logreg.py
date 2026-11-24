"""Logística multinomial regularizada sobre características congeladas."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from m2_conduccion.transfer_learning import TransferCNN
from m2_conduccion.train import preprocess


@torch.inference_mode()
def extract(model, loader, device):
    features, targets = [], []
    for images, labels in loader:
        features.append(model.embeddings(model.spatial(preprocess(images.to(device)))).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(features), np.concatenate(targets)


def fit_baseline(loaders, device, weights_dir, seed):
    model = TransferCNN(0.0, [], pretrained=True, weights_dir=weights_dir)
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    train_x, train_y = extract(model, loaders["train"], device)
    val_x, val_y = extract(model, loaders["val"], device)
    scaler = StandardScaler().fit(train_x)
    train_x, val_x = scaler.transform(train_x), scaler.transform(val_x)
    results = []
    best = None
    for c in (0.01, 0.1, 1.0):
        estimator = LogisticRegression(C=c, max_iter=1000, random_state=seed).fit(train_x, train_y)
        score = f1_score(val_y, estimator.predict(val_x), average="macro")
        if best is None or score > best[0]:
            best = score, estimator
        results.append({"family": "logreg", "C": c, "val_f1": score})
    score, estimator = best
    weight = estimator.coef_ / scaler.scale_
    bias = estimator.intercept_ - weight @ scaler.mean_
    linear = model.classifier[-1]
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(weight))
        linear.bias.copy_(torch.as_tensor(bias))
    return model, {"dropout": 0.0, "unfreeze": []}, score, results
