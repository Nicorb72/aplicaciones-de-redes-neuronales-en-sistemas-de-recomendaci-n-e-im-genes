"""Lectura de lotes, preprocesamiento compartido y ajuste por validación."""
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from m2_conduccion.cnn_scratch import ScratchCNN
from m2_conduccion.transfer_learning import TransferCNN

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Images(Dataset):
    def __init__(self, path, frame):
        self.cache = np.load(path, mmap_mode="r")
        self.rows = frame.index.to_numpy()
        self.labels = frame["label"].to_numpy()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pixels = self.cache[self.rows[i]].copy()
        return torch.from_numpy(pixels).permute(2, 0, 1), int(self.labels[i])


def make_loaders(frame, path, config):
    return {split: DataLoader(Images(path, frame[frame["split"] == split]),
                              batch_size=config["batch_size"], shuffle=False)
            for split in ("train", "val", "test")}


def preprocess(images):
    images = images.to(dtype=torch.float32, memory_format=torch.channels_last).div_(255)
    mean = images.new_tensor(MEAN)[None, :, None, None]
    std = images.new_tensor(STD)[None, :, None, None]
    return (images - mean) / std


def augmentation(level):
    if level == "none":
        return nn.Identity()
    angle, brightness = (5, 0.12) if level == "mild" else (12, 0.25)
    return v2.Compose([v2.RandomAffine(degrees=angle, translate=(0.04, 0.04),
                                       scale=(0.92, 1.08),
                                       interpolation=v2.InterpolationMode.BILINEAR),
                       v2.ColorJitter(brightness=brightness, contrast=brightness)])


def build(family, parameters, pretrained=False, weights_dir=None):
    if family == "scratch":
        return ScratchCNN(parameters["width"], parameters["depth"], parameters["dropout"])
    if family in {"logreg", "transfer"}:
        return TransferCNN(parameters["dropout"], parameters["unfreeze"], pretrained, weights_dir)
    raise ValueError(f"Familia desconocida: {family}")


@torch.inference_mode()
def infer(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for images, target in loader:
        predictions.append(model(preprocess(images.to(device))).cpu())
        labels.append(target)
    return torch.cat(predictions), torch.cat(labels).numpy()


def fit(model, train, val, parameters, config, device):
    model.to(device=device, memory_format=torch.channels_last)
    extractor = model.backbone.parameters() if isinstance(model, TransferCNN) else []
    backbone = [p for p in extractor if p.requires_grad]
    backbone_ids = {id(p) for p in backbone}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups = [{"params": head}]
    if backbone:
        groups.append({"params": backbone, "lr": parameters["lr"] * 0.2})
    optimizer_cls = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[parameters["optimizer"]]
    options = {"momentum": 0.9} if parameters["optimizer"] == "sgd" else {}
    optimizer = optimizer_cls(groups, lr=parameters["lr"], weight_decay=parameters["decay"], **options)
    augment = augmentation(parameters["augmentation"])
    criterion = nn.CrossEntropyLoss()
    shuffled = DataLoader(train.dataset, batch_size=config["batch_size"], shuffle=True,
                          generator=torch.Generator().manual_seed(config["seed"]))
    history, best_state, best_score = [], None, -1.0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in shuffled:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            # v2 comparte los parámetros aleatorios dentro de cada lote.
            loss = criterion(model(preprocess(augment(images))), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        logits, y = infer(model, val, device)
        score = f1_score(y, logits.argmax(dim=1), average="macro")
        history.append({"epoch": epoch, "loss": loss_sum / len(shuffled.dataset), "val_f1": score})
        print(f"Época {epoch}: F1 macro validación {score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_score, history
