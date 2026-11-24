"""Grad-CAM sobre los mapas espaciales que alimentan el clasificador."""
import torch
from torch.nn import functional as F


@torch.enable_grad()
def explain(model, inputs, label=None):
    model.eval()
    features = model.spatial(inputs.detach().requires_grad_(True))
    logits = model.logits(features)
    if label is None:
        label = int(logits[0].argmax())
    gradients, = torch.autograd.grad(logits[0, label], features)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / cam.max().clamp_min(1e-12)).detach().cpu().numpy()
