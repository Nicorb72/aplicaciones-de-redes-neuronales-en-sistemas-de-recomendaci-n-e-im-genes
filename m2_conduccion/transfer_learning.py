"""ResNet-18 como extractor congelado o red con ajuste parcial."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

WEIGHTS_NAME = "resnet18-f37072fd.pth"


class TransferCNN(nn.Module):
    def __init__(self, dropout, unfreeze, pretrained=False, weights_dir=None):
        super().__init__()
        local = Path(weights_dir) / WEIGHTS_NAME if weights_dir else None
        use_local = pretrained and local is not None and local.is_file()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained and not use_local else None
        self.backbone = resnet18(weights=weights)
        if use_local:
            self.backbone.load_state_dict(torch.load(local, map_location="cpu", weights_only=True))
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        for stage in unfreeze:
            getattr(self.backbone, stage).requires_grad_(True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 5))

    def train(self, mode=True):
        super().train(mode)
        # Conserva las estadísticas ImageNet incluso al ajustar bloques.
        self.backbone.eval()
        return self

    def spatial(self, images):
        model = self.backbone
        x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        return model.layer4(model.layer3(model.layer2(model.layer1(x))))

    def embeddings(self, features):
        return features.mean(dim=(2, 3))

    def logits(self, features):
        return self.classifier(self.embeddings(features))

    def forward(self, images):
        return self.logits(self.spatial(images))
