"""CNN inicializada desde cero."""
from torch import nn


class ScratchCNN(nn.Module):
    def __init__(self, width, depth, dropout):
        super().__init__()
        layers = []
        channels = 3
        for i in range(depth):
            output = width * 2**i
            stride = 2 if i == 0 else 1
            layers.extend((nn.Conv2d(channels, output, 3, stride=stride, padding=1, bias=False),
                           nn.BatchNorm2d(output), nn.ReLU(), nn.MaxPool2d(2)))
            channels = output
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
                                        nn.Dropout(dropout), nn.Linear(4 * channels, 5))

    def spatial(self, images):
        return self.features(images)

    def logits(self, features):
        return self.classifier(features)

    def forward(self, images):
        return self.logits(self.spatial(images))
