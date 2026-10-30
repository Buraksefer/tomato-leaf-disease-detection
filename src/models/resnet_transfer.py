import torch.nn as nn
from torchvision import models


class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=10, fine_tune=True):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune:
            for name, param in self.model.named_parameters():
                if name.startswith("layer4") or name.startswith("fc"):
                    param.requires_grad = True

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)