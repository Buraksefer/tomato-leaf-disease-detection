import torch.nn as nn
from torchvision import models


class DenseNetTransfer(nn.Module):
    def __init__(self, num_classes=10, fine_tune=True):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT
        self.model = models.densenet121(weights=weights)

        for param in self.model.features.parameters():
            param.requires_grad = False

        if fine_tune:
            for name, param in self.model.features.named_parameters():
                if "denseblock4" in name or "norm5" in name:
                    param.requires_grad = True

        in_features = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)