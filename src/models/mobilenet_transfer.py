import torch.nn as nn
from torchvision import models


class MobileNetTransfer(nn.Module):
    def __init__(self, num_classes=10, fine_tune=True):
        super().__init__()

        weights = models.MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=weights)

        for param in self.model.features.parameters():
            param.requires_grad = False

        if fine_tune:
            for name, param in self.model.features.named_parameters():
                if "17" in name or "18" in name:
                    param.requires_grad = True

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)