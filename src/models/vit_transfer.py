import timm
import torch.nn as nn


class ViTTransfer(nn.Module):
    def __init__(self, num_classes=10, fine_tune=True):
        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)