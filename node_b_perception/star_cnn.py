"""
Star CNN — Modified ResNet-18 for Attitude Estimation

Input  : 1-channel grayscale star image (224×224)
Output : 3 regression values (pitch, roll, yaw in degrees)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class StarCNN(nn.Module):
    """
    ResNet-18 adapted for star pattern → attitude regression.

    Modifications from standard ResNet-18:
      1. First conv layer accepts 1‑channel (grayscale) input
      2. Final FC layer outputs 3 values (pitch, roll, yaw)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load base ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        base = models.resnet18(weights=weights)

        # Modify input: 3‑channel RGB → 1‑channel grayscale
        # Average the pretrained weights across the 3 input channels
        old_conv = base.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        # Keep all intermediate layers
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Modify output: 1000 ImageNet classes → 3 attitude values
        in_features = base.fc.in_features   # 512
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),               # pitch, roll, yaw
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ─────────────────────── Quick test ─────────────────────── #
if __name__ == "__main__":
    model = StarCNN(pretrained=True)
    dummy = torch.randn(1, 1, 224, 224)
    out = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Output values: {out.detach().numpy()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")
