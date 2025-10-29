import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True, freeze_backbone=False):
        super().__init__()
        # Load ResNet18 backbone
        backbone = models.resnet18(pretrained=pretrained)
        # remove final fc
        modules = list(backbone.children())[:-1]  # remove the last fc
        self.backbone = nn.Sequential(*modules)  # outputs [B,512,1,1]
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward_one(self, x):
        x = self.backbone(x)           # [B,512,1,1]
        x = self.fc(x)                 # [B, embedding_dim]
        x = F.normalize(x, p=2, dim=1) # normalize embedding (important)
        return x

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)