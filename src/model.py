# src/model.py
import torch
import torch.nn as nn
import timm

class HybridDeiTResNet(nn.Module):
    def __init__(self, num_classes=3, deit_name='deit_tiny_patch16_224', cnn_name='resnet18', pretrained=True):
        super().__init__()
        # DeiT (ViT-like) feature extractor, output: vector
        self.deit = timm.create_model(deit_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        # CNN backbone
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0, global_pool='avg')

        deit_features = self.deit.num_features
        cnn_features = self.cnn.num_features

        hidden = 512
        self.classifier = nn.Sequential(
            nn.Linear(deit_features + cnn_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # x: B x 3 x H x W
        f_deit = self.deit(x)   # shape: (B, deit_features)
        f_cnn  = self.cnn(x)    # shape: (B, cnn_features)
        f = torch.cat([f_deit, f_cnn], dim=1)
        out = self.classifier(f)
        return out

