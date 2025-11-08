from src.model import HybridDeiTResNet
import torch

m = HybridDeiTResNet(num_classes=3)
x = torch.randn(2, 3, 224, 224)
y = m(x)
print('Output shape:', y.shape)