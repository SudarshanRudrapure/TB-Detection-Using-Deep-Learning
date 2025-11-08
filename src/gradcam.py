# src/gradcam.py
import torch
import cv2
import numpy as np

class GradCAM:
    """
    Grad-CAM for a model. Provide the actual layer object as target_layer.
    Example: target_layer = model.cnn.layer4[-1].conv2
    """
    def __init__(self, model, target_layer):   # ✅ fixed
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        """
        input_tensor: torch tensor shape (1, C, H, W)
        returns: heatmap numpy (H, W) normalized 0..1
        """
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward(retain_graph=False)

        grads = self.gradients  # (B, C, H, W)
        acts = self.activations  # (B, C, H, W)
        weights = grads.mean(dim=(2,3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1)  # (B, H, W)
        cam = torch.relu(cam)[0].cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam