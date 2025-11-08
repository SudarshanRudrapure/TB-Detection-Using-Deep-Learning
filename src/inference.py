# src/inference.py
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from src.model import HybridDeiTResNet
from src.gradcam import GradCAM
from src.data_loader import get_transforms

# Match your dataset folder names
CLASS_NAMES = ["healthy", "no_tb", "tb"]
CHECKPOINT = "checkpoints/best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = HybridDeiTResNet(num_classes=len(CLASS_NAMES))
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict_and_cam(image_pil, model=None):
    """
    image_pil: PIL.Image
    returns: (pred_name, probs_array, cam_overlay_path)
    """
    if model is None:
        model = load_model()

    # ✅ use torchvision transforms directly
    transform = get_transforms("val")
    inp = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]

    # ✅ Grad-CAM
    target_layer = model.cnn.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(inp, class_idx=pred_idx)

    # Convert to numpy + overlay heatmap
    img_arr = np.array(image_pil.convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_cv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"cam_{pred_name}.png")
    cv2.imwrite(out_path, overlay)
    return pred_name, probs, out_path


if __name__ == "__main__":
    # Allow user to pass an image path from command line
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Default test image
        test_image = "data/raw/healthy/healthy1.jpg"

    if not os.path.exists(test_image):
        print(f"❌ File not found: {test_image}")
        sys.exit(1)

    img = Image.open(test_image).convert("RGB")
    model = load_model()
    pred, probs, cam_path = predict_and_cam(img, model=model)

    print("✅ Prediction:", pred)
    print("✅ Probabilities:", probs)
    print("✅ Saved Grad-CAM overlay at:", cam_path)
