# src/eval.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import TBXDataset, get_transforms
from model import HybridDeiTResNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")
    root_dir = r"C:\Users\sangm\OneDrive\Desktop\tbx11k\images"

    ds = TBXDataset(test_df, root_dir=root_dir, transforms=get_transforms("val", size=224))
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    # Load model
    model = HybridDeiTResNet(num_classes=3)
    model.load_state_dict(torch.load("checkpoints/best.pth", map_location=device))
    model.to(device).eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Classification Report
    class_names = ["Healthy", "Non-TB", "TB"]
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("✅ Confusion matrix saved at outputs/confusion_matrix.png")

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\nOverall Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")


if __name__ == "__main__":
    main()
