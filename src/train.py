# src/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_loader import TBXDataset, get_transforms
from model import HybridDeiTResNet


def train_epoch(model, loader, opt, criterion, device):
    model.train()
    losses = []
    preds, trues = [], []
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds += out.argmax(1).detach().cpu().tolist()
        trues += labels.detach().cpu().tolist()
    return sum(losses) / len(losses), preds, trues


def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            losses.append(loss.item())
            preds += out.argmax(1).cpu().tolist()
            trues += labels.cpu().tolist()
    return sum(losses) / len(losses), preds, trues


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)

    # Load dataset splits
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # Path to your images
    root_dir = r"C:\Users\sangm\OneDrive\Desktop\tbx11k\images"

    # Create datasets and loaders
    batch_size = 16 if device == "cpu" else 32
    train_ds = TBXDataset(train_df, root_dir=root_dir, transforms=get_transforms("train", size=224))
    val_ds = TBXDataset(val_df, root_dir=root_dir, transforms=get_transforms("val", size=224))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model (3 classes)
    model = HybridDeiTResNet(num_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs = 30

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        train_loss, train_preds, train_trues = train_epoch(model, train_loader, opt, criterion, device)
        val_loss, val_preds, val_trues = eval_epoch(model, val_loader, criterion, device)

        val_acc = accuracy_score(val_trues, val_preds)
        val_prec = precision_score(val_trues, val_preds, average="macro", zero_division=0)
        val_rec = recall_score(val_trues, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_trues, val_preds, average="macro", zero_division=0)

        print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
        print(f"Precision={val_prec:.4f}, Recall={val_rec:.4f}, F1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best.pth")
            print("✅ Best model saved!")

    torch.save(model.state_dict(), "checkpoints/last.pth")
    print("\n🎯 Training complete! Best validation accuracy:", best_acc)


if __name__ == "__main__":
    main()
