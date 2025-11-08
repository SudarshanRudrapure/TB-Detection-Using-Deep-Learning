# src/data_loader.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class TBXDataset(Dataset):
    def __init__(self, dataframe, root_dir=None, transforms=None):
        self.df = dataframe.reset_index(drop=True)
        # 🔹 Set correct root directory for images
        self.root_dir = root_dir or r"C:\Users\sangm\OneDrive\Desktop\tbx11k\images"
        self.transforms = transforms

        # ✅ Correct label mapping for 3 classes
        self.label_map = {"healthy": 0, "non-tb": 1, "tb": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["fname"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        label = self.label_map[row["target"]]
        return img, label


def get_transforms(split="train", size=224):
    if split == "train":
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
