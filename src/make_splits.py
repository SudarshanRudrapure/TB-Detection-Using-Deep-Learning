# src/make_splits.py
import os, glob, pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

classes = ["Healthy", "Non-TB", "Tuberculosis"]
rows = []

for label, cls in enumerate(classes):
    pattern = os.path.join(DATA_DIR, cls, "*")
    for p in glob.glob(pattern):
        # Save as relative path from project root for compatibility
        rows.append([p.replace("\\", "/"), label, cls])

df = pd.DataFrame(rows, columns=["image_path", "label", "class_name"])
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
print(f"Saved train: {len(train_df)}, val: {len(val_df)} to {OUT_DIR}")
