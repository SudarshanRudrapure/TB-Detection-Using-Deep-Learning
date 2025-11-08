import os, pandas as pd
from sklearn.model_selection import train_test_split

root = "data/raw"
classes = ["Healthy", "Non-TB", "Tuberculosis"]
records = []

for label, cls in enumerate(classes):
    folder = os.path.join(root, cls)
    if not os.path.exists(folder):
        print(f"⚠️ Missing folder: {folder}")
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            records.append({"path": f"{cls}/{fname}", "label": label})

df = pd.DataFrame(records)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)

print("✅ Train/val CSVs created at data/processed/")
