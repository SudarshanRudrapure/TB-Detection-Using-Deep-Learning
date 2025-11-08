import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Load CSV ---
csv_path = r"C:\Users\sangm\OneDrive\Desktop\tbx11k\data.xlsx.csv"
print("📥 Reading dataset...")
df = pd.read_csv(csv_path)

print("\nPreview of raw data:")
print(df.head())

# --- Step 2: Create 3-class label mapping ---
def map_label(row):
    if row["target"] == "tb":
        return "tb"  # TB case
    elif row["image_type"].lower() == "healthy":
        return "healthy"  # Healthy case
    else:
        return "non-tb"  # Other non-TB lung diseases

df["final_label"] = df.apply(map_label, axis=1)

# --- Step 3: Keep only necessary columns ---
df = df[["fname", "final_label"]].copy()
df.rename(columns={"final_label": "target"}, inplace=True)

print("\nUnique labels after mapping:", df["target"].unique())
print("\nClass distribution before splitting:")
print(df["target"].value_counts())

# --- Step 4: Train/Val/Test Split ---
train_df, test_val_df = train_test_split(
    df, test_size=0.3, stratify=df["target"], random_state=42
)
val_df, test_df = train_test_split(
    test_val_df, test_size=0.5, stratify=test_val_df["target"], random_state=42
)

# --- Step 5: Save splits ---
output_dir = r"C:\Users\sangm\OneDrive\Desktop\Major One\data\processed"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

# --- Step 6: Summary ---
print("\n✅ Splits saved successfully to:", output_dir)
print("Train:", len(train_df), "| Val:", len(val_df), "| Test:", len(test_df))

print("\nTrain class counts:\n", train_df["target"].value_counts())
print("\nValidation class counts:\n", val_df["target"].value_counts())
print("\nTest class counts:\n", test_df["target"].value_counts()) 