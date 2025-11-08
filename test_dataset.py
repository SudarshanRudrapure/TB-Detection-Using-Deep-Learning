from src.data_loader import TBXDataset, get_transforms
import pandas as pd

df = pd.read_csv("data/raw/labels.csv")
dataset = TBXDataset(df, "data/raw/images", transforms=get_transforms("train"))
img, label = dataset[0]
print(img.shape, label)
