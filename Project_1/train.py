import os
import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "radiology_data.csv")

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"Data file not found: {csv_path}")

df = pd.read_csv(csv_path)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv(os.path.join(script_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(script_dir, "test.csv"), index=False)
