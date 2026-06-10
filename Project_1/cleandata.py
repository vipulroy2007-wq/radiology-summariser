import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "radiology_data.csv")
df = pd.read_csv(csv_path)

df.dropna(subset=["findings", "impression"], inplace=True)

df["findings"] = df["findings"].str.strip()
df["impression"] = df["impression"].str.strip()

df = df[df["findings"].str.len() > 20]
df = df[df["impression"].str.len() > 5]

df.reset_index(drop=True, inplace=True)
print(df.shape)