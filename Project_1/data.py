import os
import xml.etree.ElementTree as ET
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(script_dir, "reports")

if not os.path.isdir(reports_dir):
    raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

records = []
for filename in os.listdir(reports_dir):
    if filename.endswith(".xml"):
        tree = ET.parse(os.path.join(reports_dir, filename))
        root = tree.getroot()
        findings, impression = None, None
        for section in root.iter("AbstractText"):
            label = section.attrib.get("Label", "")
            if label == "FINDINGS":
                findings = section.text
            elif label == "IMPRESSION":
                impression = section.text
        if findings and impression:
            records.append({"findings": findings, "impression": impression})

output_csv = os.path.join(script_dir, "radiology_data.csv")
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(df.shape)
df = pd.read_csv("radiology_data.csv")

df.dropna(subset=["findings", "impression"], inplace=True)

df["findings"] = df["findings"].str.strip()
df["impression"] = df["impression"].str.strip()

df = df[df["findings"].str.len() > 20]
df = df[df["impression"].str.len() > 5]

df.reset_index(drop=True, inplace=True)
print(df.shape)
