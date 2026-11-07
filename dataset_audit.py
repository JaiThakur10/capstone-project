
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Save research outputs in a dedicated folder
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET = "yaful/MAGE"
BASE_URL = "https://datasets-server.huggingface.co"

# 1. Check dataset splits
response = requests.get(
    f"{BASE_URL}/splits",
    params={"dataset": DATASET},
    timeout=30
)
response.raise_for_status()

splits = pd.DataFrame(response.json()["splits"])
print("AVAILABLE DATASET SPLITS")
print(splits.to_string(index=False))
splits.to_csv(OUTPUT_DIR / "available_splits.csv", index=False)

# 2. Retrieve up to 1,000 training examples
response = requests.get(
    f"{BASE_URL}/rows",
    params={
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 100
    },
    timeout=60
)
response.raise_for_status()

df = pd.DataFrame(
    item["row"] for item in response.json()["rows"]
)

if df.empty:
    raise RuntimeError("No dataset rows were returned.")

required = {"text", "label", "src"}
if not required.issubset(df.columns):
    raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

print("\nROWS RETRIEVED:", len(df))
print("COLUMNS:", df.columns.tolist())

# 3. Inspect labels and sources
label_counts = df["label"].value_counts(dropna=False)
source_counts = df["src"].value_counts(dropna=False)

print("\nLABEL COUNTS")
print(label_counts.to_string())

print("\nSOURCE COUNTS")
print(source_counts.to_string())

print("\nMISSING VALUES")
print(df[["text", "label", "src"]].isna().sum().to_string())

label_counts.rename_axis("label").reset_index(
    name="count"
).to_csv(OUTPUT_DIR / "label_counts_sample.csv", index=False)

source_counts.rename_axis("src").reset_index(
    name="count"
).to_csv(OUTPUT_DIR / "source_counts_sample.csv", index=False)

# 4. Analyze text lengths
df["text_length_chars"] = df["text"].fillna("").str.len()
df["text_length_words"] = (
    df["text"].fillna("").str.split().str.len()
)

print("\nTEXT LENGTH SUMMARY")
print(
    df[["text_length_chars", "text_length_words"]]
    .describe()
    .round(2)
    .to_string()
)

df.to_csv(OUTPUT_DIR / "mage_train_sample.csv", index=False)

# 5. Generate a preliminary chart
label_counts.sort_index().plot(kind="bar")
plt.title("MAGE Training Sample: Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Examples")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "label_distribution.png", dpi=200)
plt.close()

print("\nAUDIT FINISHED")
print("Outputs saved in:", OUTPUT_DIR.resolve())


# 6. Inspect different positions in the training split
# This is a diagnostic sample, NOT a random sample.

offsets = [0, 1000, 10000, 50000, 100000, 200000, 300000]
sampled_records = []

for offset in offsets:
    response = requests.get(
        f"{BASE_URL}/rows",
        params={
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": 100
        },
        timeout=60
    )

    if response.status_code != 200:
        print(f"Offset {offset} failed:", response.text[:300])
        continue

    rows = response.json().get("rows", [])

    for item in rows:
        record = item["row"].copy()
        record["sampled_offset"] = offset
        sampled_records.append(record)

sampled_df = pd.DataFrame(sampled_records)

if not sampled_df.empty:
    print("\nLABEL COUNTS ACROSS INSPECTED OFFSETS")
    print(sampled_df["label"].value_counts(dropna=False))

    print("\nSOURCE COUNTS ACROSS INSPECTED OFFSETS")
    print(sampled_df["src"].value_counts(dropna=False).head(30))

    print("\nLABEL COUNTS BY OFFSET")
    print(pd.crosstab(
        sampled_df["sampled_offset"],
        sampled_df["label"]
    ))

    sampled_df.to_csv(
        OUTPUT_DIR / "mage_offset_diagnostic_sample.csv",
        index=False
    )

    print("\nSaved offset diagnostic sample.")

