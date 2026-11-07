
import requests
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET = "yaful/MAGE"
BASE_URL = "https://datasets-server.huggingface.co"
CONFIG = "default"

session = requests.Session()


def get_rows(split, offset=0, length=100):
    response = session.get(
        f"{BASE_URL}/rows",
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": split,
            "offset": offset,
            "length": length,
        },
        timeout=60,
    )
    response.raise_for_status()

    return [
        item["row"]
        for item in response.json().get("rows", [])
    ]


# 1. Discover available splits
response = session.get(
    f"{BASE_URL}/splits",
    params={"dataset": DATASET},
    timeout=30,
)
response.raise_for_status()

splits_df = pd.DataFrame(response.json()["splits"])
print("\nAVAILABLE SPLITS")
print(splits_df.to_string(index=False))
splits_df.to_csv(
    OUTPUT_DIR / "metadata_available_splits.csv",
    index=False,
)

# 2. Inspect the first 100 rows of every available split
frames = []

for split in splits_df["split"].dropna().unique():
    try:
        rows = get_rows(split, offset=0, length=100)

        if rows:
            part = pd.DataFrame(rows)
            part["split_name"] = split
            frames.append(part)

            print(f"\n{split}: retrieved {len(part)} rows")
            print("Columns:", part.columns.tolist())

    except requests.RequestException as error:
        print(f"Could not inspect {split}: {error}")

if not frames:
    raise RuntimeError("No rows retrieved. Check the API response.")

sample = pd.concat(frames, ignore_index=True)

required = {"text", "label", "src"}
if not required.issubset(sample.columns):
    raise ValueError(
        f"Expected {required}; found {sample.columns.tolist()}"
    )

# 3. Inspect labels and sources in the retrieved samples
label_counts = (
    sample.groupby(["split_name", "label"], dropna=False)
    .size()
    .reset_index(name="count")
)

source_counts = (
    sample.groupby(
        ["split_name", "src", "label"],
        dropna=False
    )
    .size()
    .reset_index(name="count")
)

label_counts.to_csv(
    OUTPUT_DIR / "metadata_label_counts.csv", index=False
)
source_counts.to_csv(
    OUTPUT_DIR / "metadata_source_by_label.csv", index=False
)

print("\nLABEL COUNTS BY SPLIT")
print(label_counts.to_string(index=False))

print("\nSOURCE AND LABEL COUNTS")
print(source_counts.to_string(index=False))

print("\nMISSING VALUES")
print(sample[["text", "label", "src"]].isna().sum())

# 4. Tentatively parse source names for further inspection
def parse_source(value):
    if pd.isna(value):
        return pd.Series({
            "source_type": "unknown",
            "domain_candidate": "unknown",
            "model_candidate": "unknown",
        })

    source = str(value).lower()
    parts = source.split("_")

    if "human" in parts:
        source_type = "human"
    elif "machine" in parts:
        source_type = "machine"
    else:
        source_type = "unknown"

    domain_candidate = parts[0] if parts else "unknown"

    # Candidate only: source names do not always encode
    # generation methods and models in a uniform format.
    model_candidate = "unknown"
    if "machine" in parts:
        machine_index = parts.index("machine")
        remaining = parts[machine_index + 1:]
        if remaining:
            model_candidate = "_".join(remaining)

    return pd.Series({
        "source_type": source_type,
        "domain_candidate": domain_candidate,
        "model_candidate": model_candidate,
    })


parsed = sample["src"].apply(parse_source)
sample = pd.concat([sample, parsed], axis=1)

sample.to_csv(
    OUTPUT_DIR / "metadata_split_samples.csv",
    index=False,
)

print("\nCANDIDATE DOMAIN COUNTS")
print(sample["domain_candidate"].value_counts().to_string())

print("\nCANDIDATE SOURCE-TYPE COUNTS")
print(sample["source_type"].value_counts().to_string())

print("\nMETADATA INSPECTION COMPLETE")
print("Outputs saved in:", OUTPUT_DIR.resolve())



# 5. Broader diagnostic scan of training positions
# Selected offsets are for exploration, not random sampling.

offsets = [
    0, 100, 500, 1000, 5000, 10000,
    20000, 50000, 100000, 150000,
    200000, 250000, 300000
]

diagnostic_records = []

for offset in offsets:
    try:
        rows = get_rows("train", offset=offset, length=100)

        for row in rows:
            record = row.copy()
            record["sampled_offset"] = offset
            diagnostic_records.append(record)

        print(f"Offset {offset}: retrieved {len(rows)} rows")

    except requests.RequestException as error:
        print(f"Offset {offset} failed: {error}")

diagnostic = pd.DataFrame(diagnostic_records)

if diagnostic.empty:
    raise RuntimeError("No diagnostic rows retrieved.")

print("\nDIAGNOSTIC LABEL COUNTS")
print(diagnostic["label"].value_counts(dropna=False))

print("\nSOURCE BY LABEL")
source_label_table = pd.crosstab(
    diagnostic["src"],
    diagnostic["label"],
    dropna=False
)
print(source_label_table.to_string())

print("\nLABEL COUNTS BY OFFSET")
print(pd.crosstab(
    diagnostic["sampled_offset"],
    diagnostic["label"]
).to_string())

diagnostic.to_csv(
    OUTPUT_DIR / "mage_broader_diagnostic_sample.csv",
    index=False
)

source_label_table.to_csv(
    OUTPUT_DIR / "mage_broader_source_by_label.csv"
)

print("\nSaved broader diagnostic outputs.")
