
from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET = "yaful/MAGE"

print("Loading MAGE training split...")
dataset = load_dataset(DATASET, split="train")

print("\nDATASET SIZE")
print(f"Number of training records: {len(dataset):,}")
print(f"Available columns: {dataset.column_names}")

# Convert only the label and source columns to a table.
# This avoids copying all long text into a Pandas DataFrame.
metadata = dataset.select_columns(["label", "src"]).to_pandas()

print("\nLABEL COUNTS")
print(metadata["label"].value_counts(dropna=False).sort_index())

print("\nNUMBER OF DISTINCT SOURCES")
print(metadata["src"].nunique(dropna=False))

# Count records for each source-label combination.
source_label_counts = pd.crosstab(
    metadata["src"],
    metadata["label"],
    dropna=False
)

source_label_counts.to_csv(
    OUTPUT_DIR / "full_train_source_by_label.csv"
)

# Summarize each source.
source_summary = source_label_counts.copy()
source_summary["total_records"] = source_label_counts.sum(axis=1)

source_summary.to_csv(
    OUTPUT_DIR / "full_train_source_summary.csv"
)

print("\nTOP 20 SOURCES BY RECORD COUNT")
print(
    source_summary.sort_values(
        "total_records", ascending=False
    ).head(20).to_string()
)

# Count sources containing both labels.
label_columns = [
    column for column in [0, 1]
    if column in source_label_counts.columns
]

if len(label_columns) == 2:
    both_labels = (
        (source_label_counts[0] > 0)
        & (source_label_counts[1] > 0)
    )

    print("\nSOURCE-LABEL OVERLAP")
    print(
        "Sources containing both labels:",
        int(both_labels.sum())
    )
    print(
        "Sources containing only one label:",
        int((~both_labels).sum())
    )

print("\nMISSING METADATA VALUES")
print(metadata[["label", "src"]].isna().sum())

print("\nFinished. Results saved in:", OUTPUT_DIR.resolve())
