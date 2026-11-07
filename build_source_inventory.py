
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")

source_file = OUTPUT_DIR / "full_train_source_by_label.csv"

# The CSV was saved with src as the index.
source_counts = pd.read_csv(source_file, index_col=0)
source_counts.index.name = "src"
source_counts = source_counts.reset_index()

# Calculate total records for each source.
label_columns = [
    column for column in source_counts.columns
    if str(column) in {"0", "1"}
]

source_counts["total_records"] = (
    source_counts[label_columns].sum(axis=1)
)

def parse_source(src):
    parts = str(src).lower().split("_")

    # First token is a candidate domain, not verified metadata.
    domain = parts[0] if parts else "unknown"

    if "human" in parts:
        source_type = "human"
        generator_candidate = "human"
    elif "machine" in parts:
        source_type = "machine"
        machine_index = parts.index("machine")
        generator_candidate = "_".join(
            parts[machine_index + 1:]
        ) or "unknown"
    else:
        source_type = "unknown"
        generator_candidate = "unknown"

    return pd.Series({
        "domain_candidate": domain,
        "source_type": source_type,
        "generator_candidate": generator_candidate
    })

parsed = source_counts["src"].apply(parse_source)
inventory = pd.concat([source_counts, parsed], axis=1)

inventory.to_csv(
    OUTPUT_DIR / "full_source_inventory.csv",
    index=False
)

# Summarize candidate domains by label.
domain_summary = inventory.groupby(
    ["domain_candidate", "source_type"],
    dropna=False
).agg(
    source_categories=("src", "nunique"),
    records=("total_records", "sum")
).reset_index()

domain_summary.to_csv(
    OUTPUT_DIR / "domain_inventory.csv",
    index=False
)

# Summarize machine-source candidates.
machine_inventory = inventory[
    inventory["source_type"] == "machine"
].copy()

machine_inventory.to_csv(
    OUTPUT_DIR / "machine_source_inventory.csv",
    index=False
)

print("\nCANDIDATE DOMAINS AND SOURCE TYPES")
print(domain_summary.to_string(index=False))

print("\nTOP 30 MACHINE SOURCES")
print(
    machine_inventory.sort_values(
        "total_records", ascending=False
    ).head(30).to_string(index=False)
)

print("\nFILES SAVED:")
print(OUTPUT_DIR / "full_source_inventory.csv")
print(OUTPUT_DIR / "domain_inventory.csv")
print(OUTPUT_DIR / "machine_source_inventory.csv")
