
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")

file_path = OUTPUT_DIR / "official_all_split_metadata.csv"

df = pd.read_csv(file_path)

# Find candidate domains present in test but absent from train.
train_domains = set(
    df.loc[df["split_name"] == "train", "domain"]
)

test_domains = set(
    df.loc[df["split_name"] == "test", "domain"]
)

test_only_domains = sorted(test_domains - train_domains)

print("\nTEST-ONLY DOMAIN CANDIDATES")
print(test_only_domains)

# Inspect source and label combinations.
extra = df[
    (df["split_name"] == "test")
    & (df["domain"].isin(test_only_domains))
].copy()

print("\nRECORD COUNTS BY DOMAIN AND LABEL")
print(
    pd.crosstab(
        extra["domain"],
        extra["label"]
    ).to_string()
)

print("\nSOURCE CATEGORIES AND LABELS")
source_counts = (
    extra.groupby(["domain", "src", "label"])
    .size()
    .reset_index(name="records")
)

print(source_counts.to_string(index=False))

source_counts.to_csv(
    OUTPUT_DIR / "test_only_domain_source_analysis.csv",
    index=False
)

print("\nSaved:")
print(OUTPUT_DIR / "test_only_domain_source_analysis.csv")
