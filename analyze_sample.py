
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("outputs")
sample_path = OUTPUT_DIR / "mage_offset_diagnostic_sample.csv"

df = pd.read_csv(sample_path)

print("Records available:", len(df))
print("Columns:", df.columns.tolist())

# Count labels for each source
source_label_table = pd.crosstab(
    df["src"],
    df["label"],
    margins=True
)

print("\nSOURCE BY LABEL")
print(source_label_table.to_string())

source_label_table.to_csv(
    OUTPUT_DIR / "source_by_label.csv"
)

# Number of distinct sources for each label
source_counts = (
    df.groupby("label")["src"]
    .nunique()
    .rename("distinct_sources")
    .reset_index()
)

print("\nDISTINCT SOURCES PER LABEL")
print(source_counts.to_string(index=False))

source_counts.to_csv(
    OUTPUT_DIR / "distinct_sources_per_label.csv",
    index=False
)

print("\nAnalysis saved in:", OUTPUT_DIR.resolve())
