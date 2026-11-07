
from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42

print("Loading MAGE training split...")
dataset = load_dataset("yaful/MAGE", split="train")

df = dataset.select_columns(["text", "label", "src"]).to_pandas()

# Extract candidate domain and source type.
# This follows the naming pattern observed in our audit.
df["domain"] = df["src"].str.split("_").str[0]

df["source_type"] = df["src"].apply(
    lambda src: (
        "human"
        if "human" in str(src).split("_")
        else "machine"
        if "machine" in str(src).split("_")
        else "unknown"
    )
)

# Verify the expected label mapping before sampling.
print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

print("\nCandidate domains:")
print(df["domain"].value_counts())

# Keep only the two domains for this first experiment.
selected = df[
    df["domain"].isin(["yelp", "eli5"])
    & df["source_type"].isin(["human", "machine"])
].copy()

# Verify labels and sources before selecting examples.
print("\nSource-by-label counts:")
print(pd.crosstab(selected["src"], selected["label"]))

# Use human text (label 1) and machine text (label 0).
yelp_human = selected[
    (selected["domain"] == "yelp")
    & (selected["label"] == 1)
]

yelp_machine = selected[
    (selected["domain"] == "yelp")
    & (selected["label"] == 0)
]

eli5_human = selected[
    (selected["domain"] == "eli5")
    & (selected["label"] == 1)
]

eli5_machine = selected[
    (selected["domain"] == "eli5")
    & (selected["label"] == 0)
]

# Stop rather than silently producing an incomplete experiment.
required_counts = {
    "yelp human": (yelp_human, 2000),
    "yelp machine": (yelp_machine, 2000),
    "eli5 human": (eli5_human, 500),
    "eli5 machine": (eli5_machine, 500),
}

for name, (part, required) in required_counts.items():
    print(f"{name}: available={len(part)}, required={required}")
    if len(part) < required:
        raise ValueError(
            f"Not enough records for {name}. "
            "Revise the sampling plan before proceeding."
        )

# Create a reproducible, balanced initial subset.
yelp_human_sample = yelp_human.sample(
    n=2000, random_state=SEED
)
yelp_machine_sample = yelp_machine.sample(
    n=2000, random_state=SEED
)
eli5_human_sample = eli5_human.sample(
    n=500, random_state=SEED
)
eli5_machine_sample = eli5_machine.sample(
    n=500, random_state=SEED
)

train_domain = pd.concat(
    [yelp_human_sample, yelp_machine_sample],
    ignore_index=True
).sample(frac=1, random_state=SEED)

test_domain = pd.concat(
    [eli5_human_sample, eli5_machine_sample],
    ignore_index=True
).sample(frac=1, random_state=SEED)

# Remove exact duplicate text across the two sets.
# If this changes the counts, stop and redesign the sample.
overlap = set(train_domain["text"]) & set(test_domain["text"])

print("\nExact text overlap across domains:", len(overlap))

if overlap:
    raise ValueError(
        "Exact duplicate text found across train/test domains. "
        "Remove overlaps and re-check class counts before training."
    )

train_domain.to_csv(
    OUTPUT_DIR / "yelp_domain_training_sample.csv",
    index=False
)

test_domain.to_csv(
    OUTPUT_DIR / "eli5_cross_domain_test_sample.csv",
    index=False
)

print("\nTraining sample:")
print(pd.crosstab(train_domain["domain"], train_domain["label"]))

print("\nCross-domain test sample:")
print(pd.crosstab(test_domain["domain"], test_domain["label"]))

print("\nSaved preliminary experiment datasets.")
