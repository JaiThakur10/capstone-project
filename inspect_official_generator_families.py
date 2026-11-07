
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
INPUT_FILE = OUTPUT_DIR / "official_all_split_metadata.csv"

df = pd.read_csv(INPUT_FILE)

# Official MAGE model-family definitions.
MODEL_FAMILIES = {
    "OpenAI": [
        "gpt-3.5-trubo",
        "text-davinci-003",
        "text-davinci-002",
    ],
    "LLaMA": [
        "_7b",
        "_13b",
        "_30b",
        "_65b",
    ],
    "GLM": ["glm130b"],
    "FLAN-T5": [
        "flan_t5_small",
        "flan_t5_base",
        "flan_t5_large",
        "flan_t5_xl",
        "flan_t5_xxl",
    ],
    "OPT": [
        "opt_iml_max_1.3b",
        "opt_iml_30b",
        "opt_125m",
        "opt_350m",
        "opt_1.3b",
        "opt_2.7b",
        "opt_6.7b",
        "opt_13b",
        "opt_30b",
    ],
    "BigScience": [
        "bloom_7b",
        "t0_3b",
        "t0_11b",
    ],
    "EleutherAI": [
        "gpt_j",
        "gpt_neox",
    ],
}

# Longer patterns are checked first to avoid
# matching an OPT model as a generic LLaMA suffix.
PATTERNS = sorted(
    [
        (family, pattern)
        for family, patterns in MODEL_FAMILIES.items()
        for pattern in patterns
    ],
    key=lambda item: len(item[1]),
    reverse=True,
)


def identify_family(src):

    src = str(src).lower()

    if "_machine_" not in src:
        return None

    for family, pattern in PATTERNS:
        if pattern in src:
            return family

    return "Unmapped machine source"


df["model_family"] = df["src"].apply(identify_family)

# Original ten MAGE domains.
STANDARD_DOMAINS = [
    "cmv", "yelp", "xsum", "tldr", "eli5",
    "wp", "roct", "hswag", "squad", "sci_gen",
]

# Restrict family analysis to standard-domain
# machine-generated examples.
machine = df[
    (df["label"] == 0)
    & (df["domain"].isin(STANDARD_DOMAINS))
    & df["src"].str.contains("_machine_", regex=False)
].copy()

# 1. Total AI records by official generator family.
family_counts = pd.crosstab(
    machine["split_name"],
    machine["model_family"]
)

family_counts.to_csv(
    OUTPUT_DIR / "official_model_family_counts.csv"
)

print("\nAI RECORDS BY OFFICIAL MODEL FAMILY")
print(family_counts.to_string())

# 2. Detailed domain and family counts.
domain_family = (
    machine.groupby([
        "split_name",
        "domain",
        "model_family"
    ])
    .size()
    .reset_index(name="AI_records")
)

domain_family.to_csv(
    OUTPUT_DIR / "official_domain_model_family_counts.csv",
    index=False
)

# 3. Focus on Yelp for cross-model experiment.
yelp_family = domain_family[
    domain_family["domain"] == "yelp"
].pivot(
    index="split_name",
    columns="model_family",
    values="AI_records"
).fillna(0).astype(int)

print("\nYELP AI RECORDS BY MODEL FAMILY")
print(yelp_family.to_string())

# 4. Check whether any standard machine sources
# were not mapped to an official family.
unmapped = machine[
    machine["model_family"] == "Unmapped machine source"
]

print("\nUNMAPPED STANDARD MACHINE SOURCES")
print("Number of records:", len(unmapped))

if not unmapped.empty:
    print(
        unmapped.groupby("src").size()
        .sort_values(ascending=False)
        .head(30).to_string()
    )

# 5. Inspect the additional wilder domains separately.
wilder_domains = [
    "cnn", "dialogsum", "imdb", "pubmed"
]

wilder = df[
    (df["split_name"] == "test")
    & (df["domain"].isin(wilder_domains))
]

wilder_summary = pd.crosstab(
    [wilder["domain"], wilder["src"]],
    wilder["label"]
)

wilder_summary.to_csv(
    OUTPUT_DIR / "wilder_test_source_label_counts.csv"
)

print("\nWILDER TEST SOURCE AND LABEL COUNTS")
print(wilder_summary.to_string())

print("\nAUDIT COMPLETED.")

print("\nSaved:")
print(OUTPUT_DIR / "official_model_family_counts.csv")
print(OUTPUT_DIR / "official_domain_model_family_counts.csv")
print(OUTPUT_DIR / "wilder_test_source_label_counts.csv")
