
import pandas as pd
import numpy as np

from pathlib import Path
from datasets import load_dataset


# ==================================================
# 1. EXPERIMENT CONFIGURATION
# ==================================================

OUTPUT_DIR = Path("outputs") / "milestone2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "yaful/MAGE"
SEED = 42

STANDARD_DOMAINS = [
    "cmv", "yelp", "xsum", "tldr", "eli5",
    "wp", "roct", "hswag", "squad", "sci_gen"
]

TARGET_DOMAIN = "eli5"
TARGET_MODEL_FAMILY = "OPT"

# Resource-conscious sample sizes.
DOMAIN_TRAIN_PER_CLASS = 200
DOMAIN_VALID_PER_CLASS = 100
DOMAIN_REFERENCE_TEST_PER_CLASS = 100

MODEL_TRAIN_PER_CLASS = 200
MODEL_VALID_PER_CLASS = 100


# Official MAGE model-family definitions.
MODEL_FAMILIES = {
    "OpenAI": [
        "gpt-3.5-trubo",
        "text-davinci-003",
        "text-davinci-002",
    ],
    "LLaMA": [
        "_7b", "_13b", "_30b", "_65b"
    ],
    "GLM": [
        "glm130b"
    ],
    "FLAN-T5": [
        "flan_t5_small",
        "flan_t5_base",
        "flan_t5_large",
        "flan_t5_xl",
        "flan_t5_xxl",
    ],
    "OPT": [
        "opt_125m",
        "opt_350m",
        "opt_1.3b",
        "opt_2.7b",
        "opt_6.7b",
        "opt_13b",
        "opt_30b",
        "opt_iml_30b",
        "opt_iml_max_1.3b",
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

MODEL_PATTERNS = sorted(
    [
        (family, pattern)
        for family, patterns in MODEL_FAMILIES.items()
        for pattern in patterns
    ],
    key=lambda item: len(item[1]),
    reverse=True,
)


# ==================================================
# 2. METADATA PARSING
# ==================================================

def extract_domain(src):

    src = str(src).lower()

    for domain in sorted(
        STANDARD_DOMAINS,
        key=len,
        reverse=True
    ):
        if src == domain or src.startswith(domain + "_"):
            return domain

    return "unknown"


def classify_source(src):

    src = str(src).lower()

    if src.endswith("_human"):
        return "human"

    if "_machine_" in src:
        return "machine"

    return "other"


def extract_model_family(src):

    src = str(src).lower()

    if "_machine_" not in src:
        return None

    for family, pattern in MODEL_PATTERNS:
        if pattern in src:
            return family

    return "unmapped"


# ==================================================
# 3. LOAD AND VALIDATE OFFICIAL SPLITS
# ==================================================

def load_split(split_name):

    print(f"\nLoading {split_name} split...")

    dataset = load_dataset(
        DATASET,
        split=split_name
    )

    frame = dataset.select_columns(
        ["text", "label", "src"]
    ).to_pandas()

    frame = frame.dropna(
        subset=["text", "label", "src"]
    ).copy()

    frame["text"] = frame["text"].astype(str)
    frame["src"] = frame["src"].astype(str)
    frame["label"] = frame["label"].astype(int)

    frame["domain"] = frame["src"].apply(
        extract_domain
    )

    frame["source_type"] = frame["src"].apply(
        classify_source
    )

    frame["model_family"] = frame["src"].apply(
        extract_model_family
    )

    # Retain only the original ten standard domains.
    frame = frame[
        frame["domain"].isin(STANDARD_DOMAINS)
    ].copy()

    # Remove ambiguous duplicate texts within a split.
    conflicting = (
        frame.groupby("text")["label"].nunique()
    )

    conflicting_texts = set(
        conflicting[conflicting > 1].index
    )

    if conflicting_texts:
        print(
            f"Removing {len(conflicting_texts)} "
            "texts with conflicting labels."
        )

        frame = frame[
            ~frame["text"].isin(conflicting_texts)
        ].copy()

    before = len(frame)

    frame = frame.drop_duplicates(
        subset=["text"]
    ).copy()

    print("Duplicate rows removed:", before - len(frame))

    # Validate the original MAGE source-label mapping.
    invalid_human = (
        (frame["source_type"] == "human")
        & (frame["label"] != 1)
    )

    invalid_machine = (
        (frame["source_type"] == "machine")
        & (frame["label"] != 0)
    )

    if invalid_human.any() or invalid_machine.any():
        raise ValueError(
            f"Source-label mismatch in {split_name}."
        )

    if frame["source_type"].eq("other").any():
        raise ValueError(
            f"Unknown standard source type in {split_name}."
        )

    if (
        frame.loc[
            frame["label"] == 0,
            "model_family"
        ] == "unmapped"
    ).any():
        raise ValueError(
            f"Unmapped machine generator in {split_name}."
        )

    print("Records retained:", len(frame))
    print("Label counts:")
    print(
        frame["label"].value_counts().sort_index().to_string()
    )

    print(
        "Generator families:",
        sorted(
            frame["model_family"].dropna().unique()
        )
    )

    return frame.reset_index(drop=True)


train = load_split("train")
validation = load_split("validation")
test = load_split("test")


# ==================================================
# 4. HELPERS FOR REPRODUCIBLE SAMPLING
# ==================================================

def exclude_texts(frame, forbidden_texts):

    return frame[
        ~frame["text"].isin(forbidden_texts)
    ].copy()


def sample_human(pool, n, seed):

    if len(pool) < n:
        raise ValueError(
            f"Insufficient human examples: "
            f"{len(pool)} available, {n} required."
        )

    return pool.sample(
        n=n,
        random_state=seed
    ).copy()


def sample_ai_by_family(pool, n, seed):

    if len(pool) < n:
        raise ValueError(
            f"Insufficient AI examples: "
            f"{len(pool)} available, {n} required."
        )

    # Proportionally distribute the requested sample
    # across the available model families.
    groups = {
        name: group
        for name, group in pool.groupby("model_family")
    }

    counts = pd.Series({
        name: len(group)
        for name, group in groups.items()
    }).sort_index()

    exact = counts / counts.sum() * n

    quotas = np.floor(exact).astype(int)

    remaining = n - int(quotas.sum())

    remainders = (
        exact - quotas
    ).sort_values(ascending=False)

    for family in remainders.index[:remaining]:
        quotas.loc[family] += 1

    selected = []

    for i, (family, quota) in enumerate(
        quotas.items()
    ):

        if quota == 0:
            continue

        selected.append(
            groups[family].sample(
                n=int(quota),
                random_state=seed + i
            )
        )

    result = pd.concat(
        selected,
        ignore_index=True
    )

    return result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)


def sample_balanced_by_domain(
    frame,
    domains,
    n_per_class,
    seed,
    excluded_family=None,
    forbidden_texts=None
):

    selected = []

    forbidden_texts = (
        forbidden_texts
        if forbidden_texts is not None
        else set()
    )

    for i, domain in enumerate(domains):

        domain_data = frame[
            frame["domain"] == domain
        ].copy()

        domain_data = exclude_texts(
            domain_data,
            forbidden_texts
        )

        human_pool = domain_data[
            (domain_data["label"] == 1)
            & (domain_data["source_type"] == "human")
        ]

        machine_pool = domain_data[
            (domain_data["label"] == 0)
            & (domain_data["source_type"] == "machine")
        ]

        if excluded_family is not None:

            machine_pool = machine_pool[
                machine_pool["model_family"]
                != excluded_family
            ]

        human_sample = sample_human(
            human_pool,
            n_per_class,
            seed + i * 100
        )

        machine_sample = sample_ai_by_family(
            machine_pool,
            n_per_class,
            seed + i * 100 + 1
        )

        selected.extend([
            human_sample,
            machine_sample
        ])

    result = pd.concat(
        selected,
        ignore_index=True
    )

    return result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)


def remove_seen_text(frame, forbidden_texts, name):

    overlap = frame["text"].isin(forbidden_texts)

    removed = int(overlap.sum())

    if removed:
        print(
            f"{name}: removed {removed} "
            "exact overlapping records."
        )

    return frame.loc[~overlap].copy()


def save_frame(frame, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    frame.to_csv(
        path,
        index=False
    )

    print(
        f"Saved {len(frame)} records:",
        path
    )


def describe_frame(name, frame):

    print(f"\n{name}")
    print("Total records:", len(frame))

    print("\nLabels:")
    print(
        frame["label"].value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDomains:")
    print(frame["domain"].value_counts().to_string())

    machine = frame[
        frame["source_type"] == "machine"
    ]

    print("\nGenerator families:")
    print(
        machine["model_family"]
        .value_counts()
        .to_string()
    )


# ==================================================
# 5. EXPERIMENT A: UNSEEN DOMAIN
# ==================================================

print("\n" + "=" * 65)
print("EXPERIMENT A: UNSEEN-DOMAIN EVALUATION")
print("=" * 65)

seen_domains = [
    domain for domain in STANDARD_DOMAINS
    if domain != TARGET_DOMAIN
]

domain_dir = OUTPUT_DIR / "unseen_domain_eli5"

# Training: nine domains; ELI5 excluded.
domain_train = sample_balanced_by_domain(
    train,
    seen_domains,
    DOMAIN_TRAIN_PER_CLASS,
    seed=SEED
)

# Validation: same nine domains; ELI5 excluded.
domain_validation = sample_balanced_by_domain(
    validation,
    seen_domains,
    DOMAIN_VALID_PER_CLASS,
    seed=SEED + 1000,
    forbidden_texts=set(domain_train["text"])
)

# ID test: new records from the nine seen domains.
domain_test_seen = sample_balanced_by_domain(
    test,
    seen_domains,
    DOMAIN_REFERENCE_TEST_PER_CLASS,
    seed=SEED + 2000,
    forbidden_texts=(
        set(domain_train["text"])
        | set(domain_validation["text"])
    )
)

# OOD test: all eligible ELI5 records in official test.
domain_test_unseen = test[
    test["domain"] == TARGET_DOMAIN
].copy()

# Remove exact text already used in the selected datasets.
forbidden_domain_test = (
    set(domain_train["text"])
    | set(domain_validation["text"])
    | set(domain_test_seen["text"])
)

domain_test_unseen = remove_seen_text(
    domain_test_unseen,
    forbidden_domain_test,
    "Unseen-domain test"
)

if domain_test_unseen.empty:
    raise ValueError("No unseen-domain test examples remain.")

if not (
    domain_test_unseen["domain"] == TARGET_DOMAIN
).all():
    raise ValueError("Wrong domain in unseen-domain test.")

# Save the experiment data.
save_frame(
    domain_train,
    domain_dir / "train.csv"
)

save_frame(
    domain_validation,
    domain_dir / "validation.csv"
)

save_frame(
    domain_test_seen,
    domain_dir / "test_in_domain.csv"
)

save_frame(
    domain_test_unseen,
    domain_dir / "test_unseen_domain.csv"
)

describe_frame(
    "UNSEEN-DOMAIN TRAINING",
    domain_train
)

describe_frame(
    "UNSEEN-DOMAIN REFERENCE TEST",
    domain_test_seen
)

describe_frame(
    "UNSEEN-DOMAIN OOD TEST",
    domain_test_unseen
)


# ==================================================
# 6. EXPERIMENT B: UNSEEN MODEL FAMILY
# ==================================================

print("\n" + "=" * 65)
print("EXPERIMENT B: UNSEEN MODEL-FAMILY EVALUATION")
print("=" * 65)

model_dir = OUTPUT_DIR / "unseen_model_OPT"

all_domains = STANDARD_DOMAINS

# Training excludes all OPT machine sources.
model_train = sample_balanced_by_domain(
    train,
    all_domains,
    MODEL_TRAIN_PER_CLASS,
    seed=SEED + 3000,
    excluded_family=TARGET_MODEL_FAMILY
)

# Validation also excludes the OPT family.
model_validation = sample_balanced_by_domain(
    validation,
    all_domains,
    MODEL_VALID_PER_CLASS,
    seed=SEED + 4000,
    excluded_family=TARGET_MODEL_FAMILY,
    forbidden_texts=set(model_train["text"])
)

# Remove exact overlap with selected train/validation records.
forbidden_model_test = (
    set(model_train["text"])
    | set(model_validation["text"])
)

clean_test = remove_seen_text(
    test,
    forbidden_model_test,
    "Cross-model test"
)

seen_ai_parts = []
unseen_ai_parts = []
human_parts = []

# Match test-domain composition across the two conditions.
for i, domain in enumerate(all_domains):

    target_ai = clean_test[
        (clean_test["domain"] == domain)
        & (clean_test["label"] == 0)
        & (
            clean_test["model_family"]
            == TARGET_MODEL_FAMILY
        )
    ]

    n = len(target_ai)

    if n == 0:
        raise ValueError(
            f"No held-out OPT AI examples in {domain}."
        )

    seen_ai_pool = clean_test[
        (clean_test["domain"] == domain)
        & (clean_test["label"] == 0)
        & (
            clean_test["model_family"]
            != TARGET_MODEL_FAMILY
        )
    ]

    human_pool = clean_test[
        (clean_test["domain"] == domain)
        & (clean_test["label"] == 1)
        & (clean_test["source_type"] == "human")
    ]

    # Select the same number of seen-model AI examples
    # as unseen-family AI examples in this domain.
    seen_ai = sample_ai_by_family(
        seen_ai_pool,
        n,
        SEED + 5000 + i
    )

    # Select human examples once and reuse them in both
    # evaluation conditions as matched human controls.
    human = sample_human(
        human_pool,
        n,
        SEED + 6000 + i
    )

    seen_ai_parts.append(seen_ai)
    unseen_ai_parts.append(target_ai.copy())
    human_parts.append(human)


seen_ai_test = pd.concat(
    seen_ai_parts,
    ignore_index=True
)

unseen_ai_test = pd.concat(
    unseen_ai_parts,
    ignore_index=True
)

shared_human_test = pd.concat(
    human_parts,
    ignore_index=True
)

# Seen-family reference test.
model_test_seen = pd.concat(
    [seen_ai_test, shared_human_test],
    ignore_index=True
).sample(
    frac=1,
    random_state=SEED + 7000
).reset_index(drop=True)

# Unseen-family test uses exactly the same human examples.
model_test_unseen = pd.concat(
    [unseen_ai_test, shared_human_test],
    ignore_index=True
).sample(
    frac=1,
    random_state=SEED + 8000
).reset_index(drop=True)


# ==================================================
# 7. VALIDATE FAMILY SEPARATION AND TEST MATCHING
# ==================================================

training_families = set(
    model_train.loc[
        model_train["label"] == 0,
        "model_family"
    ].unique()
)

if TARGET_MODEL_FAMILY in training_families:
    raise ValueError(
        "OPT family leaked into the training sample."
    )

if TARGET_MODEL_FAMILY in set(
    model_validation.loc[
        model_validation["label"] == 0,
        "model_family"
    ].unique()
):
    raise ValueError(
        "OPT family leaked into validation."
    )

if not (
    set(unseen_ai_test["model_family"].unique())
    == {TARGET_MODEL_FAMILY}
):
    raise ValueError(
        "Unexpected family in held-out AI test."
    )

# Same human texts across the two test conditions are
# intentional matched controls.
assert (
    set(seen_ai_test["text"])
    .isdisjoint(set(unseen_ai_test["text"]))
)

assert (
    set(model_train["text"])
    .isdisjoint(set(model_test_seen["text"]))
)

assert (
    set(model_train["text"])
    .isdisjoint(set(model_test_unseen["text"]))
)

# Check matching counts per domain.
seen_human_counts = (
    model_test_seen[
        model_test_seen["label"] == 1
    ]["domain"].value_counts().sort_index()
)

unseen_human_counts = (
    model_test_unseen[
        model_test_unseen["label"] == 1
    ]["domain"].value_counts().sort_index()
)

seen_ai_counts = (
    model_test_seen[
        model_test_seen["label"] == 0
    ]["domain"].value_counts().sort_index()
)

unseen_ai_counts = (
    model_test_unseen[
        model_test_unseen["label"] == 0
    ]["domain"].value_counts().sort_index()
)

if not seen_human_counts.equals(unseen_human_counts):
    raise ValueError("Human test-domain distributions differ.")

if not seen_ai_counts.equals(unseen_ai_counts):
    raise ValueError("AI test-domain distributions differ.")

# Save cross-model datasets.
save_frame(
    model_train,
    model_dir / "train.csv"
)

save_frame(
    model_validation,
    model_dir / "validation.csv"
)

save_frame(
    model_test_seen,
    model_dir / "test_seen_family.csv"
)

save_frame(
    model_test_unseen,
    model_dir / "test_unseen_OPT_family.csv"
)


describe_frame(
    "UNSEEN-MODEL TRAINING",
    model_train
)

describe_frame(
    "SEEN-FAMILY REFERENCE TEST",
    model_test_seen
)

describe_frame(
    "UNSEEN OPT FAMILY TEST",
    model_test_unseen
)


# ==================================================
# 8. SAVE EXPERIMENT SUMMARY
# ==================================================

summary = []

def add_summary(experiment, condition, frame):

    counts = frame["label"].value_counts()

    summary.append({
        "experiment": experiment,
        "condition": condition,
        "records": len(frame),
        "AI_label_0": int(counts.get(0, 0)),
        "human_label_1": int(counts.get(1, 0)),
        "distinct_domains": frame["domain"].nunique(),
        "distinct_sources": frame["src"].nunique(),
    })


add_summary(
    "Unseen domain",
    "Training",
    domain_train
)
add_summary(
    "Unseen domain",
    "Validation",
    domain_validation
)
add_summary(
    "Unseen domain",
    "Seen-domain reference test",
    domain_test_seen
)
add_summary(
    "Unseen domain",
    "Unseen ELI5 test",
    domain_test_unseen
)

add_summary(
    "Unseen model",
    "Training",
    model_train
)
add_summary(
    "Unseen model",
    "Validation",
    model_validation
)
add_summary(
    "Unseen model",
    "Seen-family reference test",
    model_test_seen
)
add_summary(
    "Unseen model",
    "Unseen OPT-family test",
    model_test_unseen
)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_DIR / "milestone2_preparation_summary.csv",
    index=False
)

print("\n" + "=" * 65)
print("MILESTONE 2 DATA PREPARATION COMPLETED")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nOutput folder:", OUTPUT_DIR.resolve())
