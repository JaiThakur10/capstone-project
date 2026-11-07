
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEEDS = [42, 123, 2026]

STANDARD_DOMAINS = [
    "cmv", "yelp", "xsum", "tldr", "eli5",
    "wp", "roct", "hswag", "squad", "sci_gen"
]

UNSEEN_DOMAIN = "eli5"
UNSEEN_FAMILY = "OPT"

TRAIN_PER_CLASS_PER_DOMAIN = 200

BASE_DIR = Path("outputs/milestone2")
DOMAIN_DIR = BASE_DIR / "unseen_domain_eli5"
MODEL_DIR = BASE_DIR / "unseen_model_OPT"

REPEAT_DIR = BASE_DIR / "repeatability"
MODEL_SAVE_DIR = REPEAT_DIR / "models"

REPEAT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. OFFICIAL MAGE GENERATOR-FAMILY MAPPING
# ============================================================

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

MODEL_PATTERNS = sorted(
    [
        (family, pattern)
        for family, patterns in MODEL_FAMILIES.items()
        for pattern in patterns
    ],
    key=lambda item: len(item[1]),
    reverse=True,
)


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


# ============================================================
# 3. LOAD THE OFFICIAL TRAINING SPLIT
# ============================================================

print("=" * 65)
print("MILESTONE 2 REPEATABILITY EXPERIMENT")
print("=" * 65)

print("\nLoading official MAGE training split...")

train = load_dataset(
    "yaful/MAGE",
    split="train"
).select_columns(
    ["text", "label", "src"]
).to_pandas()

train = train.dropna(
    subset=["text", "label", "src"]
).copy()

train["text"] = train["text"].astype(str)
train["src"] = train["src"].astype(str)
train["label"] = train["label"].astype(int)

train["domain"] = train["src"].apply(extract_domain)

train["source_type"] = train["src"].apply(classify_source)

train["model_family"] = train["src"].apply(
    extract_model_family
)

# Keep only the ten standard domains.
train = train[
    train["domain"].isin(STANDARD_DOMAINS)
].copy()

# Keep only clearly identified original human and machine sources.
train = train[
    train["source_type"].isin(["human", "machine"])
].copy()

# Verify source-label mapping.
invalid_human = (
    (train["source_type"] == "human")
    & (train["label"] != 1)
)

invalid_machine = (
    (train["source_type"] == "machine")
    & (train["label"] != 0)
)

if invalid_human.any() or invalid_machine.any():
    raise ValueError("Unexpected source-label mismatch.")

if (
    train.loc[
        train["label"] == 0,
        "model_family"
    ] == "unmapped"
).any():
    raise ValueError("Unmapped machine generator found.")

train = train.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

print("Official training records available:", len(train))


# ============================================================
# 4. LOAD THE FIXED VALIDATION AND TEST DATASETS
# ============================================================

def load_prepared(path, name):

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "text", "label", "src",
        "domain", "source_type",
        "model_family"
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{name} is missing: {missing}"
        )

    frame = frame.dropna(
        subset=["text", "label", "src", "domain"]
    ).copy()

    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)

    if frame["text"].duplicated().any():
        raise ValueError(
            f"Duplicate text found in {name}."
        )

    return frame.reset_index(drop=True)


domain_validation = load_prepared(
    DOMAIN_DIR / "validation.csv",
    "Domain validation"
)

domain_test_seen = load_prepared(
    DOMAIN_DIR / "test_in_domain.csv",
    "Domain reference test"
)

domain_test_unseen = load_prepared(
    DOMAIN_DIR / "test_unseen_domain.csv",
    "Unseen ELI5 test"
)

model_validation = load_prepared(
    MODEL_DIR / "validation.csv",
    "Model validation"
)

model_test_seen = load_prepared(
    MODEL_DIR / "test_seen_family.csv",
    "Seen-family test"
)

model_test_unseen = load_prepared(
    MODEL_DIR / "test_unseen_OPT_family.csv",
    "Unseen OPT test"
)


# ============================================================
# 5. VERIFY THAT THE TEST CONDITIONS ARE CORRECT
# ============================================================

if not (
    domain_test_unseen["domain"] == UNSEEN_DOMAIN
).all():
    raise ValueError("Unexpected domain in ELI5 test.")

if not (
    model_test_unseen.loc[
        model_test_unseen["label"] == 0,
        "model_family"
    ] == UNSEEN_FAMILY
).all():
    raise ValueError("Unexpected family in OPT test.")

if not (
    model_test_seen.loc[
        model_test_seen["label"] == 0,
        "model_family"
    ] != UNSEEN_FAMILY
).all():
    raise ValueError("OPT examples found in seen-family reference test.")

# The two model-family tests intentionally share human examples.
seen_human = set(
    model_test_seen.loc[
        model_test_seen["label"] == 1, "text"
    ]
)

unseen_human = set(
    model_test_unseen.loc[
        model_test_unseen["label"] == 1, "text"
    ]
)

if seen_human != unseen_human:
    raise ValueError(
        "Model-family tests do not contain identical human examples."
    )

print("\nFixed validation and test datasets loaded.")
print("ELI5 test records:", len(domain_test_unseen))
print("Seen-family test records:", len(model_test_seen))
print("Unseen OPT test records:", len(model_test_unseen))


# ============================================================
# 6. SAMPLING FUNCTIONS
# ============================================================

def sample_human(pool, count, seed):

    if len(pool) < count:
        raise ValueError(
            f"Insufficient human records: {len(pool)}"
        )

    return pool.sample(
        n=count,
        random_state=seed
    ).copy()


def sample_ai_by_family(pool, count, seed):

    pool = pool[
        pool["model_family"].notna()
        & (pool["model_family"] != "unmapped")
    ].copy()

    if len(pool) < count:
        raise ValueError(
            f"Insufficient machine records: {len(pool)}"
        )

    groups = {
        family: group
        for family, group in pool.groupby("model_family")
    }

    sizes = pd.Series({
        family: len(group)
        for family, group in groups.items()
    }).sort_index()

    exact_quotas = sizes / sizes.sum() * count

    quotas = np.floor(exact_quotas).astype(int)

    remainder = count - int(quotas.sum())

    fractional = (
        exact_quotas - quotas
    ).sort_values(ascending=False)

    for family in fractional.index[:remainder]:
        quotas.loc[family] += 1

    parts = []

    for index, (family, quota) in enumerate(
        quotas.items()
    ):

        if quota == 0:
            continue

        parts.append(
            groups[family].sample(
                n=int(quota),
                random_state=seed + index
            )
        )

    result = pd.concat(
        parts,
        ignore_index=True
    )

    return result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)


def remove_forbidden_text(pool, forbidden_texts):

    return pool[
        ~pool["text"].isin(forbidden_texts)
    ].copy()


# ============================================================
# 7. BUILD FRESH UNSEEN-DOMAIN TRAINING SAMPLES
# ============================================================

def build_domain_training(seed):

    allowed_domains = [
        domain for domain in STANDARD_DOMAINS
        if domain != UNSEEN_DOMAIN
    ]

    forbidden = (
        set(domain_validation["text"])
        | set(domain_test_seen["text"])
        | set(domain_test_unseen["text"])
    )

    parts = []

    for index, domain in enumerate(allowed_domains):

        subset = train[
            train["domain"] == domain
        ].copy()

        subset = remove_forbidden_text(
            subset, forbidden
        )

        human_pool = subset[
            (subset["label"] == 1)
            & (subset["source_type"] == "human")
        ]

        machine_pool = subset[
            (subset["label"] == 0)
            & (subset["source_type"] == "machine")
        ]

        human_sample = sample_human(
            human_pool,
            TRAIN_PER_CLASS_PER_DOMAIN,
            seed + index * 100
        )

        machine_sample = sample_ai_by_family(
            machine_pool,
            TRAIN_PER_CLASS_PER_DOMAIN,
            seed + index * 100 + 1
        )

        parts.extend([
            human_sample,
            machine_sample
        ])

    result = pd.concat(
        parts,
        ignore_index=True
    ).sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    if len(result) != 3600:
        raise ValueError(
            "Unexpected unseen-domain training size."
        )

    if result["domain"].eq(UNSEEN_DOMAIN).any():
        raise ValueError("ELI5 leaked into training.")

    return result


# ============================================================
# 8. BUILD FRESH UNSEEN-MODEL TRAINING SAMPLES
# ============================================================

def build_model_training(seed):

    forbidden = (
        set(model_validation["text"])
        | set(model_test_seen["text"])
        | set(model_test_unseen["text"])
    )

    parts = []

    for index, domain in enumerate(STANDARD_DOMAINS):

        subset = train[
            train["domain"] == domain
        ].copy()

        subset = remove_forbidden_text(
            subset, forbidden
        )

        human_pool = subset[
            (subset["label"] == 1)
            & (subset["source_type"] == "human")
        ]

        machine_pool = subset[
            (subset["label"] == 0)
            & (subset["source_type"] == "machine")
            & (
                subset["model_family"]
                != UNSEEN_FAMILY
            )
        ]

        human_sample = sample_human(
            human_pool,
            TRAIN_PER_CLASS_PER_DOMAIN,
            seed + index * 100
        )

        machine_sample = sample_ai_by_family(
            machine_pool,
            TRAIN_PER_CLASS_PER_DOMAIN,
            seed + index * 100 + 1
        )

        parts.extend([
            human_sample,
            machine_sample
        ])

    result = pd.concat(
        parts,
        ignore_index=True
    ).sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    if len(result) != 4000:
        raise ValueError(
            "Unexpected unseen-model training size."
        )

    if (
        result.loc[
            result["label"] == 0,
            "model_family"
        ] == UNSEEN_FAMILY
    ).any():
        raise ValueError("OPT family leaked into training.")

    return result


# ============================================================
# 9. CHECK SPLIT OVERLAP
# ============================================================

def check_overlap(train_frame, validation, tests, name):

    training_texts = set(train_frame["text"])
    validation_texts = set(validation["text"])

    if training_texts & validation_texts:
        raise ValueError(
            f"{name}: training-validation overlap."
        )

    if not training_texts.isdisjoint(validation_texts):
        raise ValueError("Unexpected overlap.")

    for condition, test_frame in tests.items():

        test_texts = set(test_frame["text"])

        if training_texts & test_texts:
            raise ValueError(
                f"{name}: training-test overlap in {condition}."
            )

        if validation_texts & test_texts:
            raise ValueError(
                f"{name}: validation-test overlap in {condition}."
            )

    print(f"{name}: exact split-overlap checks passed.")


# ============================================================
# 10. CREATE THE CLASSIFIER
# ============================================================

def make_model(seed):

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=seed
            )
        )
    ])


# ============================================================
# 11. SELECT BALANCED-ACCURACY THRESHOLD ON VALIDATION
# ============================================================

def get_ai_scores(model, frame):

    probabilities = model.predict_proba(
        frame["text"]
    )

    classes = list(model.classes_)

    ai_index = classes.index(0)

    return probabilities[:, ai_index]


def choose_balanced_threshold(model, validation):

    true_ai = (
        validation["label"].to_numpy() == 0
    ).astype(int)

    scores = get_ai_scores(
        model, validation
    )

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(
        true_ai, scores
    )

    balanced_values = (
        tpr + 1 - fpr
    ) / 2

    finite = np.isfinite(thresholds)

    valid_indices = np.where(finite)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            "No finite validation threshold found."
        )

    best_index = valid_indices[
        np.argmax(balanced_values[finite])
    ]

    return float(thresholds[best_index])


# ============================================================
# 12. EVALUATE ONE TEST CONDITION
# ============================================================

def evaluate_condition(
    model,
    frame,
    threshold,
    experiment,
    condition,
    seed
):

    true_labels = frame["label"].to_numpy()

    scores = get_ai_scores(
        model, frame
    )

    predictions = np.where(
        scores >= threshold, 0, 1
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1]
    )

    human_total = matrix[1].sum()

    human_fpr = (
        matrix[1, 0] / human_total
        if human_total else float("nan")
    )

    metrics = {
        "seed": seed,
        "experiment": experiment,
        "condition": condition,
        "records": len(frame),
        "threshold": threshold,

        "accuracy": accuracy_score(
            true_labels, predictions
        ),

        "balanced_accuracy": balanced_accuracy_score(
            true_labels, predictions
        ),

        "ai_precision": precision_score(
            true_labels,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "ai_recall": recall_score(
            true_labels,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "ai_f1": f1_score(
            true_labels,
            predictions,
            pos_label=0,
            zero_division=0
        ),

        "human_false_positive_rate": human_fpr,

        "ai_roc_auc": roc_auc_score(
            (true_labels == 0).astype(int),
            scores
        )
    }

    confusion = {
        "seed": seed,
        "experiment": experiment,
        "condition": condition,
        "true_AI_pred_AI": int(matrix[0, 0]),
        "true_AI_pred_Human": int(matrix[0, 1]),
        "true_Human_pred_AI": int(matrix[1, 0]),
        "true_Human_pred_Human": int(matrix[1, 1]),
    }

    return metrics, confusion


# ============================================================
# 13. RUN THE SIX TRAINING RUNS
# ============================================================

all_metrics = []
all_confusions = []
training_provenance = []
threshold_records = []

for seed in SEEDS:

    print("\n" + "=" * 65)
    print(f"TRAINING SEED: {seed}")
    print("=" * 65)

    # --------------------------------------------------------
    # A. UNSEEN-DOMAIN EXPERIMENT
    # --------------------------------------------------------

    print("\nBuilding unseen-domain training sample...")

    domain_train = build_domain_training(seed)

    check_overlap(
        domain_train,
        domain_validation,
        {
            "in-domain": domain_test_seen,
            "unseen ELI5": domain_test_unseen
        },
        "Unseen-domain experiment"
    )

    print("Training records:", len(domain_train))

    domain_model = make_model(seed)

    domain_model.fit(
        domain_train["text"],
        domain_train["label"]
    )

    domain_threshold = choose_balanced_threshold(
        domain_model,
        domain_validation
    )

    print(
        "Unseen-domain validation threshold:",
        round(domain_threshold, 4)
    )

    domain_model_path = (
        MODEL_SAVE_DIR
        / f"unseen_domain_seed_{seed}.joblib"
    )

    joblib.dump(
        domain_model,
        domain_model_path,
        compress=3
    )

    threshold_records.append({
        "seed": seed,
        "experiment": "Unseen domain",
        "threshold": domain_threshold
    })

    for condition, test_frame in [
        ("In-domain reference", domain_test_seen),
        ("Unseen ELI5", domain_test_unseen)
    ]:

        metrics, confusion = evaluate_condition(
            domain_model,
            test_frame,
            domain_threshold,
            "Unseen domain",
            condition,
            seed
        )

        all_metrics.append(metrics)
        all_confusions.append(confusion)

    # Store source distribution for reproducibility.
    provenance = (
        domain_train.groupby([
            "domain", "source_type",
            "model_family", "label"
        ], dropna=False)
        .size()
        .reset_index(name="records")
    )

    provenance["experiment"] = "Unseen domain"
    provenance["seed"] = seed

    training_provenance.append(provenance)

    # --------------------------------------------------------
    # B. UNSEEN-MODEL EXPERIMENT
    # --------------------------------------------------------

    print("\nBuilding unseen-model training sample...")

    model_train = build_model_training(seed)

    check_overlap(
        model_train,
        model_validation,
        {
            "seen families": model_test_seen,
            "unseen OPT": model_test_unseen
        },
        "Unseen-model experiment"
    )

    print("Training records:", len(model_train))

    model_classifier = make_model(seed)

    model_classifier.fit(
        model_train["text"],
        model_train["label"]
    )

    model_threshold = choose_balanced_threshold(
        model_classifier,
        model_validation
    )

    print(
        "Unseen-model validation threshold:",
        round(model_threshold, 4)
    )

    model_path = (
        MODEL_SAVE_DIR
        / f"unseen_OPT_seed_{seed}.joblib"
    )

    joblib.dump(
        model_classifier,
        model_path,
        compress=3
    )

    threshold_records.append({
        "seed": seed,
        "experiment": "Unseen model family",
        "threshold": model_threshold
    })

    for condition, test_frame in [
        ("Seen model families", model_test_seen),
        ("Unseen OPT family", model_test_unseen)
    ]:

        metrics, confusion = evaluate_condition(
            model_classifier,
            test_frame,
            model_threshold,
            "Unseen model family",
            condition,
            seed
        )

        all_metrics.append(metrics)
        all_confusions.append(confusion)

    provenance = (
        model_train.groupby([
            "domain", "source_type",
            "model_family", "label"
        ], dropna=False)
        .size()
        .reset_index(name="records")
    )

    provenance["experiment"] = "Unseen model family"
    provenance["seed"] = seed

    training_provenance.append(provenance)


# ============================================================
# 14. SAVE ALL INDIVIDUAL RUN RESULTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
confusion_df = pd.DataFrame(all_confusions)
threshold_df = pd.DataFrame(threshold_records)

provenance_df = pd.concat(
    training_provenance,
    ignore_index=True
)

metrics_df.to_csv(
    REPEAT_DIR / "milestone2_repeat_runs.csv",
    index=False
)

confusion_df.to_csv(
    REPEAT_DIR / "milestone2_repeat_confusions.csv",
    index=False
)

threshold_df.to_csv(
    REPEAT_DIR / "milestone2_repeat_thresholds.csv",
    index=False
)

provenance_df.to_csv(
    REPEAT_DIR / "milestone2_training_provenance.csv",
    index=False
)


# ============================================================
# 15. CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "ai_precision",
    "ai_recall",
    "ai_f1",
    "human_false_positive_rate",
    "ai_roc_auc",
]

summary_rows = []

for (experiment, condition), group in metrics_df.groupby(
    ["experiment", "condition"]
):

    row = {
        "experiment": experiment,
        "condition": condition,
        "runs": len(group),
    }

    for metric in METRICS:

        row[f"{metric}_mean"] = group[metric].mean()

        row[f"{metric}_std"] = group[metric].std(
            ddof=1
        )

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    REPEAT_DIR / "milestone2_repeat_summary.csv",
    index=False
)


# ============================================================
# 16. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("MILESTONE 2 REPEATABILITY COMPLETED")
print("=" * 65)

print("\nALL INDIVIDUAL RUNS")
print(
    metrics_df[
        [
            "seed",
            "experiment",
            "condition",
            "threshold",
            "accuracy",
            "ai_f1",
            "human_false_positive_rate",
            "ai_roc_auc"
        ]
    ].round(4).to_string(index=False)
)

print("\nMEAN AND STANDARD DEVIATION")
print(
    summary_df.round(4).to_string(index=False)
)

print("\nTHRESHOLDS SELECTED FROM VALIDATION")
print(
    threshold_df.to_string(index=False)
)

print("\nFILES SAVED IN:")
print(REPEAT_DIR.resolve())
