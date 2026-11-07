
import pandas as pd
import joblib

from pathlib import Path
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 2026]

N_TRAIN_PER_CLASS = 2000
N_TEST_PER_CLASS = 500

TEST_FILE = OUTPUT_DIR / "adaptation_eli5_test.csv"

print("=" * 60)
print("ADAPTATION REPEATABILITY EXPERIMENT")
print("=" * 60)


# --------------------------------------------------
# 2. Load the training data
# --------------------------------------------------

print("\nLoading MAGE training split...")

train = load_dataset(
    "yaful/MAGE",
    split="train"
).select_columns(["text", "label", "src"]).to_pandas()


# --------------------------------------------------
# 3. Prepare metadata
# --------------------------------------------------

def prepare(frame):

    frame = frame.dropna(
        subset=["text", "label", "src"]
    ).copy()

    frame["text"] = frame["text"].astype(str)
    frame["src"] = frame["src"].astype(str)
    frame["label"] = frame["label"].astype(int)

    frame["domain"] = (
        frame["src"].str.lower().str.split("_").str[0]
    )

    def classify_source(src):

        src = src.lower()

        if src in {
            "yelp_human",
            "squad_human",
            "eli5_human",
        }:
            return "human"

        if "_machine_" in src:
            return "machine"

        return "other"

    frame["source_type"] = frame["src"].apply(
        classify_source
    )

    def get_generator_config(src):

        parts = src.lower().split("_")

        if "machine" not in parts:
            return None

        index = parts.index("machine")

        return "_".join(parts[index + 1:])

    frame["generator_config"] = frame["src"].apply(
        get_generator_config
    )

    # Verify the known source-label mapping.
    expected = frame["source_type"].map({
        "human": 1,
        "machine": 0,
    })

    known = expected.notna()

    mismatch = (
        known
        & frame["label"].ne(expected)
    )

    if mismatch.any():
        raise ValueError(
            "Unexpected source-label mismatch. "
            "Inspect the affected source categories."
        )

    frame = frame[
        frame["source_type"].isin(["human", "machine"])
    ].copy()

    return frame.drop_duplicates(
        subset=["text"]
    ).copy()


train = prepare(train)


# --------------------------------------------------
# 4. Load the same fixed ELI5 test sample
# --------------------------------------------------

print("Loading fixed ELI5 test sample...")

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found: {TEST_FILE}"
    )

test = pd.read_csv(TEST_FILE)

required_columns = {
    "text", "label", "src"
}

if not required_columns.issubset(test.columns):
    raise ValueError(
        "The saved ELI5 test file is missing required columns."
    )

test = test.dropna(
    subset=["text", "label", "src"]
).copy()

test["text"] = test["text"].astype(str)
test["label"] = test["label"].astype(int)
test["src"] = test["src"].astype(str)

if len(test) != 1000:
    raise ValueError(
        f"Expected 1000 fixed test records, found {len(test)}."
    )

if test["label"].value_counts().to_dict() != {0: 500, 1: 500}:
    raise ValueError(
        "The fixed test sample must contain 500 examples per class."
    )

if not test["src"].str.startswith("eli5_").all():
    raise ValueError("The fixed test sample contains non-ELI5 sources.")

if test["text"].duplicated().any():
    raise ValueError("Duplicate text detected in the fixed test set.")

print("Fixed test size:", len(test))
print("Test label counts:")
print(test["label"].value_counts().sort_index())


# --------------------------------------------------
# 5. Reconstruct the training pools
# --------------------------------------------------

def machine_configs(frame, domain):

    return set(
        frame.loc[
            (frame["domain"] == domain)
            & (frame["source_type"] == "machine")
            & (frame["label"] == 0),
            "generator_config"
        ].dropna()
    )


# Maintain the same common-generator restriction
# as the original adaptation experiment.

def extract_generator_config(src):
    parts = str(src).lower().split("_")

    if "machine" not in parts:
        return None

    index = parts.index("machine")
    return "_".join(parts[index + 1:])


test["generator_config"] = test["src"].apply(
    extract_generator_config
)

common_configs = (
    machine_configs(train, "yelp")
    & machine_configs(train, "squad")
    & machine_configs(train, "eli5")
    & set(
        test.loc[
            test["label"] == 0,
            "generator_config"
        ].dropna()
    )
)

# Verify that common configurations were found.
# The original experiment used 27.
print("\nCommon generator configurations:", len(common_configs))

if len(common_configs) == 0:
    raise ValueError(
        "No common generator configurations found. "
        "Inspect source parsing before continuing."
    )


def get_pools(domain):

    domain_data = train[
        train["domain"] == domain
    ]

    human = domain_data[
        (domain_data["source_type"] == "human")
        & (domain_data["label"] == 1)
    ].copy()

    machine = domain_data[
        (domain_data["source_type"] == "machine")
        & (domain_data["label"] == 0)
        & (domain_data["generator_config"].isin(common_configs))
    ].copy()

    return human, machine


yelp_human, yelp_machine = get_pools("yelp")
squad_human, squad_machine = get_pools("squad")

# Ensure no exact test-text overlap with the training data.
test_texts = set(test["text"])

for name, pool in [
    ("Yelp human", yelp_human),
    ("Yelp machine", yelp_machine),
    ("SQuAD human", squad_human),
    ("SQuAD machine", squad_machine),
]:
    pool.drop(
        pool[pool["text"].isin(test_texts)].index,
        inplace=True
    )

    print(f"{name}: {len(pool)} records available")


# --------------------------------------------------
# 6. Build the two model strategies
# --------------------------------------------------

def create_training_sets(seed):

    # First, select a Yelp-only sample.
    yelp_h = yelp_human.sample(
        n=N_TRAIN_PER_CLASS,
        random_state=seed
    )

    yelp_m = yelp_machine.sample(
        n=N_TRAIN_PER_CLASS,
        random_state=seed + 1
    )

    single_domain = pd.concat(
        [yelp_h, yelp_m],
        ignore_index=True
    ).sample(
        frac=1,
        random_state=seed + 10
    ).reset_index(drop=True)

    # Select half of the Yelp examples for the
    # domain-diverse strategy.
    diverse_yelp_h = yelp_h.iloc[:1000].copy()
    diverse_yelp_m = yelp_m.iloc[:1000].copy()

    # Add SQuAD examples.
    squad_h = squad_human.sample(
        n=1000,
        random_state=seed + 2
    )

    squad_m = squad_machine.sample(
        n=1000,
        random_state=seed + 3
    )

    domain_diverse = pd.concat(
        [
            diverse_yelp_h,
            diverse_yelp_m,
            squad_h,
            squad_m,
        ],
        ignore_index=True
    ).sample(
        frac=1,
        random_state=seed + 11
    ).reset_index(drop=True)

    return single_domain, domain_diverse


# --------------------------------------------------
# 7. Define the ML model
# --------------------------------------------------

def make_model():

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
                random_state=42
            )
        )
    ])


# --------------------------------------------------
# 8. Evaluate models on the same test examples
# --------------------------------------------------

def evaluate(model, strategy, seed):

    y_true = test["label"]

    predictions = model.predict(test["text"])

    probabilities = model.predict_proba(test["text"])

    ai_index = list(model.classes_).index(0)

    ai_scores = probabilities[:, ai_index]

    # Matrix rows and columns: AI=0, Human=1.
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    human_total = matrix[1].sum()

    human_fpr = (
        matrix[1, 0] / human_total
        if human_total else float("nan")
    )

    results = {
        "seed": seed,
        "strategy": strategy,
        "test_records": len(test),
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "ai_precision": precision_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0
        ),
        "ai_recall": recall_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0
        ),
        "ai_f1": f1_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0
        ),
        "human_false_positive_rate": human_fpr,
        "ai_roc_auc": roc_auc_score(
            (y_true == 0).astype(int),
            ai_scores
        ),
    }

    matrix_results = {
        "seed": seed,
        "strategy": strategy,
        "true_ai_pred_ai": int(matrix[0, 0]),
        "true_ai_pred_human": int(matrix[0, 1]),
        "true_human_pred_ai": int(matrix[1, 0]),
        "true_human_pred_human": int(matrix[1, 1]),
    }

    return results, matrix_results


# --------------------------------------------------
# 9. Run all experiments
# --------------------------------------------------

all_metrics = []
all_matrices = []

for seed in SEEDS:

    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT WITH SEED {seed}")
    print("=" * 60)

    single_train, diverse_train = create_training_sets(seed)

    print("\nTraining sample sizes:")
    print("Yelp-only:", len(single_train))
    print("Yelp + SQuAD:", len(diverse_train))

    assert len(single_train) == 4000
    assert len(diverse_train) == 4000

    # Train single-domain model.
    print("\nTraining Yelp-only model...")

    single_model = make_model()

    single_model.fit(
        single_train["text"],
        single_train["label"]
    )

    single_results, single_matrix = evaluate(
        single_model,
        "Yelp-only",
        seed
    )

    # Train domain-diverse model.
    print("Training domain-diverse model...")

    diverse_model = make_model()

    diverse_model.fit(
        diverse_train["text"],
        diverse_train["label"]
    )

    diverse_results, diverse_matrix = evaluate(
        diverse_model,
        "Yelp + SQuAD",
        seed
    )

    all_metrics.extend([
        single_results,
        diverse_results
    ])

    all_matrices.extend([
        single_matrix,
        diverse_matrix
    ])

    print("\nRun results:")
    print(pd.DataFrame([
        single_results,
        diverse_results
    ]).to_string(index=False))


# --------------------------------------------------
# 10. Save individual runs
# --------------------------------------------------

results_df = pd.DataFrame(all_metrics)

matrices_df = pd.DataFrame(all_matrices)

results_df.to_csv(
    OUTPUT_DIR / "adaptation_repeat_runs.csv",
    index=False
)

matrices_df.to_csv(
    OUTPUT_DIR / "adaptation_repeat_confusion_matrices.csv",
    index=False
)


# --------------------------------------------------
# 11. Calculate mean and standard deviation
# --------------------------------------------------

metric_columns = [
    "accuracy",
    "ai_precision",
    "ai_recall",
    "ai_f1",
    "human_false_positive_rate",
    "ai_roc_auc",
]

summary_rows = []

for strategy, group in results_df.groupby("strategy"):

    row = {"strategy": strategy}

    for metric in metric_columns:

        row[f"{metric}_mean"] = group[metric].mean()

        row[f"{metric}_std"] = group[metric].std(ddof=1)

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "adaptation_repeat_summary.csv",
    index=False
)


# --------------------------------------------------
# 12. Display final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("ALL EXPERIMENT RUNS COMPLETED")
print("=" * 60)

print("\nINDIVIDUAL RESULTS")
print(results_df.to_string(index=False))

print("\nMEAN AND STANDARD DEVIATION ACROSS THREE RUNS")
print(summary_df.round(4).to_string(index=False))

print("\nFILES SAVED:")
print(OUTPUT_DIR / "adaptation_repeat_runs.csv")
print(OUTPUT_DIR / "adaptation_repeat_confusion_matrices.csv")
print(OUTPUT_DIR / "adaptation_repeat_summary.csv")
