
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
    classification_report,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
TARGET_MODEL = "text-davinci-003"

N_TRAIN_HUMAN = 2000
N_TRAIN_MACHINE = 2000
N_TEST_HUMAN = 90
N_TEST_MACHINE = 90


# --------------------------------------------------
# 1. Load the official train and test splits
# --------------------------------------------------
print("Loading MAGE train split...")
train_df = load_dataset(
    "yaful/MAGE", split="train"
).select_columns(["text", "label", "src"]).to_pandas()

print("Loading MAGE test split...")
test_df = load_dataset(
    "yaful/MAGE", split="test"
).select_columns(["text", "label", "src"]).to_pandas()


def prepare_frame(frame):
    frame = frame.dropna(
        subset=["text", "label", "src"]
    ).copy()

    frame["text"] = frame["text"].astype(str)
    frame["src"] = frame["src"].astype(str)
    frame["label"] = frame["label"].astype(int)

    # Keep the Yelp domain for this experiment.
    frame = frame[
        frame["src"].str.startswith("yelp_")
    ].copy()

    # Remove duplicate texts within this split.
    frame = frame.drop_duplicates(
        subset=["text"]
    ).copy()

    return frame


train_df = prepare_frame(train_df)
test_df = prepare_frame(test_df)

# Human and machine source definitions follow the
# documented MAGE src naming convention.
human_train_pool = train_df[
    (train_df["src"] == "yelp_human")
    & (train_df["label"] == 1)
].copy()

machine_train_pool = train_df[
    train_df["src"].str.startswith("yelp_machine_")
    & (train_df["label"] == 0)
].copy()

# Exclude the target generator from every training source.
heldout_mask_train = machine_train_pool["src"].str.contains(
    TARGET_MODEL, case=False, regex=False
)

heldout_train_count = int(heldout_mask_train.sum())

machine_train_pool = machine_train_pool[
    ~heldout_mask_train
].copy()

print("\nHELD-OUT MODEL CHECK")
print("Target:", TARGET_MODEL)
print("Target-model records excluded from training:",
      heldout_train_count)

print("\nAVAILABLE TRAINING RECORDS")
print("Yelp human:", len(human_train_pool))
print("Yelp machine from other sources:", len(machine_train_pool))

# Sample training data reproducibly.
if (
    len(human_train_pool) < N_TRAIN_HUMAN
    or len(machine_train_pool) < N_TRAIN_MACHINE
):
    raise ValueError(
        "Not enough training records for the requested sample."
    )

human_train = human_train_pool.sample(
    n=N_TRAIN_HUMAN, random_state=SEED
)

machine_train = machine_train_pool.sample(
    n=N_TRAIN_MACHINE, random_state=SEED + 1
)

train_sample = pd.concat(
    [human_train, machine_train],
    ignore_index=True
).sample(frac=1, random_state=SEED).reset_index(drop=True)

train_texts = set(train_sample["text"])

# --------------------------------------------------
# 2. Create test pools from the official test split
# --------------------------------------------------
human_test_pool = test_df[
    (test_df["src"] == "yelp_human")
    & (test_df["label"] == 1)
].copy()

machine_test_pool = test_df[
    test_df["src"].str.startswith("yelp_machine_")
    & (test_df["label"] == 0)
].copy()

# The held-out model must not appear in training.
heldout_machine_test_pool = machine_test_pool[
    machine_test_pool["src"].str.contains(
        TARGET_MODEL, case=False, regex=False
    )
].copy()

# Reference AI examples must come from source categories
# represented in the training sample.
seen_sources = set(machine_train["src"])

seen_machine_test_pool = machine_test_pool[
    machine_test_pool["src"].isin(seen_sources)
    & ~machine_test_pool["src"].str.contains(
        TARGET_MODEL, case=False, regex=False
    )
].copy()

# Guard against exact-text leakage across train and test.
human_test_pool = human_test_pool[
    ~human_test_pool["text"].isin(train_texts)
].copy()

seen_machine_test_pool = seen_machine_test_pool[
    ~seen_machine_test_pool["text"].isin(train_texts)
].copy()

heldout_machine_test_pool = heldout_machine_test_pool[
    ~heldout_machine_test_pool["text"].isin(train_texts)
].copy()

print("\nAVAILABLE TEST RECORDS")
print("Human Yelp:", len(human_test_pool))
print("Seen-source AI Yelp:", len(seen_machine_test_pool))
print("Held-out-model AI Yelp:", len(heldout_machine_test_pool))

# Use the same human examples for both comparisons.
# This makes the human reference identical across the two tests.
if (
    len(human_test_pool) < N_TEST_HUMAN
    or len(seen_machine_test_pool) < N_TEST_MACHINE
    or len(heldout_machine_test_pool) < N_TEST_MACHINE
):
    raise ValueError(
        "Insufficient official test records for one or more groups. "
        "Check the available counts and reduce sample sizes if needed."
    )

human_test = human_test_pool.sample(
    n=N_TEST_HUMAN, random_state=SEED + 2
)

seen_machine_test = seen_machine_test_pool.sample(
    n=N_TEST_MACHINE, random_state=SEED + 3
)

heldout_machine_test = heldout_machine_test_pool.sample(
    n=N_TEST_MACHINE, random_state=SEED + 4
)

seen_test = pd.concat(
    [human_test, seen_machine_test],
    ignore_index=True
).sample(frac=1, random_state=SEED).reset_index(drop=True)

unseen_test = pd.concat(
    [human_test, heldout_machine_test],
    ignore_index=True
).sample(frac=1, random_state=SEED + 1).reset_index(drop=True)

# Save the exact datasets used for this experiment.
train_sample.to_csv(
    OUTPUT_DIR / "cross_model_train.csv", index=False
)

seen_test.to_csv(
    OUTPUT_DIR / "cross_model_seen_test.csv", index=False
)

unseen_test.to_csv(
    OUTPUT_DIR / "cross_model_unseen_test.csv", index=False
)

print("\nDATASET SIZES")
print("Training:", len(train_sample))
print("Seen-model reference test:", len(seen_test))
print("Unseen-model test:", len(unseen_test))

print("\nTRAINING LABEL COUNTS")
print(train_sample["label"].value_counts().sort_index())

print("\nSEEN TEST LABEL COUNTS")
print(seen_test["label"].value_counts().sort_index())

print("\nUNSEEN TEST LABEL COUNTS")
print(unseen_test["label"].value_counts().sort_index())

print("\nTRAINING MODEL SOURCES")
print(sorted(machine_train["src"].unique()))

assert not train_sample["src"].str.contains(
    TARGET_MODEL, case=False, regex=False
).any(), "Held-out model leaked into training!"

assert not (
    set(train_sample["text"]) & set(seen_test["text"])
), "Exact text overlap in seen-model test!"

assert not (
    set(train_sample["text"]) & set(unseen_test["text"])
), "Exact text overlap in unseen-model test!"


# --------------------------------------------------
# 3. Train TF-IDF + Logistic Regression
# --------------------------------------------------
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=SEED,
        )
    )
])

print("\nTraining cross-model baseline...")
model.fit(train_sample["text"], train_sample["label"])


# --------------------------------------------------
# 4. Evaluate both test conditions
# --------------------------------------------------
def evaluate(name, frame):
    y_true = frame["label"]
    y_pred = model.predict(frame["text"])

    probabilities = model.predict_proba(frame["text"])
    ai_index = list(model.classes_).index(0)
    ai_scores = probabilities[:, ai_index]

    # Matrix rows/columns are ordered [AI=0, Human=1].
    matrix = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    human_count = matrix[1].sum()
    human_fpr = (
        matrix[1, 0] / human_count
        if human_count else float("nan")
    )

    results = {
        "evaluation": name,
        "records": len(frame),
        "accuracy": accuracy_score(y_true, y_pred),
        "ai_precision": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "ai_recall": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "ai_f1": f1_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "human_false_positive_rate": human_fpr,
        "ai_roc_auc": roc_auc_score(
            (y_true == 0).astype(int), ai_scores
        )
    }

    print(f"\n{'=' * 12} {name} {'=' * 12}")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["AI-generated (0)", "Human-written (1)"],
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix [AI=0, Human=1]:")
    print(matrix)
    print("Human false-positive rate:", round(human_fpr, 4))

    return results


seen_results = evaluate(
    "Seen-model reference", seen_test
)

unseen_results = evaluate(
    "Held-out text-davinci-003", unseen_test
)

results = pd.DataFrame([seen_results, unseen_results])

results.to_csv(
    OUTPUT_DIR / "cross_model_metrics.csv",
    index=False
)

joblib.dump(
    model,
    OUTPUT_DIR / "cross_model_tfidf_logreg.joblib"
)

print("\nCROSS-MODEL METRICS SUMMARY")
print(results.to_string(index=False))

print("\nSaved cross-model datasets, metrics, and model.")
