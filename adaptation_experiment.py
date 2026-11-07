
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

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_TRAIN_PER_CLASS = 2000
N_TEST_PER_CLASS = 500

print("Loading official MAGE train split...")
train = load_dataset(
    "yaful/MAGE", split="train"
).select_columns(["text", "label", "src"]).to_pandas()

print("Loading official MAGE test split...")
test = load_dataset(
    "yaful/MAGE", split="test"
).select_columns(["text", "label", "src"]).to_pandas()




def prepare(frame, split_name):
    frame = frame.dropna(
        subset=["text", "label", "src"]
    ).copy()

    frame["text"] = frame["text"].astype(str)
    frame["src"] = frame["src"].astype(str)
    frame["label"] = frame["label"].astype(int)

    # Extract the candidate domain from the source name.
    frame["domain"] = (
        frame["src"].str.lower().str.split("_").str[0]
    )

    # This experiment uses only these three domains.
    frame = frame[
        frame["domain"].isin(["yelp", "squad", "eli5"])
    ].copy()

    def classify_source(src):
        src = src.lower()

        # Match the complete human-source name.
        if src in {
            "yelp_human",
            "squad_human",
            "eli5_human",
        }:
            return "human"

        # Machine-generated source categories.
        if "_machine_" in src:
            return "machine"

        # Do not infer the class from the word "human"
        # in special names such as *_human_para.
        return "other"

    frame["source_type"] = frame["src"].apply(
        classify_source
    )

    def get_config(src):
        parts = src.lower().split("_")

        if "machine" not in parts:
            return None

        index = parts.index("machine")
        return "_".join(parts[index + 1:])

    frame["generator_config"] = frame["src"].apply(
        get_config
    )

    # Check labels only for source types explicitly recognized
    # by this experiment's source-name rules.
    expected = frame["source_type"].map({
        "human": 1,
        "machine": 0,
    })

    known = expected.notna()
    mismatch_mask = known & frame["label"].ne(expected)

    if mismatch_mask.any():
        mismatches = frame.loc[
            mismatch_mask,
            ["src", "label", "source_type"]
        ]

        print(f"\nMismatches in {split_name}:")
        print(
            mismatches.groupby(
                ["src", "label", "source_type"]
            ).size().to_string()
        )

        raise ValueError(
            f"Unexpected source-label mismatch in {split_name}. "
            "Inspect the affected sources before continuing."
        )

    # Remove unknown source types from this experiment.
    frame = frame[
        frame["source_type"].isin(["human", "machine"])
    ].copy()

    return frame.drop_duplicates(
        subset=["text"]
    ).copy()




train = prepare(train, "train")
test = prepare(test, "test")
# Get generator configurations present in each required group.
def machine_configs(frame, domain):
    return set(
        frame.loc[
            (frame["domain"] == domain)
            & (frame["source_type"] == "machine")
            & (frame["label"] == 0),
            "generator_config"
        ].dropna()
    )


common_configs = (
    machine_configs(train, "yelp")
    & machine_configs(train, "squad")
    & machine_configs(train, "eli5")
    & machine_configs(test, "eli5")
)

print("\nCommon generator configurations:", len(common_configs))

if len(common_configs) < 5:
    raise ValueError(
        "Too few common generator configurations. "
        "Inspect the source names before proceeding."
    )

# Keep only sources in the common configuration set for
# the AI records. Human records are selected by domain.
def domain_pools(frame, domain):
    domain_df = frame[frame["domain"] == domain]

    human = domain_df[
        (domain_df["source_type"] == "human")
        & (domain_df["label"] == 1)
    ].copy()

    machine = domain_df[
        (domain_df["source_type"] == "machine")
        & (domain_df["label"] == 0)
        & (domain_df["generator_config"].isin(common_configs))
    ].copy()

    return human, machine


yelp_human, yelp_machine = domain_pools(train, "yelp")
squad_human, squad_machine = domain_pools(train, "squad")
eli5_human, eli5_machine = domain_pools(test, "eli5")

# Prevent exact text overlap with ANY training-split text.
all_train_texts = set(train["text"])

eli5_human = eli5_human[
    ~eli5_human["text"].isin(all_train_texts)
].copy()

eli5_machine = eli5_machine[
    ~eli5_machine["text"].isin(all_train_texts)
].copy()

print("\nAVAILABLE POOL SIZES")
for name, pool in [
    ("Yelp human", yelp_human),
    ("Yelp machine", yelp_machine),
    ("SQuAD human", squad_human),
    ("SQuAD machine", squad_machine),
    ("ELI5 human test", eli5_human),
    ("ELI5 machine test", eli5_machine),
]:
    print(f"{name}: {len(pool)}")

if min(
    len(yelp_human),
    len(yelp_machine),
    len(squad_human),
    len(squad_machine)
) < N_TRAIN_PER_CLASS:
    raise ValueError(
        "Insufficient training data. Adjust sampling sizes."
    )

if min(len(eli5_human), len(eli5_machine)) < N_TEST_PER_CLASS:
    raise ValueError(
        "Insufficient ELI5 test data. Adjust test sample size."
    )

# A. Single-domain training: Yelp only.
single_train = pd.concat([
    yelp_human.sample(
        n=N_TRAIN_PER_CLASS, random_state=SEED
    ),
    yelp_machine.sample(
        n=N_TRAIN_PER_CLASS, random_state=SEED + 1
    ),
], ignore_index=True).sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

# B. Domain-diverse training: equal class/domain budgets.
diverse_train = pd.concat([
    yelp_human.sample(
        n=N_TRAIN_PER_CLASS // 2, random_state=SEED
    ),
    yelp_machine.sample(
        n=N_TRAIN_PER_CLASS // 2, random_state=SEED + 1
    ),
    squad_human.sample(
        n=N_TRAIN_PER_CLASS // 2, random_state=SEED + 2
    ),
    squad_machine.sample(
        n=N_TRAIN_PER_CLASS // 2, random_state=SEED + 3
    ),
], ignore_index=True).sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

# Use the same untouched target-domain test examples for both models.
target_test = pd.concat([
    eli5_human.sample(
        n=N_TEST_PER_CLASS, random_state=SEED + 4
    ),
    eli5_machine.sample(
        n=N_TEST_PER_CLASS, random_state=SEED + 5
    ),
], ignore_index=True).sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

for name, frame in [
    ("Single-domain training", single_train),
    ("Domain-diverse training", diverse_train),
    ("ELI5 test", target_test),
]:
    print(f"\n{name}: {len(frame)} records")
    print(frame["label"].value_counts().sort_index())

# Save the exact experiment datasets.
single_train.to_csv(
    OUTPUT_DIR / "adaptation_single_domain_train.csv",
    index=False
)
diverse_train.to_csv(
    OUTPUT_DIR / "adaptation_domain_diverse_train.csv",
    index=False
)
target_test.to_csv(
    OUTPUT_DIR / "adaptation_eli5_test.csv",
    index=False
)


def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=SEED,
        )),
    ])


def evaluate(name, model):
    y_true = target_test["label"]
    y_pred = model.predict(target_test["text"])

    probabilities = model.predict_proba(target_test["text"])
    ai_column = list(model.classes_).index(0)
    ai_scores = probabilities[:, ai_column]

    matrix = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    human_count = matrix[1].sum()
    human_fpr = (
        matrix[1, 0] / human_count
        if human_count else float("nan")
    )

    return {
        "training_strategy": name,
        "test_records": len(target_test),
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
        ),
    }, matrix


print("\nTraining single-domain model...")
single_model = make_model()
single_model.fit(single_train["text"], single_train["label"])

print("Training domain-diverse model...")
diverse_model = make_model()
diverse_model.fit(diverse_train["text"], diverse_train["label"])

single_results, single_matrix = evaluate(
    "Single-domain Yelp", single_model
)
diverse_results, diverse_matrix = evaluate(
    "Domain-diverse Yelp + SQuAD", diverse_model
)

results = pd.DataFrame([single_results, diverse_results])
results.to_csv(
    OUTPUT_DIR / "adaptation_metrics.csv", index=False
)

joblib.dump(
    single_model, OUTPUT_DIR / "adaptation_single_model.joblib"
)
joblib.dump(
    diverse_model, OUTPUT_DIR / "adaptation_diverse_model.joblib"
)

print("\nSINGLE-DOMAIN CONFUSION MATRIX [AI=0, Human=1]")
print(single_matrix)

print("\nDOMAIN-DIVERSE CONFUSION MATRIX [AI=0, Human=1]")
print(diverse_matrix)

print("\nADAPTATION RESULTS")
print(results.to_string(index=False))

print("\nSaved adaptation datasets, metrics, and models.")
