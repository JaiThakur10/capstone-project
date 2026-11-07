
from pathlib import Path
import gc

import numpy as np
import pandas as pd

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR / "outputs" / "milestone2" / "adaptation_combined"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]

STANDARD_DOMAINS = [
    "cmv",
    "yelp",
    "xsum",
    "tldr",
    "eli5",
    "wp",
    "roct",
    "hswag",
    "squad",
    "sci_gen",
]

NINE_TRAIN_DOMAINS = [
    domain
    for domain in STANDARD_DOMAINS
    if domain != "eli5"
]

# Each strategy uses exactly 3,600 records.
STRATEGIES = {
    "Yelp-only": {
        "yelp": {0: 1800, 1: 1800}
    },
    "Yelp+SQuAD": {
        "yelp": {0: 900, 1: 900},
        "squad": {0: 900, 1: 900},
    },
    "Nine-domain": {
        domain: {0: 200, 1: 200}
        for domain in NINE_TRAIN_DOMAINS
    },
}


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["text"].str.strip() != ""]

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path.name}")

    return df


def identify_domain(src):
    """Identify a standard MAGE domain from its source name."""

    src = str(src)

    for domain in sorted(
        STANDARD_DOMAINS,
        key=len,
        reverse=True
    ):
        if src.startswith(domain + "_"):
            return domain

    return None


def load_official_training_data(reserved_texts):

    print("\nLoading official MAGE training split...")

    dataset = load_dataset(
        "yaful/MAGE",
        split="train"
    )

    # Keep only the fields needed for this experiment.
    df = pd.DataFrame({
        "text": dataset["text"],
        "label": dataset["label"],
        "src": dataset["src"],
    })

    df = df.dropna(subset=["text", "label", "src"]).copy()

    df["text"] = df["text"].astype(str)
    df["src"] = df["src"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["text"].str.strip() != ""]

    df["domain"] = df["src"].map(identify_domain)

    # Retain only the standard MAGE domains.
    df = df[df["domain"].isin(STANDARD_DOMAINS)].copy()

    # Remove any training text that appears in fixed validation
    # or test data.
    before = len(df)
    df = df[~df["text"].isin(reserved_texts)].copy()
    removed_reserved = before - len(df)

    # Prevent duplicate text records from being selected twice.
    df = df.drop_duplicates(subset=["text"]).copy()

    print(f"Eligible official training records: {len(df)}")
    print(f"Reserved validation/test texts removed: {removed_reserved}")
    print(f"Domains available: {sorted(df['domain'].unique())}")

    return df


# --------------------------------------------------
# TRAINING SAMPLE CONSTRUCTION
# --------------------------------------------------

def build_training_sample(
    official_df,
    strategy_name,
    allocations,
    seed,
):
    rng = np.random.default_rng(seed)

    samples = []
    provenance = []

    for domain, class_allocations in allocations.items():

        for label, required_count in class_allocations.items():

            pool = official_df[
                (official_df["domain"] == domain)
                & (official_df["label"] == label)
            ]

            if len(pool) < required_count:
                raise ValueError(
                    f"Not enough records for {strategy_name}: "
                    f"{domain}, label={label}. "
                    f"Required={required_count}, available={len(pool)}"
                )

            sample_seed = int(
                rng.integers(0, 2**32 - 1)
            )

            selected = pool.sample(
                n=required_count,
                replace=False,
                random_state=sample_seed,
            )

            samples.append(selected)

            provenance.append({
                "seed": seed,
                "strategy": strategy_name,
                "domain": domain,
                "label": label,
                "records_selected": len(selected),
                "source_categories": selected["src"].nunique(),
            })

    training = pd.concat(samples, ignore_index=True)

    training = training.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    if len(training) != 3600:
        raise ValueError(
            f"Expected 3600 training examples, got {len(training)}"
        )

    if training["label"].value_counts().to_dict() != {0: 1800, 1: 1800}:
        raise ValueError("Training sample is not class-balanced.")

    return training, provenance


# --------------------------------------------------
# COMBINED TF-IDF MODEL
# --------------------------------------------------

def create_combined_vectorizer():

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    character_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    return FeatureUnion([
        ("word", word_vectorizer),
        ("char", character_vectorizer),
    ])


# --------------------------------------------------
# VALIDATION THRESHOLD
# --------------------------------------------------

def select_threshold(validation_labels, validation_scores):

    # MAGE label 0 = AI-generated.
    y_validation = (
        validation_labels == 0
    ).astype(int)

    fpr, tpr, thresholds = roc_curve(
        y_validation,
        validation_scores,
    )

    balanced_accuracy = (
        tpr + 1 - fpr
    ) / 2

    best_index = int(np.argmax(balanced_accuracy))

    return float(thresholds[best_index])


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(
    test_labels,
    scores,
    threshold,
):
    # AI-generated is the positive class.
    actual = (test_labels == 0).astype(int)

    predicted = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    human_fpr = (
        cm[0, 1] / (cm[0, 0] + cm[0, 1])
        if (cm[0, 0] + cm[0, 1]) > 0
        else float("nan")
    )

    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "ai_precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "ai_recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "ai_f1": f1_score(
            actual, predicted, zero_division=0
        ),
        "human_false_positive_rate": human_fpr,
        "ai_roc_auc": roc_auc_score(actual, scores),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MILESTONE 7: ADAPTATION WITH COMBINED TF-IDF")
    print("=" * 65)

    test_dir = (
        BASE_DIR / "outputs" / "milestone2"
        / "unseen_domain_eli5"
    )

    validation = load_csv(test_dir / "validation.csv")

    test_sets = {
        "In-domain reference": load_csv(
            test_dir / "test_in_domain.csv"
        ),
        "Unseen ELI5": load_csv(
            test_dir / "test_unseen_domain.csv"
        ),
    }

    reserved_texts = set(validation["text"])

    for test in test_sets.values():
        reserved_texts.update(test["text"])

    official_df = load_official_training_data(reserved_texts)

    print("\nFixed datasets:")
    print(f"Validation: {len(validation)}")

    for condition, test in test_sets.items():
        print(f"{condition}: {len(test)}")

    all_results = []
    all_thresholds = []
    all_provenance = []

    validation_texts = validation["text"].tolist()
    validation_labels = validation["label"].to_numpy()

    for strategy_name, allocations in STRATEGIES.items():

        print("\n" + "=" * 65)
        print(f"TRAINING STRATEGY: {strategy_name}")
        print("=" * 65)

        for seed in SEEDS:

            print("\n" + "-" * 60)
            print(f"SEED: {seed}")
            print("-" * 60)

            training, provenance = build_training_sample(
                official_df,
                strategy_name,
                allocations,
                seed,
            )

            all_provenance.extend(provenance)

            # Check for exact overlap with fixed validation and test sets.
            training_texts = set(training["text"])

            if training_texts.intersection(reserved_texts):
                raise ValueError(
                    "Training overlaps with validation/test data."
                )

            print(f"Training records: {len(training)}")
            print("Exact split-overlap checks: PASSED")

            print("Training class counts:")
            print(training["label"].value_counts().sort_index().to_dict())

            # Build and fit the combined-feature model.
            vectorizer = create_combined_vectorizer()

            X_train = vectorizer.fit_transform(
                training["text"].tolist()
            )

            X_validation = vectorizer.transform(
                validation_texts
            )

            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )

            model.fit(
                X_train,
                training["label"].to_numpy(),
            )

            # MAGE label 0 = AI-generated.
            ai_index = list(model.classes_).index(0)

            validation_scores = model.predict_proba(
                X_validation
            )[:, ai_index]

            threshold = select_threshold(
                validation_labels,
                validation_scores,
            )

            all_thresholds.append({
                "strategy": strategy_name,
                "seed": seed,
                "threshold": threshold,
            })

            print(
                f"Validation-selected threshold: {threshold:.4f}"
            )

            for condition, test in test_sets.items():

                X_test = vectorizer.transform(
                    test["text"].tolist()
                )

                test_scores = model.predict_proba(
                    X_test
                )[:, ai_index]

                result = evaluate(
                    test["label"].to_numpy(),
                    test_scores,
                    threshold,
                )

                row = {
                    "strategy": strategy_name,
                    "seed": seed,
                    "condition": condition,
                    "threshold": threshold,
                    "training_records": len(training),
                    **result,
                }

                all_results.append(row)

                print(
                    f"\n{condition}: "
                    f"Accuracy={result['accuracy']:.4f}, "
                    f"AI F1={result['ai_f1']:.4f}, "
                    f"Human FPR="
                    f"{result['human_false_positive_rate']:.4f}, "
                    f"AUC={result['ai_roc_auc']:.4f}"
                )

            del vectorizer, model
            del X_train, X_validation
            gc.collect()

    # --------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)
    thresholds_df = pd.DataFrame(all_thresholds)
    provenance_df = pd.DataFrame(all_provenance)

    results_path = (
        OUTPUT_DIR / "adaptation_combined_repeat_runs.csv"
    )

    thresholds_path = (
        OUTPUT_DIR / "adaptation_combined_thresholds.csv"
    )

    provenance_path = (
        OUTPUT_DIR / "adaptation_combined_training_provenance.csv"
    )

    results_df.to_csv(results_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)
    provenance_df.to_csv(provenance_path, index=False)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "ai_precision",
        "ai_recall",
        "ai_f1",
        "human_false_positive_rate",
        "ai_roc_auc",
    ]

    summary = (
        results_df
        .groupby(["strategy", "condition"])[metrics]
        .agg(["mean", "std"])
    )

    summary_path = (
        OUTPUT_DIR / "adaptation_combined_repeat_summary.csv"
    )

    summary.to_csv(summary_path)

    print("\n" + "=" * 65)
    print("MILESTONE 7: ADAPTATION SUMMARY")
    print("=" * 65)

    print(
        summary[
            [
                ("accuracy", "mean"),
                ("accuracy", "std"),
                ("ai_f1", "mean"),
                ("ai_f1", "std"),
                ("human_false_positive_rate", "mean"),
                ("ai_roc_auc", "mean"),
                ("ai_roc_auc", "std"),
            ]
        ].to_string()
    )

    print("\nFiles saved:")
    print(results_path)
    print(summary_path)
    print(thresholds_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
