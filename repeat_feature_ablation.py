
from pathlib import Path
import gc

import numpy as np
import pandas as pd

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
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "milestone2" / "feature_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
TRAINING_FRACTION = 0.90

EXPERIMENTS = {
    "Unseen domain": {
        "folder": "unseen_domain_eli5",
        "tests": {
            "In-domain reference": "test_in_domain.csv",
            "Unseen ELI5": "test_unseen_domain.csv",
        },
    },
    "Unseen model family": {
        "folder": "unseen_model_OPT",
        "tests": {
            "Seen model families": "test_seen_family.csv",
            "Unseen OPT family": "test_unseen_OPT_family.csv",
        },
    },
}


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = {"text", "label", "domain"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["text", "label", "domain"]).copy()

    df["text"] = df["text"].astype(str)
    df["domain"] = df["domain"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["text"].str.strip() != ""]

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"{path.name} must contain both labels 0 and 1.")

    return df


def sample_training_data(train, seed):
    """Sample 90% separately within each domain and class."""

    rng = np.random.default_rng(seed)
    sampled_groups = []

    for _, group in train.groupby(["domain", "label"]):
        n = max(1, int(round(len(group) * TRAINING_FRACTION)))

        group_seed = int(rng.integers(0, 2**32 - 1))

        sampled_groups.append(
            group.sample(n=n, replace=False, random_state=group_seed)
        )

    sampled = pd.concat(sampled_groups, ignore_index=True)

    return sampled.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)


# --------------------------------------------------
# FEATURE REPRESENTATIONS
# --------------------------------------------------

def create_vectorizer(name):

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    if name == "Word TF-IDF":
        return word

    if name == "Character TF-IDF":
        return char

    if name == "Combined TF-IDF":
        return FeatureUnion([
            ("word", word),
            ("char", char),
        ])

    raise ValueError(f"Unknown feature representation: {name}")


# --------------------------------------------------
# THRESHOLD AND EVALUATION
# --------------------------------------------------

def select_threshold(validation_labels, validation_scores):

    # MAGE label 0 = AI-generated.
    y_validation = (validation_labels == 0).astype(int)

    fpr, tpr, thresholds = roc_curve(
        y_validation,
        validation_scores,
    )

    balanced_accuracy = (tpr + 1 - fpr) / 2

    return float(thresholds[np.argmax(balanced_accuracy)])


def evaluate(test_labels, scores, threshold):

    # AI-generated is the positive class.
    actual = (test_labels == 0).astype(int)
    predicted = (scores >= threshold).astype(int)

    human_mask = actual == 0

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
        "human_false_positive_rate": (
            (predicted[human_mask] == 1).mean()
            if human_mask.any() else np.nan
        ),
        "ai_roc_auc": roc_auc_score(actual, scores),
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MILESTONE 6: REPEATED FEATURE ABLATION")
    print("=" * 65)

    feature_names = [
        "Word TF-IDF",
        "Character TF-IDF",
        "Combined TF-IDF",
    ]

    all_results = []

    for experiment, config in EXPERIMENTS.items():

        print("\n" + "=" * 65)
        print(experiment.upper())
        print("=" * 65)

        folder = (
            BASE_DIR / "outputs" / "milestone2" / config["folder"]
        )

        train = load_data(folder / "train.csv")
        validation = load_data(folder / "validation.csv")

        test_sets = {
            condition: load_data(folder / filename)
            for condition, filename in config["tests"].items()
        }

        print(f"Original training records: {len(train)}")
        print(f"Fixed validation records: {len(validation)}")

        for condition, test in test_sets.items():
            print(f"{condition} test records: {len(test)}")

        for seed in SEEDS:

            print("\n" + "-" * 65)
            print(f"TRAINING SEED: {seed}")
            print("-" * 65)

            sampled_train = sample_training_data(train, seed)

            # Check that sampling has not introduced split overlap.
            train_texts = set(sampled_train["text"])
            validation_texts = set(validation["text"])

            if train_texts.intersection(validation_texts):
                raise ValueError(
                    "Training/validation text overlap detected."
                )

            for condition, test in test_sets.items():
                if train_texts.intersection(set(test["text"])):
                    raise ValueError(
                        f"Training/test overlap detected: {condition}"
                    )

            print(f"Sampled training records: {len(sampled_train)}")
            print("Exact text overlap checks: PASSED")

            for feature_name in feature_names:

                print(f"\nFeature representation: {feature_name}")

                vectorizer = create_vectorizer(feature_name)

                X_train = vectorizer.fit_transform(
                    sampled_train["text"].tolist()
                )

                X_validation = vectorizer.transform(
                    validation["text"].tolist()
                )

                model = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                )

                model.fit(X_train, sampled_train["label"])

                # AI class is MAGE label 0.
                ai_index = list(model.classes_).index(0)

                validation_scores = model.predict_proba(
                    X_validation
                )[:, ai_index]

                threshold = select_threshold(
                    validation["label"].to_numpy(),
                    validation_scores,
                )

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
                        "experiment": experiment,
                        "condition": condition,
                        "seed": seed,
                        "vectorizer": feature_name,
                        "threshold": threshold,
                        "training_records": len(sampled_train),
                        **result,
                    }

                    all_results.append(row)

                    print(
                        f"{condition}: "
                        f"Accuracy={result['accuracy']:.4f}, "
                        f"AI F1={result['ai_f1']:.4f}, "
                        f"FPR={result['human_false_positive_rate']:.4f}, "
                        f"AUC={result['ai_roc_auc']:.4f}"
                    )

                del vectorizer, model
                del X_train, X_validation
                gc.collect()

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    runs_path = OUTPUT_DIR / "feature_ablation_repeat_runs.csv"
    results_df.to_csv(runs_path, index=False)

    metric_columns = [
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
        .groupby(["experiment", "condition", "vectorizer"])[
            metric_columns
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "feature_ablation_repeat_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("MILESTONE 6: REPEATABILITY SUMMARY")
    print("=" * 65)

    print(
        summary[
            [
                ("experiment", ""),
                ("condition", ""),
                ("vectorizer", ""),
                ("accuracy", "mean"),
                ("accuracy", "std"),
                ("ai_f1", "mean"),
                ("ai_f1", "std"),
                ("human_false_positive_rate", "mean"),
                ("ai_roc_auc", "mean"),
                ("ai_roc_auc", "std"),
            ]
        ].to_string(index=False)
    )

    print("\nFiles saved:")
    print(runs_path)
    print(summary_path)


if __name__ == "__main__":
    main()
