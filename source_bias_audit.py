from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "milestone2" / "source_bias"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "Unseen domain": {
        "train": BASE_DIR / "outputs/milestone2/unseen_domain_eli5/train.csv",
        "tests": {
            "In-domain reference": BASE_DIR / "outputs/milestone2/unseen_domain_eli5/test_in_domain.csv",
            "Unseen ELI5": BASE_DIR / "outputs/milestone2/unseen_domain_eli5/test_unseen_domain.csv",
        },
    },
    "Unseen model family": {
        "train": BASE_DIR / "outputs/milestone2/unseen_model_OPT/train.csv",
        "tests": {
            "Seen model families": BASE_DIR / "outputs/milestone2/unseen_model_OPT/test_seen_family.csv",
            "Unseen OPT family": BASE_DIR / "outputs/milestone2/unseen_model_OPT/test_unseen_OPT_family.csv",
        },
    },
}


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    required = {"label", "src"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    if df["label"].isna().any() or df["src"].isna().any():
        raise ValueError(f"Missing label or source values in {path.name}")

    df["label"] = df["label"].astype(int)
    df["src"] = df["src"].astype(str)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path.name}")

    return df


def evaluate_source_only(train, test, experiment, condition):
    # Learn the most common label for each source category.
    source_mapping = train.groupby("src")["label"].agg(
        lambda values: values.mode().iloc[0]
    )

    # Record how consistently each source category maps to one label.
    source_label_counts = train.groupby("src")["label"].nunique()
    mixed_sources = int((source_label_counts > 1).sum())

    # A source not seen during training receives the majority training label.
    fallback_label = int(train["label"].mode().iloc[0])

    known_source = test["src"].isin(source_mapping.index)

    predicted_labels = (
        test["src"]
        .map(source_mapping)
        .fillna(fallback_label)
        .astype(int)
    )

    actual = test["label"].to_numpy()
    predicted = predicted_labels.to_numpy()

    # MAGE labels: 0 = AI-generated, 1 = human-written.
    # Convert to AI-positive form for the detection metrics.
    actual_ai = (actual == 0).astype(int)
    predicted_ai = (predicted == 0).astype(int)

    human_mask = actual == 1
    ai_mask = actual == 0
    known_mask = known_source.to_numpy()
    unknown_mask = ~known_mask

    cm = confusion_matrix(actual_ai, predicted_ai, labels=[0, 1])

    results = {
        "experiment": experiment,
        "condition": condition,
        "train_records": len(train),
        "test_records": len(test),
        "train_source_categories": train["src"].nunique(),
        "test_source_categories": test["src"].nunique(),
        "known_source_coverage": known_mask.mean(),
        "unknown_source_rate": unknown_mask.mean(),
        "mixed_label_train_sources": mixed_sources,
        "fallback_label": fallback_label,
        "accuracy": accuracy_score(actual_ai, predicted_ai),
        "balanced_accuracy": balanced_accuracy_score(actual_ai, predicted_ai),
        "ai_precision": precision_score(
            actual_ai, predicted_ai, zero_division=0
        ),
        "ai_recall": recall_score(
            actual_ai, predicted_ai, zero_division=0
        ),
        "ai_f1": f1_score(
            actual_ai, predicted_ai, zero_division=0
        ),
        "human_false_positive_rate": (
            (predicted[human_mask] == 0).mean()
            if human_mask.any() else float("nan")
        ),
        "known_source_accuracy": (
            accuracy_score(actual[known_mask], predicted[known_mask])
            if known_mask.any() else float("nan")
        ),
        "unknown_source_accuracy": (
            accuracy_score(actual[unknown_mask], predicted[unknown_mask])
            if unknown_mask.any() else float("nan")
        ),
        "tn_human": int(cm[0, 0]),
        "fp_human_flagged_as_ai": int(cm[0, 1]),
        "fn_ai_missed": int(cm[1, 0]),
        "tp_ai_detected": int(cm[1, 1]),
    }

    return results


def main():
    print("=" * 65)
    print("MILESTONE 3: SOURCE-BIAS AUDIT")
    print("=" * 65)

    all_results = []
    consistency_results = []

    for experiment, config in EXPERIMENTS.items():
        print(f"\n{'=' * 65}")
        print(experiment.upper())
        print("=" * 65)

        train = load_dataset(config["train"])

        source_label_counts = train.groupby("src")["label"].nunique()
        number_of_sources = len(source_label_counts)
        mixed_sources = int((source_label_counts > 1).sum())

        consistency_results.append({
            "experiment": experiment,
            "training_records": len(train),
            "source_categories": number_of_sources,
            "single_label_sources": number_of_sources - mixed_sources,
            "mixed_label_sources": mixed_sources,
            "single_label_source_percentage": (
                100 * (number_of_sources - mixed_sources) / number_of_sources
                if number_of_sources else float("nan")
            ),
        })

        print(f"Training records: {len(train)}")
        print(f"Training source categories: {number_of_sources}")
        print(f"Source categories with mixed labels: {mixed_sources}")

        for condition, test_path in config["tests"].items():
            test = load_dataset(test_path)

            result = evaluate_source_only(
                train, test, experiment, condition
            )
            all_results.append(result)

            print(f"\n{condition}")
            print(f"  Test records: {len(test)}")
            print(
                f"  Known-source coverage: "
                f"{result['known_source_coverage']:.2%}"
            )
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(
                f"  Accuracy on known sources: "
                f"{result['known_source_accuracy']:.4f}"
            )
            print(
                f"  Accuracy on unknown sources: "
                f"{result['unknown_source_accuracy']:.4f}"
            )
            print(
                f"  AI F1: {result['ai_f1']:.4f}"
            )
            print(
                f"  Human false-positive rate: "
                f"{result['human_false_positive_rate']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    consistency_df = pd.DataFrame(consistency_results)

    results_path = OUTPUT_DIR / "source_only_baseline_results.csv"
    consistency_path = OUTPUT_DIR / "source_label_consistency.csv"

    results_df.to_csv(results_path, index=False)
    consistency_df.to_csv(consistency_path, index=False)

    print("\n" + "=" * 65)
    print("SOURCE-BIAS AUDIT COMPLETED")
    print("=" * 65)

    print("\nSOURCE-ONLY BASELINE RESULTS")
    print(
        results_df[
            [
                "experiment",
                "condition",
                "known_source_coverage",
                "accuracy",
                "known_source_accuracy",
                "unknown_source_accuracy",
                "ai_f1",
                "human_false_positive_rate",
            ]
        ].to_string(index=False)
    )

    print("\nFILES SAVED:")
    print(results_path)
    print(consistency_path)


if __name__ == "__main__":
    main()