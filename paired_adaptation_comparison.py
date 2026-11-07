
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
    f1_score,
    roc_auc_score,
    roc_curve,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR / "outputs" / "milestone2"
    / "paired_adaptation_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]

N_BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2026

STANDARD_DOMAINS = [
    "cmv", "yelp", "xsum", "tldr", "eli5",
    "wp", "roct", "hswag", "squad", "sci_gen",
]

NINE_TRAIN_DOMAINS = [
    domain for domain in STANDARD_DOMAINS
    if domain != "eli5"
]

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

METRICS = [
    "accuracy",
    "ai_f1",
    "human_false_positive_rate",
    "ai_roc_auc",
]

COMPARISONS = [
    ("Nine-domain", "Yelp-only"),
    ("Nine-domain", "Yelp+SQuAD"),
    ("Yelp+SQuAD", "Yelp-only"),
]


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing {missing}"
        )

    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["text"].str.strip() != ""]

    return df.reset_index(drop=True)


def identify_domain(src):

    src = str(src)

    for domain in sorted(
        STANDARD_DOMAINS,
        key=len,
        reverse=True,
    ):
        if src.startswith(domain + "_"):
            return domain

    return None


def load_official_training_data(reserved_texts):

    print("\nLoading official MAGE training split...")

    dataset = load_dataset(
        "yaful/MAGE",
        split="train",
    )

    df = pd.DataFrame({
        "text": dataset["text"],
        "label": dataset["label"],
        "src": dataset["src"],
    })

    df = df.dropna(
        subset=["text", "label", "src"]
    ).copy()

    df["text"] = df["text"].astype(str)
    df["src"] = df["src"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["text"].str.strip() != ""]

    df["domain"] = df["src"].map(identify_domain)

    df = df[
        df["domain"].isin(STANDARD_DOMAINS)
    ].copy()

    before = len(df)

    df = df[
        ~df["text"].isin(reserved_texts)
    ].copy()

    removed = before - len(df)

    df = df.drop_duplicates(
        subset=["text"]
    ).copy()

    print("Eligible training records:", len(df))
    print("Reserved texts removed:", removed)
    print("Available domains:", sorted(df["domain"].unique()))

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

    for domain, class_allocations in allocations.items():

        for label, required_count in class_allocations.items():

            pool = official_df[
                (official_df["domain"] == domain)
                & (official_df["label"] == label)
            ]

            if len(pool) < required_count:
                raise ValueError(
                    f"Insufficient records for {strategy_name}: "
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

    training = pd.concat(
        samples,
        ignore_index=True,
    )

    training = training.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    if len(training) != 3600:
        raise ValueError(
            f"Expected 3600 records, got {len(training)}"
        )

    counts = training["label"].value_counts().to_dict()

    if counts != {0: 1800, 1: 1800}:
        raise ValueError(
            f"Training is not class-balanced: {counts}"
        )

    return training


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
# THRESHOLD SELECTION
# --------------------------------------------------

def select_threshold(labels, scores):

    # MAGE label 0 = AI-generated.
    actual = (labels == 0).astype(int)

    fpr, tpr, thresholds = roc_curve(
        actual,
        scores,
    )

    balanced_accuracy = (
        tpr + 1 - fpr
    ) / 2

    return float(
        thresholds[np.argmax(balanced_accuracy)]
    )


# --------------------------------------------------
# METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(labels, scores, threshold):

    # AI-generated is the positive class.
    actual = (labels == 0).astype(int)

    predicted = (
        scores >= threshold
    ).astype(int)

    human_mask = actual == 0

    return {
        "accuracy": accuracy_score(
            actual, predicted
        ),

        "ai_f1": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),

        "human_false_positive_rate": (
            (predicted[human_mask] == 1).mean()
        ),

        "ai_roc_auc": roc_auc_score(
            actual, scores
        ),
    }


# --------------------------------------------------
# PAIRED BOOTSTRAP
# --------------------------------------------------

def paired_bootstrap(
    labels,
    predictions,
    thresholds,
):

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    actual = (labels == 0).astype(int)

    ai_indices = np.where(actual == 1)[0]
    human_indices = np.where(actual == 0)[0]

    strategies = list(predictions.keys())

    # Store the paired difference for every
    # comparison and every bootstrap replicate.
    distributions = {}

    point_differences = {}

    # First calculate the observed differences.
    for first, second in COMPARISONS:

        differences = []

        for seed in SEEDS:

            first_metrics = calculate_metrics(
                labels,
                predictions[first][seed],
                thresholds[first][seed],
            )

            second_metrics = calculate_metrics(
                labels,
                predictions[second][seed],
                thresholds[second][seed],
            )

            differences.append({
                metric: (
                    first_metrics[metric]
                    - second_metrics[metric]
                )
                for metric in METRICS
            })

        point_differences[(first, second)] = {
            metric: np.mean([
                row[metric] for row in differences
            ])
            for metric in METRICS
        }

        distributions[(first, second)] = {
            metric: []
            for metric in METRICS
        }

    print("\nStarting paired bootstrap...")
    print("Bootstrap resamples:", N_BOOTSTRAPS)

    for iteration in range(N_BOOTSTRAPS):

        # Same sampled test examples for every
        # strategy and every training seed.
        sampled_ai = rng.choice(
            ai_indices,
            size=len(ai_indices),
            replace=True,
        )

        sampled_human = rng.choice(
            human_indices,
            size=len(human_indices),
            replace=True,
        )

        sampled_indices = np.concatenate([
            sampled_ai,
            sampled_human,
        ])

        sampled_labels = labels[sampled_indices]

        # Calculate metrics for every strategy/seed.
        sampled_metrics = {}

        for strategy in strategies:

            sampled_metrics[strategy] = {}

            for seed in SEEDS:

                sampled_scores = predictions[strategy][seed][
                    sampled_indices
                ]

                sampled_metrics[strategy][seed] = (
                    calculate_metrics(
                        sampled_labels,
                        sampled_scores,
                        thresholds[strategy][seed],
                    )
                )

        # Calculate paired differences.
        for first, second in COMPARISONS:

            for metric in METRICS:

                seed_differences = []

                for seed in SEEDS:

                    difference = (
                        sampled_metrics[first][seed][metric]
                        - sampled_metrics[second][seed][metric]
                    )

                    seed_differences.append(difference)

                # Average paired difference across seeds.
                average_difference = np.mean(
                    seed_differences
                )

                distributions[(first, second)][metric].append(
                    average_difference
                )

        if (iteration + 1) % 500 == 0:
            print(
                f"Completed {iteration + 1}/{N_BOOTSTRAPS}"
            )

    # Summarize paired differences.
    summary_rows = []
    raw_rows = []

    for first, second in COMPARISONS:

        for metric in METRICS:

            values = np.asarray(
                distributions[(first, second)][metric]
            )

            lower, upper = np.percentile(
                values,
                [2.5, 97.5],
            )

            estimate = point_differences[
                (first, second)
            ][metric]

            summary_rows.append({
                "comparison": f"{first} minus {second}",
                "first_strategy": first,
                "second_strategy": second,
                "metric": metric,
                "mean_paired_difference": estimate,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "bootstrap_resamples": N_BOOTSTRAPS,
            })

            for index, value in enumerate(values):

                raw_rows.append({
                    "comparison": f"{first} minus {second}",
                    "metric": metric,
                    "bootstrap_iteration": index + 1,
                    "paired_difference": value,
                })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(raw_rows),
    )


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("=" * 70)
    print("MILESTONE 9: PAIRED ADAPTATION COMPARISON")
    print("=" * 70)

    data_dir = (
        BASE_DIR / "outputs" / "milestone2"
        / "unseen_domain_eli5"
    )

    validation = load_csv(
        data_dir / "validation.csv"
    )

    test = load_csv(
        data_dir / "test_unseen_domain.csv"
    )

    validation_texts = validation["text"].tolist()
    validation_labels = validation["label"].to_numpy()

    test_texts = test["text"].tolist()
    test_labels = test["label"].to_numpy()

    reserved_texts = set(validation_texts)
    reserved_texts.update(test_texts)

    official_df = load_official_training_data(
        reserved_texts
    )

    print("\nFixed evaluation data:")
    print("Validation records:", len(validation))
    print("Unseen ELI5 test records:", len(test))

    predictions = {
        strategy: {}
        for strategy in STRATEGIES
    }

    thresholds = {
        strategy: {}
        for strategy in STRATEGIES
    }

    run_rows = []
    prediction_rows = []

    # Train all strategies with the same three seeds.
    for strategy_name, allocations in STRATEGIES.items():

        print("\n" + "=" * 65)
        print("STRATEGY:", strategy_name)
        print("=" * 65)

        for seed in SEEDS:

            print(f"\nTraining seed: {seed}")

            training = build_training_sample(
                official_df,
                strategy_name,
                allocations,
                seed,
            )

            training_texts = set(training["text"])

            if training_texts.intersection(reserved_texts):
                raise ValueError(
                    "Training overlaps with validation/test data."
                )

            print("Training records:", len(training))
            print("Class counts:", training["label"].value_counts().to_dict())

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

            ai_index = list(model.classes_).index(0)

            validation_scores = model.predict_proba(
                X_validation
            )[:, ai_index]

            threshold = select_threshold(
                validation_labels,
                validation_scores,
            )

            thresholds[strategy_name][seed] = threshold

            X_test = vectorizer.transform(test_texts)

            test_scores = model.predict_proba(
                X_test
            )[:, ai_index]

            predictions[strategy_name][seed] = test_scores

            metrics = calculate_metrics(
                test_labels,
                test_scores,
                threshold,
            )

            run_rows.append({
                "strategy": strategy_name,
                "seed": seed,
                "threshold": threshold,
                **metrics,
            })

            for row_index, score in enumerate(test_scores):

                prediction_rows.append({
                    "strategy": strategy_name,
                    "seed": seed,
                    "test_row": row_index,
                    "label": test_labels[row_index],
                    "ai_score": score,
                    "threshold": threshold,
                })

            print(f"Threshold: {threshold:.4f}")
            print(
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"AI F1: {metrics['ai_f1']:.4f} | "
                f"Human FPR: "
                f"{metrics['human_false_positive_rate']:.4f} | "
                f"AUC: {metrics['ai_roc_auc']:.4f}"
            )

            del vectorizer, model
            del X_train, X_validation, X_test
            gc.collect()

    # --------------------------------------------------
    # SAVE MODEL RUNS AND PREDICTIONS
    # --------------------------------------------------

    runs_df = pd.DataFrame(run_rows)

    runs_path = OUTPUT_DIR / "adaptation_model_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    predictions_df = pd.DataFrame(prediction_rows)

    predictions_path = (
        OUTPUT_DIR / "adaptation_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\nSaved model runs:", runs_path)
    print("Saved test predictions:", predictions_path)

    # --------------------------------------------------
    # PAIRED BOOTSTRAP
    # --------------------------------------------------

    summary_df, raw_df = paired_bootstrap(
        test_labels,
        predictions,
        thresholds,
    )

    summary_path = (
        OUTPUT_DIR / "paired_bootstrap_summary.csv"
    )

    raw_path = (
        OUTPUT_DIR / "paired_bootstrap_distributions.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PAIRED BOOTSTRAP RESULTS")
    print("=" * 70)

    print(summary_df.to_string(index=False))

    print("\nFiles saved:")
    print(runs_path)
    print(predictions_path)
    print(summary_path)
    print(raw_path)

    print("\nMILESTONE 9 COMPLETED")


if __name__ == "__main__":
    main()
