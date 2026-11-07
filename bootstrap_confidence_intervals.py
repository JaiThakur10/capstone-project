
import os
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
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = "outputs/milestone2"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "bootstrap_confidence_intervals"
)

N_BOOTSTRAPS = 2000
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_csv(path):
    df = pd.read_csv(path)

    required = {"text", "label"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns: {required}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path}")

    return df.reset_index(drop=True)


# --------------------------------------------------
# COMBINED TF-IDF MODEL
# --------------------------------------------------

def create_vectorizer():

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

    return FeatureUnion([
        ("word", word),
        ("char", char),
    ])


# --------------------------------------------------
# VALIDATION THRESHOLD
# --------------------------------------------------

def select_threshold(labels, scores):

    # MAGE label 0 = AI-generated.
    actual = (labels == 0).astype(int)

    fpr, tpr, thresholds = roc_curve(
        actual,
        scores,
    )

    balanced_accuracy = (tpr + 1 - fpr) / 2

    return float(thresholds[np.argmax(balanced_accuracy)])


# --------------------------------------------------
# METRICS
# --------------------------------------------------

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "ai_precision",
    "ai_recall",
    "ai_f1",
    "human_false_positive_rate",
    "ai_roc_auc",
]


def calculate_metrics(labels, scores, threshold):

    # AI-generated = positive class.
    actual = (labels == 0).astype(int)
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
        ),

        "ai_roc_auc": roc_auc_score(
            actual, scores
        ),
    }


# --------------------------------------------------
# STRATIFIED BOOTSTRAP
# --------------------------------------------------

def bootstrap_metrics(labels, scores, threshold):

    rng = np.random.default_rng(RANDOM_SEED)

    actual = (labels == 0).astype(int)

    ai_indices = np.where(actual == 1)[0]
    human_indices = np.where(actual == 0)[0]

    point_estimates = calculate_metrics(
        labels, scores, threshold
    )

    bootstrap_results = {
        metric: [] for metric in METRIC_NAMES
    }

    for iteration in range(N_BOOTSTRAPS):

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
        sampled_scores = scores[sampled_indices]

        results = calculate_metrics(
            sampled_labels,
            sampled_scores,
            threshold,
        )

        for metric in METRIC_NAMES:
            bootstrap_results[metric].append(
                results[metric]
            )

    summary = []

    for metric in METRIC_NAMES:

        values = np.asarray(
            bootstrap_results[metric]
        )

        lower, upper = np.percentile(
            values,
            [2.5, 97.5],
        )

        summary.append({
            "metric": metric,
            "estimate": point_estimates[metric],
            "ci_lower_95": lower,
            "ci_upper_95": upper,
        })

    return pd.DataFrame(summary)


# --------------------------------------------------
# EXPERIMENT DEFINITIONS
# --------------------------------------------------

EXPERIMENTS = [
    {
        "name": "unseen_domain",
        "folder": "unseen_domain_eli5",
        "tests": {
            "in_domain_reference": "test_in_domain.csv",
            "unseen_ELI5": "test_unseen_domain.csv",
        },
    },
    {
        "name": "unseen_model_family",
        "folder": "unseen_model_OPT",
        "tests": {
            "seen_family_reference": "test_seen_family.csv",
            "unseen_OPT": "test_unseen_OPT_family.csv",
        },
    },
]


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    all_results = []

    for experiment in EXPERIMENTS:

        print("\n" + "=" * 65)
        print("EXPERIMENT:", experiment["name"])
        print("=" * 65)

        folder = os.path.join(
            BASE_DIR,
            experiment["folder"],
        )

        train = load_csv(
            os.path.join(folder, "train.csv")
        )

        validation = load_csv(
            os.path.join(folder, "validation.csv")
        )

        print("Training records:", len(train))
        print("Validation records:", len(validation))

        # Fit vectorizer only on training text.
        vectorizer = create_vectorizer()

        X_train = vectorizer.fit_transform(
            train["text"].tolist()
        )

        X_validation = vectorizer.transform(
            validation["text"].tolist()
        )

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )

        model.fit(
            X_train,
            train["label"],
        )

        # MAGE label 0 = AI-generated.
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

        for condition, filename in experiment["tests"].items():

            print("\nTest condition:", condition)

            test = load_csv(
                os.path.join(folder, filename)
            )

            X_test = vectorizer.transform(
                test["text"].tolist()
            )

            test_scores = model.predict_proba(
                X_test
            )[:, ai_index]

            print("Test records:", len(test))
            print("Running bootstrap...")

            result = bootstrap_metrics(
                test["label"].to_numpy(),
                test_scores,
                threshold,
            )

            result.insert(0, "condition", condition)
            result.insert(0, "experiment", experiment["name"])
            result["threshold"] = threshold
            result["test_size"] = len(test)

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{experiment['name']}_{condition}_bootstrap.csv",
            )

            result.to_csv(
                output_path,
                index=False,
            )

            all_results.append(result)

            print(result.to_string(index=False))
            print("Saved:", output_path)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_path = os.path.join(
        OUTPUT_DIR,
        "all_bootstrap_results.csv",
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("BOOTSTRAP EXPERIMENT COMPLETED")
    print("=" * 65)

    print("Results folder:", OUTPUT_DIR)
    print("Combined results:", combined_path)


if __name__ == "__main__":
    main()
