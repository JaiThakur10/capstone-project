
from pathlib import Path

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
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR / "outputs" / "milestone2" / "feature_ablation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_data(path):
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

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(
            f"{path.name} must contain both MAGE labels 0 and 1."
        )

    return df


def create_word_vectorizer():
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def create_char_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def create_vectorizer(name):

    if name == "Word TF-IDF":
        return create_word_vectorizer()

    if name == "Character TF-IDF":
        return create_char_vectorizer()

    if name == "Combined TF-IDF":
        return FeatureUnion([
            ("word", create_word_vectorizer()),
            ("char", create_char_vectorizer()),
        ])

    raise ValueError(f"Unknown vectorizer: {name}")


def choose_threshold(validation_labels, validation_scores):

    # MAGE label 0 = AI-generated.
    # Convert to AI-positive labels.
    y_validation = (validation_labels == 0).astype(int)

    fpr, tpr, thresholds = roc_curve(
        y_validation,
        validation_scores,
    )

    balanced_accuracy = (tpr + 1 - fpr) / 2

    best_index = int(np.argmax(balanced_accuracy))

    return float(thresholds[best_index])


def evaluate(
    experiment,
    condition,
    vectorizer_name,
    threshold,
    actual_labels,
    scores,
):
    # AI-generated is the positive class.
    actual = (actual_labels == 0).astype(int)
    predicted = (scores >= threshold).astype(int)

    cm = confusion_matrix(actual, predicted, labels=[0, 1])

    human_false_positive_rate = (
        cm[0, 1] / (cm[0, 0] + cm[0, 1])
        if (cm[0, 0] + cm[0, 1]) > 0
        else float("nan")
    )

    return {
        "experiment": experiment,
        "condition": condition,
        "vectorizer": vectorizer_name,
        "threshold": threshold,
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
        "human_false_positive_rate": human_false_positive_rate,
        "ai_roc_auc": roc_auc_score(actual, scores),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main():

    print("=" * 65)
    print("MILESTONE 5: TEXT FEATURE ABLATION")
    print("=" * 65)

    vectorizer_names = [
        "Word TF-IDF",
        "Character TF-IDF",
        "Combined TF-IDF",
    ]

    all_results = []

    for experiment, config in EXPERIMENTS.items():

        print("\n" + "=" * 65)
        print(experiment.upper())
        print("=" * 65)

        folder = BASE_DIR / "outputs" / "milestone2" / config["folder"]

        train = load_data(folder / "train.csv")
        validation = load_data(folder / "validation.csv")

        print(f"Training records: {len(train)}")
        print(f"Validation records: {len(validation)}")

        X_train_text = train["text"].tolist()
        X_validation_text = validation["text"].tolist()

        y_train = train["label"].to_numpy()
        y_validation = validation["label"].to_numpy()

        for vectorizer_name in vectorizer_names:

            print("\n" + "-" * 60)
            print(f"MODEL: {vectorizer_name}")
            print("-" * 60)

            vectorizer = create_vectorizer(vectorizer_name)

            X_train = vectorizer.fit_transform(X_train_text)
            X_validation = vectorizer.transform(
                X_validation_text
            )

            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )

            model.fit(X_train, y_train)

            # Obtain AI-generated scores using MAGE label 0.
            ai_class_index = list(model.classes_).index(0)

            validation_scores = model.predict_proba(
                X_validation
            )[:, ai_class_index]

            threshold = choose_threshold(
                y_validation,
                validation_scores,
            )

            print(f"Selected validation threshold: {threshold:.4f}")

            # Evaluate the selected threshold on each test condition.
            for condition, filename in config["tests"].items():

                test = load_data(folder / filename)

                X_test = vectorizer.transform(
                    test["text"].tolist()
                )

                test_scores = model.predict_proba(
                    X_test
                )[:, ai_class_index]

                result = evaluate(
                    experiment=experiment,
                    condition=condition,
                    vectorizer_name=vectorizer_name,
                    threshold=threshold,
                    actual_labels=test["label"].to_numpy(),
                    scores=test_scores,
                )

                all_results.append(result)

                print(f"\n{condition}")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(
                    f"  Balanced accuracy: "
                    f"{result['balanced_accuracy']:.4f}"
                )
                print(f"  AI F1: {result['ai_f1']:.4f}")
                print(
                    f"  Human FPR: "
                    f"{result['human_false_positive_rate']:.4f}"
                )
                print(f"  ROC-AUC: {result['ai_roc_auc']:.4f}")

            del vectorizer, model

    results_df = pd.DataFrame(all_results)

    output_path = OUTPUT_DIR / "feature_ablation_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("FEATURE ABLATION SUMMARY")
    print("=" * 65)

    print(
        results_df[
            [
                "experiment",
                "condition",
                "vectorizer",
                "accuracy",
                "balanced_accuracy",
                "ai_f1",
                "human_false_positive_rate",
                "ai_roc_auc",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved:")
    print(output_path)


if __name__ == "__main__":
    main()
