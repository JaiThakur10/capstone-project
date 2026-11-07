
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

BASE = Path("outputs/milestone2")
RESULTS = BASE / "results"

EXPERIMENTS = {
    "Unseen domain": {
        "model": RESULTS / "unseen_domain_eli5_tfidf_logreg.joblib",
        "threshold_file": RESULTS / "unseen_domain_eli5_threshold.json",
        "validation": BASE / "unseen_domain_eli5/validation.csv",
        "tests": {
            "In-domain reference": BASE / "unseen_domain_eli5/test_in_domain.csv",
            "Unseen ELI5": BASE / "unseen_domain_eli5/test_unseen_domain.csv",
        },
    },
    "Unseen model family": {
        "model": RESULTS / "unseen_opt_family_tfidf_logreg.joblib",
        "threshold_file": RESULTS / "unseen_opt_family_threshold.json",
        "validation": BASE / "unseen_model_OPT/validation.csv",
        "tests": {
            "Seen model families": BASE / "unseen_model_OPT/test_seen_family.csv",
            "Unseen OPT family": BASE / "unseen_model_OPT/test_unseen_OPT_family.csv",
        },
    },
}


def load_data(path):
    frame = pd.read_csv(path)
    frame = frame.dropna(
        subset=["text", "label"]
    ).copy()

    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)

    return frame


def ai_scores(model, frame):
    probabilities = model.predict_proba(frame["text"])
    ai_index = list(model.classes_).index(0)
    return probabilities[:, ai_index]


def choose_thresholds(model, validation):

    y_true_ai = (
        validation["label"].to_numpy() == 0
    ).astype(int)

    scores = ai_scores(model, validation)

    # Strategy 1: maximize AI-class F1.
    precision, recall, thresholds = precision_recall_curve(
        y_true_ai, scores
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    f1_index = int(np.nanargmax(f1_values))
    f1_threshold = float(thresholds[f1_index])

    # Strategy 2: maximize balanced accuracy.
    fpr, tpr, roc_thresholds = roc_curve(
        y_true_ai, scores
    )

    balanced_values = (tpr + 1 - fpr) / 2

    finite = np.isfinite(roc_thresholds)

    valid_indices = np.where(finite)[0]

    balanced_index = valid_indices[
        np.argmax(balanced_values[finite])
    ]

    balanced_threshold = float(
        roc_thresholds[balanced_index]
    )

    return {
        "Max AI F1": f1_threshold,
        "Max balanced accuracy": balanced_threshold,
    }


def calculate_metrics(model, frame, threshold):

    y_true = frame["label"].to_numpy()
    scores = ai_scores(model, frame)

    predictions = np.where(
        scores >= threshold, 0, 1
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    human_total = matrix[1].sum()

    human_fpr = (
        matrix[1, 0] / human_total
        if human_total else np.nan
    )

    return {
        "threshold": threshold,
        "records": len(frame),
        "predicted_AI": int((predictions == 0).sum()),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "ai_precision": precision_score(
            y_true, predictions,
            pos_label=0, zero_division=0
        ),
        "ai_recall": recall_score(
            y_true, predictions,
            pos_label=0, zero_division=0
        ),
        "ai_f1": f1_score(
            y_true, predictions,
            pos_label=0, zero_division=0
        ),
        "human_false_positive_rate": human_fpr,
        "ai_roc_auc": roc_auc_score(
            (y_true == 0).astype(int), scores
        ),
        "true_AI_pred_AI": int(matrix[0, 0]),
        "true_AI_pred_Human": int(matrix[0, 1]),
        "true_Human_pred_AI": int(matrix[1, 0]),
        "true_Human_pred_Human": int(matrix[1, 1]),
    }


all_results = []

for experiment_name, config in EXPERIMENTS.items():

    print("\n" + "=" * 65)
    print(experiment_name.upper())
    print("=" * 65)

    model = joblib.load(config["model"])

    validation = load_data(config["validation"])

    thresholds = choose_thresholds(
        model, validation
    )

    with open(
        config["threshold_file"],
        encoding="utf-8"
    ) as file:
        previous = json.load(file)

    print(
        "\nPreviously saved threshold:",
        previous["validation_selected_threshold"]
    )

    print("\nValidation-selected thresholds:")
    for name, threshold in thresholds.items():
        print(name, ":", round(threshold, 4))

    for condition, path in config["tests"].items():

        frame = load_data(path)

        print("\nTest condition:", condition)

        for strategy, threshold in thresholds.items():

            metrics = calculate_metrics(
                model,
                frame,
                threshold
            )

            metrics["experiment"] = experiment_name
            metrics["condition"] = condition
            metrics["threshold_strategy"] = strategy

            all_results.append(metrics)

            print("\n", strategy)
            print(
                pd.Series(metrics).to_string()
            )


results = pd.DataFrame(all_results)

results.to_csv(
    RESULTS / "threshold_sensitivity_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("THRESHOLD SENSITIVITY ANALYSIS COMPLETED")
print("=" * 65)

print(
    results[
        [
            "experiment",
            "condition",
            "threshold_strategy",
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "ai_f1",
            "human_false_positive_rate",
            "ai_roc_auc",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved:")
print(RESULTS / "threshold_sensitivity_results.csv")
