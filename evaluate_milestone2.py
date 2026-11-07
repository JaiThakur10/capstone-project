
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
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
    precision_recall_curve,
    balanced_accuracy_score,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path("outputs/milestone2")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------
# Experiment A: Unseen domain
# --------------------------------------------------

DOMAIN_DIR = BASE_DIR / "unseen_domain_eli5"

# --------------------------------------------------
# Experiment B: Unseen model family
# --------------------------------------------------

MODEL_DIR = BASE_DIR / "unseen_model_OPT"


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

def load_frame(path, name):

    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found: {path}"
        )

    frame = pd.read_csv(path)

    required = {"text", "label", "src", "domain"}

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )

    frame = frame.dropna(
        subset=["text", "label", "src", "domain"]
    ).copy()

    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {name}")

    if frame["text"].duplicated().any():
        raise ValueError(
            f"Duplicate text found in {name}"
        )

    return frame.reset_index(drop=True)


# ==================================================
# 3. CHECK TEXT OVERLAP
# ==================================================

def check_overlap(train, validation, tests):

    train_text = set(train["text"])
    validation_text = set(validation["text"])

    if train_text & validation_text:
        raise ValueError(
            "Exact overlap between training and validation."
        )

    for name, test in tests.items():

        test_text = set(test["text"])

        if train_text & test_text:
            raise ValueError(
                f"Exact training/test overlap in {name}."
            )

        if validation_text & test_text:
            raise ValueError(
                f"Exact validation/test overlap in {name}."
            )

    print("\nExact text-overlap checks passed.")


# ==================================================
# 4. DEFINE THE CLASSIFIER
# ==================================================

def make_model():

    return Pipeline([
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


# ==================================================
# 5. GET AI-GENERATED TEXT SCORES
# ==================================================

def get_ai_scores(model, texts):

    probabilities = model.predict_proba(texts)

    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError(
            "The trained model does not contain both labels."
        )

    ai_index = classes.index(0)

    return probabilities[:, ai_index]


# ==================================================
# 6. SELECT THRESHOLD USING VALIDATION ONLY
# ==================================================

def select_threshold(model, validation):

    y_true = validation["label"].to_numpy()

    ai_scores = get_ai_scores(
        model,
        validation["text"]
    )

    # Treat AI-generated text (label 0) as the positive class.
    ai_true = (y_true == 0).astype(int)

    precision, recall, thresholds = precision_recall_curve(
        ai_true,
        ai_scores
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to select a decision threshold."
        )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    best_index = int(np.nanargmax(f1_values))

    threshold = float(thresholds[best_index])

    print("\nValidation threshold selected:", round(threshold, 4))
    print(
        "Validation AI F1:",
        round(float(f1_values[best_index]), 4)
    )

    return threshold


# ==================================================
# 7. EVALUATE A TEST CONDITION
# ==================================================

def evaluate_condition(
    model,
    frame,
    threshold,
    experiment,
    condition
):

    y_true = frame["label"].to_numpy()

    ai_scores = get_ai_scores(
        model,
        frame["text"]
    )

    # Score above threshold means AI-generated (label 0).
    predictions = np.where(
        ai_scores >= threshold,
        0,
        1
    )

    # Confusion matrix ordering: AI=0, Human=1.
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    true_ai, false_negative, false_positive, true_human = (
        int(matrix[0, 0]),
        int(matrix[0, 1]),
        int(matrix[1, 0]),
        int(matrix[1, 1]),
    )

    human_total = int(matrix[1].sum())

    human_fpr = (
        false_positive / human_total
        if human_total else float("nan")
    )

    ai_auc = roc_auc_score(
        (y_true == 0).astype(int),
        ai_scores
    )

    metrics = {
        "experiment": experiment,
        "condition": condition,
        "records": len(frame),
        "AI_records": int((y_true == 0).sum()),
        "human_records": int((y_true == 1).sum()),
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
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
        "ai_roc_auc": ai_auc,
    }

    confusion = {
        "experiment": experiment,
        "condition": condition,
        "true_AI_pred_AI": true_ai,
        "true_AI_pred_Human": false_negative,
        "true_Human_pred_AI": false_positive,
        "true_Human_pred_Human": true_human,
    }

    predictions_frame = frame[
        ["text", "label", "src", "domain"]
    ].copy()

    predictions_frame["experiment"] = experiment
    predictions_frame["condition"] = condition
    predictions_frame["predicted_label"] = predictions
    predictions_frame["ai_score"] = ai_scores
    predictions_frame["decision_threshold"] = threshold

    return metrics, confusion, predictions_frame


# ==================================================
# 8. RUN ONE EXPERIMENT
# ==================================================

def run_experiment(
    experiment_name,
    train_file,
    validation_file,
    test_files,
):

    print("\n" + "=" * 65)
    print(experiment_name.upper())
    print("=" * 65)

    train = load_frame(
        train_file,
        "Training"
    )

    validation = load_frame(
        validation_file,
        "Validation"
    )

    tests = {
        name: load_frame(path, name)
        for name, path in test_files.items()
    }

    check_overlap(
        train,
        validation,
        tests
    )

    print("\nTRAINING LABEL COUNTS")
    print(train["label"].value_counts().sort_index())

    print("\nVALIDATION LABEL COUNTS")
    print(validation["label"].value_counts().sort_index())

    print("\nTRAINING TF-IDF + LOGISTIC REGRESSION")

    model = make_model()

    model.fit(
        train["text"],
        train["label"]
    )

    # Threshold is selected from validation, not test.
    threshold = select_threshold(
        model,
        validation
    )

    all_metrics = []
    all_confusions = []
    all_predictions = []

    for condition, frame in tests.items():

        print("\nEvaluating:", condition)

        metrics, confusion, predictions = evaluate_condition(
            model,
            frame,
            threshold,
            experiment_name,
            condition
        )

        all_metrics.append(metrics)
        all_confusions.append(confusion)
        all_predictions.append(predictions)

        print("\nMETRICS")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

            elif key in {"experiment", "condition", "records",
                         "AI_records", "human_records"}:
                print(f"{key}: {value}")

        print("\nCONFUSION MATRIX")
        print(
            "[[True AI predicted AI, True AI predicted Human],"
        )
        print(
            " [True Human predicted AI, True Human predicted Human]]"
        )
        print(
            [
                [confusion["true_AI_pred_AI"],
                 confusion["true_AI_pred_Human"]],
                [confusion["true_Human_pred_AI"],
                 confusion["true_Human_pred_Human"]],
            ]
        )

    # Save one model per experiment.
    model_path = RESULTS_DIR / (
        experiment_name.replace(" ", "_").lower()
        + "_tfidf_logreg.joblib"
    )

    joblib.dump(model, model_path)

    threshold_path = RESULTS_DIR / (
        experiment_name.replace(" ", "_").lower()
        + "_threshold.json"
    )

    with open(threshold_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": experiment_name,
                "validation_selected_threshold": threshold,
                "selection_metric": "AI-class F1-score",
                "random_seed": SEED,
            },
            file,
            indent=4
        )

    return (
        all_metrics,
        all_confusions,
        all_predictions
    )


# ==================================================
# 9. EXPERIMENT A: UNSEEN DOMAIN
# ==================================================

domain_results = run_experiment(
    experiment_name="Unseen Domain ELI5",

    train_file=DOMAIN_DIR / "train.csv",

    validation_file=DOMAIN_DIR / "validation.csv",

    test_files={
        "In-domain reference": (
            DOMAIN_DIR / "test_in_domain.csv"
        ),
        "Unseen ELI5 domain": (
            DOMAIN_DIR / "test_unseen_domain.csv"
        ),
    }
)


# ==================================================
# 10. EXPERIMENT B: UNSEEN MODEL FAMILY
# ==================================================

model_results = run_experiment(
    experiment_name="Unseen OPT Family",

    train_file=MODEL_DIR / "train.csv",

    validation_file=MODEL_DIR / "validation.csv",

    test_files={
        "Seen model families": (
            MODEL_DIR / "test_seen_family.csv"
        ),
        "Unseen OPT family": (
            MODEL_DIR / "test_unseen_OPT_family.csv"
        ),
    }
)


# ==================================================
# 11. SAVE ALL RESULTS
# ==================================================

all_metrics = (
    domain_results[0]
    + model_results[0]
)

all_confusions = (
    domain_results[1]
    + model_results[1]
)

all_predictions = (
    domain_results[2]
    + model_results[2]
)

metrics_df = pd.DataFrame(all_metrics)

confusion_df = pd.DataFrame(all_confusions)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

metrics_df.to_csv(
    RESULTS_DIR / "milestone2_metrics.csv",
    index=False
)

confusion_df.to_csv(
    RESULTS_DIR / "milestone2_confusion_matrices.csv",
    index=False
)

predictions_df.to_csv(
    RESULTS_DIR / "milestone2_predictions.csv",
    index=False
)

print("\n" + "=" * 65)
print("MILESTONE 2 EVALUATION COMPLETED")
print("=" * 65)

print("\nFINAL METRICS")
print(metrics_df.round(4).to_string(index=False))

print("\nCONFUSION MATRICES")
print(confusion_df.to_string(index=False))

print("\nResults saved in:")
print(RESULTS_DIR.resolve())
