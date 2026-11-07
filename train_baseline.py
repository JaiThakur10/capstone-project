
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

OUTPUT_DIR = Path("outputs")
SEED = 42

TRAIN_FILE = OUTPUT_DIR / "yelp_domain_training_sample.csv"
CROSS_DOMAIN_FILE = OUTPUT_DIR / "eli5_cross_domain_test_sample.csv"

# 1. Load the samples prepared earlier
yelp = pd.read_csv(TRAIN_FILE)
eli5 = pd.read_csv(CROSS_DOMAIN_FILE)

required = {"text", "label", "src", "source_type"}

for name, frame in [("Yelp", yelp), ("ELI5", eli5)]:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    frame = frame.dropna(
    subset=["text", "label", "src"]
).copy()
    frame["text"] = frame["text"].astype(str)

    frame["label"] = pd.to_numeric(
        frame["label"], errors="raise"
    ).astype(int)

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {name}")

    # Verify the mapping used in this project:
    # machine = 0, human = 1
    expected = frame["source_type"].map({
        "machine": 0,
        "human": 1
    })

    known = expected.notna()
    if not (frame.loc[known, "label"] ==
            expected.loc[known]).all():
        raise ValueError(
            f"Source type and label mapping disagree in {name}"
        )

# 2. Remove ambiguous labels and duplicate text
def clean_texts(frame, name):
    conflicting = (
        frame.groupby("text")["label"].nunique()
    )
    conflicting_texts = set(
        conflicting[conflicting > 1].index
    )

    if conflicting_texts:
        print(
            f"{name}: removing {len(conflicting_texts)} "
            "texts with conflicting labels"
        )
        frame = frame[
            ~frame["text"].isin(conflicting_texts)
        ].copy()

    before = len(frame)
    frame = frame.drop_duplicates(
        subset=["text"], keep="first"
    ).copy()

    print(
        f"{name}: removed {before - len(frame)} "
        "duplicate rows"
    )
    return frame

yelp = clean_texts(yelp, "Yelp")
eli5 = clean_texts(eli5, "ELI5")

overlap = set(yelp["text"]) & set(eli5["text"])
print("Exact text overlap between domains:", len(overlap))

if overlap:
    raise ValueError(
        "Text overlap remains between domains. "
        "Resolve it before training."
    )

# 3. Divide Yelp into model-training and in-domain evaluation sets
X_train, X_yelp_test, y_train, y_yelp_test = train_test_split(
    yelp["text"],
    yelp["label"],
    test_size=0.20,
    random_state=SEED,
    stratify=yelp["label"]
)

print("\nTraining records:", len(X_train))
print("Yelp in-domain test records:", len(X_yelp_test))
print("ELI5 cross-domain test records:", len(eli5))

# 4. Build a text-classification pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=SEED
        )
    )
])

# 5. Train only on Yelp training records
print("\nTraining TF-IDF + Logistic Regression...")
model.fit(X_train, y_train)

# 6. Evaluate using the same model and default threshold
def evaluate(name, texts, labels):
    predictions = model.predict(texts)

    # Probability that a record is AI-generated (label 0)
    probabilities = model.predict_proba(texts)
    ai_column = list(model.classes_).index(0)
    ai_probabilities = probabilities[:, ai_column]

    cm = confusion_matrix(
        labels, predictions, labels=[0, 1]
    )

    # Matrix order: rows=true labels, columns=predicted labels.
    # Human texts wrongly predicted as AI are cm[1, 0].
    human_total = cm[1, :].sum()
    human_false_positive_rate = (
        cm[1, 0] / human_total
        if human_total else float("nan")
    )

    metrics = {
        "evaluation": name,
        "records": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "ai_precision": precision_score(
            labels, predictions, pos_label=0,
            zero_division=0
        ),
        "ai_recall": recall_score(
            labels, predictions, pos_label=0,
            zero_division=0
        ),
        "ai_f1": f1_score(
            labels, predictions, pos_label=0,
            zero_division=0
        ),
        "human_false_positive_rate": human_false_positive_rate,
        "ai_roc_auc": roc_auc_score(
            (labels == 0).astype(int), ai_probabilities
        ),
    }

    print(f"\n{'=' * 18} {name} {'=' * 18}")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["AI-generated (0)", "Human-written (1)"],
        digits=4,
        zero_division=0
    ))

    print("Confusion matrix (rows=true, columns=predicted)")
    print("Labels ordered as [AI=0, Human=1]")
    print(cm)
    print(
        "Human false-positive rate:",
        round(human_false_positive_rate, 4)
    )

    return metrics, predictions

yelp_metrics, yelp_predictions = evaluate(
    "Yelp in-domain holdout",
    X_yelp_test,
    y_yelp_test
)

eli5_metrics, eli5_predictions = evaluate(
    "ELI5 cross-domain pilot",
    eli5["text"],
    eli5["label"]
)

# 7. Save metrics, predictions, and the trained pipeline
metrics_df = pd.DataFrame([
    yelp_metrics,
    eli5_metrics
])
metrics_df.to_csv(
    OUTPUT_DIR / "baseline_metrics.csv",
    index=False
)

predictions_df = pd.concat([
    pd.DataFrame({
        "evaluation": "yelp_in_domain",
        "src": yelp.loc[
            X_yelp_test.index, "src"
        ].values,
        "true_label": y_yelp_test.values,
        "predicted_label": yelp_predictions,
    }),
    pd.DataFrame({
        "evaluation": "eli5_cross_domain",
        "src": eli5["src"].values,
        "true_label": eli5["label"].values,
        "predicted_label": eli5_predictions,
    })
], ignore_index=True)

predictions_df.to_csv(
    OUTPUT_DIR / "baseline_predictions.csv",
    index=False
)

joblib.dump(
    model,
    OUTPUT_DIR / "tfidf_logreg_baseline.joblib"
)

print("\nMETRICS SUMMARY")
print(metrics_df.to_string(index=False))
print("\nSaved baseline metrics, predictions, and model.")
