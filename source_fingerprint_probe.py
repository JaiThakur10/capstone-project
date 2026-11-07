from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "outputs" / "milestone2" / "unseen_model_OPT"

OUTPUT_DIR = (
    BASE_DIR / "outputs" / "milestone2" / "source_fingerprint"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SOURCE_FRACTION = 0.20


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = {"text", "label", "src", "domain", "model_family"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    return df


def calculate_metrics(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "macro_f1": f1_score(
            actual, predicted, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            actual, predicted, average="weighted", zero_division=0
        ),
    }


def main():

    print("=" * 65)
    print("MILESTONE 4: TEXT-ONLY SOURCE FINGERPRINT PROBE")
    print("=" * 65)

    # --------------------------------------------------
    # 1. LOAD TRAINING AND VALIDATION DATA
    # --------------------------------------------------

    print("\nLoading prepared training and validation datasets...")

    train = load_data(DATA_DIR / "train.csv")
    validation = load_data(DATA_DIR / "validation.csv")

    df = pd.concat([train, validation], ignore_index=True)

    # Keep only AI-generated records.
    # MAGE label 0 = AI-generated.
    df = df[df["label"] == 0].copy()

    df = df.dropna(
        subset=["text", "src", "domain", "model_family"]
    )

    for column in ["text", "src", "domain", "model_family"]:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df["text"] != "")
        & (df["src"] != "")
        & (df["domain"] != "")
        & (df["model_family"] != "")
    ].copy()

    print(f"AI-generated records: {len(df)}")
    print(f"Source categories: {df['src'].nunique()}")
    print(f"Domains: {df['domain'].nunique()}")

    print("\nGenerator-family distribution:")
    print(df["model_family"].value_counts().to_string())

    # --------------------------------------------------
    # 2. HOLD OUT COMPLETE SOURCE CATEGORIES
    # --------------------------------------------------

    print("\nSelecting held-out source categories...")

    rng = np.random.default_rng(SEED)

    heldout_sources = set()
    source_split_rows = []

    for family, family_df in df.groupby("model_family"):

        sources = sorted(family_df["src"].unique())

        # A family needs at least two source categories:
        # at least one for training and one for testing.
        if len(sources) < 2:
            print(
                f"Skipping {family}: only "
                f"{len(sources)} source category available."
            )

            for src in sources:
                source_split_rows.append({
                    "model_family": family,
                    "src": src,
                    "partition": "not_eligible",
                    "records": int((family_df["src"] == src).sum()),
                })

            continue

        shuffled_sources = np.array(sources, dtype=object)
        rng.shuffle(shuffled_sources)

        number_to_holdout = max(
            1,
            int(np.ceil(len(sources) * TEST_SOURCE_FRACTION))
        )

        number_to_holdout = min(
            number_to_holdout,
            len(sources) - 1
        )

        family_test_sources = set(
            shuffled_sources[:number_to_holdout]
        )

        heldout_sources.update(family_test_sources)

        for src in sources:
            source_split_rows.append({
                "model_family": family,
                "src": src,
                "partition": (
                    "test" if src in family_test_sources else "train"
                ),
                "records": int((family_df["src"] == src).sum()),
            })

        print(
            f"{family}: {len(sources)} source categories; "
            f"{len(family_test_sources)} held out."
        )

    if not heldout_sources:
        raise ValueError(
            "No source categories could be held out."
        )

    # --------------------------------------------------
    # 3. CREATE SOURCE-DISJOINT TRAIN AND TEST SETS
    # --------------------------------------------------

    test_mask = df["src"].isin(heldout_sources)

    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    # Remove any exact text overlap between train and test.
    training_texts = set(train_df["text"])

    overlap_mask = test_df["text"].isin(training_texts)
    overlap_count = int(overlap_mask.sum())

    test_df = test_df[~overlap_mask].copy()

    print("\nSplit checks:")
    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")
    print(f"Exact text overlaps removed from test: {overlap_count}")

    train_sources = set(train_df["src"])
    test_sources = set(test_df["src"])

    if train_sources.intersection(test_sources):
        raise ValueError("Source-category overlap detected!")

    print("Source-category overlap check: PASSED")

    if set(train_df["model_family"]) != set(test_df["model_family"]):
        raise ValueError(
            "Training and test do not contain the same generator "
            "families. Inspect the source split before continuing."
        )

    if not set(test_df["domain"]).issubset(set(train_df["domain"])):
        raise ValueError(
            "The test set contains a domain absent from training."
        )

    families = sorted(train_df["model_family"].unique())

    print("\nTraining family counts:")
    print(train_df["model_family"].value_counts().to_string())

    print("\nTest family counts:")
    print(test_df["model_family"].value_counts().to_string())

    print("\nTest domain distribution:")
    print(test_df["domain"].value_counts().to_string())

    # --------------------------------------------------
    # 4. TEXT-ONLY GENERATOR-FAMILY CLASSIFIER
    # --------------------------------------------------

    print("\nTraining text-only TF-IDF classifier...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    classifier.fit(X_train, train_df["model_family"])

    text_predictions = classifier.predict(X_test)

    actual = test_df["model_family"].to_numpy()

    text_metrics = calculate_metrics(
        actual, text_predictions
    )

    # --------------------------------------------------
    # 5. DOMAIN-ONLY BASELINE
    # --------------------------------------------------

    print("\nCalculating domain-only baseline...")

    # For each domain, predict its most common training family.
    domain_mapping = (
        train_df.groupby("domain")["model_family"]
        .agg(lambda values: values.mode().iloc[0])
    )

    fallback_family = train_df["model_family"].mode().iloc[0]

    domain_predictions = (
        test_df["domain"]
        .map(domain_mapping)
        .fillna(fallback_family)
        .to_numpy()
    )

    domain_metrics = calculate_metrics(
        actual, domain_predictions
    )

    # --------------------------------------------------
    # 6. RESULTS
    # --------------------------------------------------

    results = pd.DataFrame([
        {
            "model": "Text-only TF-IDF",
            **text_metrics,
        },
        {
            "model": "Domain-only baseline",
            **domain_metrics,
        },
    ])

    print("\n" + "=" * 65)
    print("COMPARISON OF MODELS")
    print("=" * 65)

    print(results.to_string(index=False))

    print("\nTEXT-ONLY CLASSIFICATION REPORT")

    report = classification_report(
        actual,
        text_predictions,
        labels=families,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    print(report_df.to_string())

    # --------------------------------------------------
    # 7. CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        actual,
        text_predictions,
        labels=families,
    )

    cm_df = pd.DataFrame(
        cm,
        index=families,
        columns=families,
    )

    # --------------------------------------------------
    # 8. SAVE RESULTS
    # --------------------------------------------------

    results_path = OUTPUT_DIR / "fingerprint_metrics.csv"
    report_path = OUTPUT_DIR / "fingerprint_per_family_report.csv"
    confusion_path = OUTPUT_DIR / "fingerprint_confusion_matrix.csv"
    split_path = OUTPUT_DIR / "fingerprint_source_split.csv"

    results.to_csv(results_path, index=False)
    report_df.to_csv(report_path)
    cm_df.to_csv(confusion_path)

    pd.DataFrame(source_split_rows).to_csv(
        split_path, index=False
    )

    print("\n" + "=" * 65)
    print("MILESTONE 4 COMPLETED")
    print("=" * 65)

    print("\nFiles saved:")
    print(results_path)
    print(report_path)
    print(confusion_path)
    print(split_path)


if __name__ == "__main__":
    main()