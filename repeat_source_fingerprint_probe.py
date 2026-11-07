
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "milestone2" / "unseen_model_OPT"
OUTPUT_DIR = BASE_DIR / "outputs" / "milestone2" / "source_fingerprint"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
TEST_SOURCE_FRACTION = 0.20


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = {"text", "label", "src", "domain", "model_family"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return df


def metrics(actual, predicted):
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


def build_split(df, seed):
    rng = np.random.default_rng(seed)
    heldout_sources = set()
    split_records = []

    for family, family_df in df.groupby("model_family"):
        sources = sorted(family_df["src"].unique())

        if len(sources) < 2:
            raise ValueError(
                f"{family} has fewer than two source categories."
            )

        shuffled = np.array(sources, dtype=object)
        rng.shuffle(shuffled)

        n_test = max(
            1,
            int(np.ceil(len(sources) * TEST_SOURCE_FRACTION))
        )
        n_test = min(n_test, len(sources) - 1)

        family_test_sources = set(shuffled[:n_test])
        heldout_sources.update(family_test_sources)

        for src in sources:
            split_records.append({
                "seed": seed,
                "model_family": family,
                "src": src,
                "partition": (
                    "test" if src in family_test_sources else "train"
                ),
                "records": int((family_df["src"] == src).sum()),
            })

    test_mask = df["src"].isin(heldout_sources)

    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    # Remove exact text overlaps.
    training_texts = set(train_df["text"])
    overlap = test_df["text"].isin(training_texts)
    removed = int(overlap.sum())
    test_df = test_df[~overlap].copy()

    if set(train_df["src"]) & set(test_df["src"]):
        raise ValueError("Source overlap detected.")

    if set(train_df["model_family"]) != set(test_df["model_family"]):
        raise ValueError("A generator family is missing from a split.")

    if not set(test_df["domain"]).issubset(set(train_df["domain"])):
        raise ValueError("Test contains a domain absent from training.")

    return train_df, test_df, split_records, removed


def main():
    print("=" * 65)
    print("REPEATED SOURCE FINGERPRINT EXPERIMENT")
    print("=" * 65)

    train = load_data(DATA_DIR / "train.csv")
    validation = load_data(DATA_DIR / "validation.csv")

    df = pd.concat([train, validation], ignore_index=True)

    # MAGE label 0 = AI-generated.
    df = df[df["label"] == 0].copy()

    for col in ["text", "src", "domain", "model_family"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(
        subset=["text", "src", "domain", "model_family"]
    )

    df = df[
        (df["text"] != "")
        & (df["src"] != "")
        & (df["domain"] != "")
        & (df["model_family"] != "")
    ].copy()

    print(f"AI-generated records: {len(df)}")
    print(f"Source categories: {df['src'].nunique()}")
    print(f"Families: {df['model_family'].nunique()}")

    all_metrics = []
    all_reports = []
    all_confusions = []
    all_splits = []

    for seed in SEEDS:
        print("\n" + "=" * 65)
        print(f"SEED: {seed}")
        print("=" * 65)

        train_df, test_df, split_records, removed = build_split(
            df, seed
        )

        all_splits.extend(split_records)

        print(f"Training records: {len(train_df)}")
        print(f"Test records: {len(test_df)}")
        print(f"Exact overlaps removed: {removed}")
        print("Source-disjoint check: PASSED")

        families = sorted(train_df["model_family"].unique())

        # Text-only classifier. src and domain are NOT features.
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
            random_state=seed,
        )

        classifier.fit(X_train, train_df["model_family"])
        text_pred = classifier.predict(X_test)

        actual = test_df["model_family"].to_numpy()

        # Domain-only baseline.
        domain_mapping = (
            train_df.groupby("domain")["model_family"]
            .agg(lambda x: x.mode().iloc[0])
        )

        fallback = train_df["model_family"].mode().iloc[0]

        domain_pred = (
            test_df["domain"]
            .map(domain_mapping)
            .fillna(fallback)
            .to_numpy()
        )

        # Majority-family baseline.
        majority = train_df["model_family"].mode().iloc[0]
        majority_pred = np.full(len(test_df), majority)

        predictions = {
            "Text-only TF-IDF": text_pred,
            "Domain-only baseline": domain_pred,
            "Majority-family baseline": majority_pred,
        }

        for model_name, predicted in predictions.items():
            row = {
                "seed": seed,
                "model": model_name,
                "train_records": len(train_df),
                "test_records": len(test_df),
                **metrics(actual, predicted),
            }
            all_metrics.append(row)

        # Per-family report for text-only model.
        report = classification_report(
            actual,
            text_pred,
            labels=families,
            output_dict=True,
            zero_division=0,
        )

        for family in families:
            all_reports.append({
                "seed": seed,
                "model_family": family,
                **report[family],
            })

        cm = confusion_matrix(
            actual, text_pred, labels=families
        )

        for i, actual_family in enumerate(families):
            for j, predicted_family in enumerate(families):
                all_confusions.append({
                    "seed": seed,
                    "actual_family": actual_family,
                    "predicted_family": predicted_family,
                    "count": int(cm[i, j]),
                })

        print("\nModel comparison:")
        print(
            pd.DataFrame([
                {
                    "model": name,
                    **metrics(actual, pred),
                }
                for name, pred in predictions.items()
            ]).to_string(index=False)
        )

        print("\nText-only family report:")
        print(
            pd.DataFrame(report).transpose().to_string()
        )

    # Save detailed results.
    metrics_df = pd.DataFrame(all_metrics)
    reports_df = pd.DataFrame(all_reports)
    confusions_df = pd.DataFrame(all_confusions)
    splits_df = pd.DataFrame(all_splits)

    metrics_df.to_csv(
        OUTPUT_DIR / "fingerprint_repeat_runs.csv", index=False
    )
    reports_df.to_csv(
        OUTPUT_DIR / "fingerprint_repeat_family_reports.csv",
        index=False,
    )
    confusions_df.to_csv(
        OUTPUT_DIR / "fingerprint_repeat_confusions.csv",
        index=False,
    )
    splits_df.to_csv(
        OUTPUT_DIR / "fingerprint_repeat_source_splits.csv",
        index=False,
    )

    # Mean and sample standard deviation across seeds.
    summary = (
        metrics_df
        .groupby("model")[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR / "fingerprint_repeat_summary.csv"
    )

    print("\n" + "=" * 65)
    print("REPEATABILITY SUMMARY")
    print("=" * 65)
    print(summary.to_string())

    print("\nFiles saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
