from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR / "outputs" / "milestone2" / "source_bias"
)
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


def load_data(path):
    df = pd.read_csv(path)

    required = {"src", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    if df[["src", "label"]].isna().any().any():
        raise ValueError(f"Missing source or label in {path}")

    df["src"] = df["src"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def main():
    all_details = []
    all_summaries = []

    for experiment, config in EXPERIMENTS.items():

        print("\n" + "=" * 65)
        print(experiment.upper())
        print("=" * 65)

        train = load_data(config["train"])

        # Learn source-to-label associations from training.
        source_mapping = (
            train.groupby("src")["label"]
            .agg(lambda x: x.mode().iloc[0])
        )

        fallback_label = int(train["label"].mode().iloc[0])

        for condition, test_path in config["tests"].items():

            test = load_data(test_path)

            test["known_in_training"] = (
                test["src"].isin(source_mapping.index)
            )

            test["training_label"] = (
                test["src"].map(source_mapping)
            )

            test["predicted_label"] = (
                test["training_label"]
                .fillna(fallback_label)
                .astype(int)
            )

            test["correct"] = (
                test["label"] == test["predicted_label"]
            )

            details = (
                test.groupby("src")
                .agg(
                    test_records=("label", "size"),
                    ai_records=("label", lambda x: (x == 0).sum()),
                    human_records=("label", lambda x: (x == 1).sum()),
                    known_in_training=("known_in_training", "first"),
                    training_label=("training_label", "first"),
                    predicted_label=("predicted_label", "first"),
                    correct_predictions=("correct", "sum"),
                )
                .reset_index()
            )

            details.insert(0, "condition", condition)
            details.insert(0, "experiment", experiment)

            details["ai_percentage"] = (
                details["ai_records"] / details["test_records"] * 100
            )

            details["source_accuracy"] = (
                details["correct_predictions"] /
                details["test_records"]
            )

            all_details.append(details)

            # Summarize records and source categories by source coverage.
            summary = (
                test.groupby("known_in_training")
                .agg(
                    records=("label", "size"),
                    source_categories=("src", "nunique"),
                    ai_records=("label", lambda x: (x == 0).sum()),
                    human_records=("label", lambda x: (x == 1).sum()),
                    correct_predictions=("correct", "sum"),
                )
                .reset_index()
            )

            summary.insert(0, "condition", condition)
            summary.insert(0, "experiment", experiment)

            summary["accuracy"] = (
                summary["correct_predictions"] / summary["records"]
            )

            all_summaries.append(summary)

            print(f"\n{condition}")
            print(
                details[
                    [
                        "src",
                        "test_records",
                        "ai_records",
                        "human_records",
                        "known_in_training",
                        "ai_percentage",
                    ]
                ].to_string(index=False)
            )

    details_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)

    details_path = OUTPUT_DIR / "source_category_details.csv"
    summary_path = OUTPUT_DIR / "source_coverage_summary.csv"

    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("SOURCE DETAIL INSPECTION COMPLETED")
    print("=" * 65)

    print("\nSOURCE COVERAGE SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nFiles saved:")
    print(details_path)
    print(summary_path)


if __name__ == "__main__":
    main()