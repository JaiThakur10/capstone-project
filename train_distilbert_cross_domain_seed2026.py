"""Fine-tune DistilBERT on the prepared MAGE cross-domain split.

MAGE labels: 0 = AI-generated, 1 = human-written.
This is a first, single-seed transformer experiment intended for comparison
with the existing Combined TF-IDF cross-domain baseline.
"""

from pathlib import Path
import json
import random
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
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
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "milestone2" / "unseen_domain_eli5"
OUTPUT_DIR = BASE_DIR / "outputs" / "milestone2" / "distilbert_cross_domain"
MODEL_DIR = OUTPUT_DIR / "best_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 2026
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 1  # Start conservatively on the 6 GB GTX 1660 Ti.
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
NUM_WORKERS = 0  # Safer default for Windows.

ID2LABEL = {0: "AI-generated", 1: "Human-written"}
LABEL2ID = {"AI-generated": 0, "Human-written": 1}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared split: {path}")
    frame = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["text"].str.strip().ne("")].reset_index(drop=True)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {filename}")
    return frame


def tokenize_frame(tokenizer, frame: pd.DataFrame) -> TensorDataset:
    encoded = tokenizer(
        frame["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.long)
    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        labels,
    )


def make_loader(dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


@torch.inference_mode()
def predict_ai_scores(model, loader, device) -> np.ndarray:
    model.eval()
    score_parts = []
    for input_ids, attention_mask, _labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        probabilities = torch.softmax(logits, dim=1)
        # Class index 0 is AI-generated in this project.
        score_parts.append(probabilities[:, 0].detach().cpu().numpy())
    return np.concatenate(score_parts)


def select_threshold(labels: np.ndarray, ai_scores: np.ndarray) -> float:
    actual_ai = (labels == 0).astype(int)
    fpr, tpr, thresholds = roc_curve(actual_ai, ai_scores)
    balanced_accuracy = (tpr + 1.0 - fpr) / 2.0
    return float(thresholds[int(np.argmax(balanced_accuracy))])


def calculate_metrics(labels: np.ndarray, ai_scores: np.ndarray, threshold: float) -> dict:
    actual_ai = (labels == 0).astype(int)
    predicted_ai = (ai_scores >= threshold).astype(int)
    human_mask = actual_ai == 0
    return {
        "accuracy": float(accuracy_score(actual_ai, predicted_ai)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_ai, predicted_ai)),
        "ai_precision": float(precision_score(actual_ai, predicted_ai, zero_division=0)),
        "ai_recall": float(recall_score(actual_ai, predicted_ai, zero_division=0)),
        "ai_f1": float(f1_score(actual_ai, predicted_ai, zero_division=0)),
        "human_false_positive_rate": float((predicted_ai[human_mask] == 1).mean()),
        "ai_roc_auc": float(roc_auc_score(actual_ai, ai_scores)),
    }


def main() -> None:
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stop and check the PyTorch GPU installation.")

    device = torch.device("cuda")
    print("=" * 65)
    print("DISTILBERT: FIRST CROSS-DOMAIN FINE-TUNING RUN")
    print("=" * 65)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory total (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print("Configuration:", {
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "precision": "FP32",
    })

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_in_df = load_split("test_in_domain.csv")
    test_unseen_df = load_split("test_unseen_domain.csv")

    print("\nDataset sizes:")
    print("Train:", len(train_df), train_df["label"].value_counts().sort_index().to_dict())
    print("Validation:", len(validation_df), validation_df["label"].value_counts().sort_index().to_dict())
    print("In-domain reference test:", len(test_in_df))
    print("Unseen ELI5 test:", len(test_unseen_df))

    # Confirm exact-text isolation between train, validation and both tests.
    train_texts = set(train_df["text"])
    reserved_texts = set(validation_df["text"]) | set(test_in_df["text"]) | set(test_unseen_df["text"])
    overlap = train_texts.intersection(reserved_texts)
    if overlap:
        raise ValueError(f"Found {len(overlap)} exact train/validation/test text overlaps.")
    print("Exact train/test text-overlap check: PASSED")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing prepared splits...")
    train_dataset = tokenize_frame(tokenizer, train_df)
    validation_dataset = tokenize_frame(tokenizer, validation_df)
    in_test_dataset = tokenize_frame(tokenizer, test_in_df)
    unseen_test_dataset = tokenize_frame(tokenizer, test_unseen_df)

    train_loader = make_loader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    validation_loader = make_loader(validation_dataset, EVAL_BATCH_SIZE)
    in_test_loader = make_loader(in_test_dataset, EVAL_BATCH_SIZE)
    unseen_test_loader = make_loader(unseen_test_dataset, EVAL_BATCH_SIZE)

    print("\nLoading pretrained DistilBERT classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS
    total_updates = updates_per_epoch * EPOCHS
    warmup_steps = int(total_updates * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    best_validation_auc = -1.0
    best_epoch = 0
    global_step = 0
    history = []

    print("\nStarting fine-tuning...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []

        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids, attention_mask, labels = [x.to(device, non_blocking=True) for x in batch]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            batch_losses.append(float(loss.detach().cpu()))

            is_update_step = (batch_index % GRADIENT_ACCUMULATION_STEPS == 0) or (batch_index == len(train_loader))
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} | Batch {batch_index}/{len(train_loader)} "
                    f"| Loss {np.mean(batch_losses[-100:]):.4f} "
                    f"| GPU allocated {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB"
                )

        validation_scores = predict_ai_scores(model, validation_loader, device)
        validation_labels = validation_df["label"].to_numpy()
        validation_auc = float(roc_auc_score((validation_labels == 0).astype(int), validation_scores))
        mean_train_loss = float(np.mean(batch_losses))

        history.append({
            "epoch": epoch,
            "mean_training_loss": mean_train_loss,
            "validation_ai_roc_auc": validation_auc,
            "optimizer_steps": global_step,
        })

        print(
            f"Epoch {epoch} complete | mean train loss={mean_train_loss:.4f} "
            f"| validation ROC-AUC={validation_auc:.4f}"
        )

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_epoch = epoch
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print("Saved current best validation checkpoint:", MODEL_DIR)

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # Reload the best validation checkpoint before final test evaluation.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nLoading best validation checkpoint for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    best_model.to(device)
    best_model.eval()

    validation_scores = predict_ai_scores(best_model, validation_loader, device)
    threshold = select_threshold(validation_df["label"].to_numpy(), validation_scores)
    print(f"Validation-selected threshold (max balanced accuracy): {threshold:.6f}")

    validation_metrics = calculate_metrics(
        validation_df["label"].to_numpy(), validation_scores, threshold
    )
    in_scores = predict_ai_scores(best_model, in_test_loader, device)
    unseen_scores = predict_ai_scores(best_model, unseen_test_loader, device)

    results = []
    for condition, frame, scores in [
        ("in_domain_reference", test_in_df, in_scores),
        ("unseen_ELI5", test_unseen_df, unseen_scores),
    ]:
        metrics = calculate_metrics(frame["label"].to_numpy(), scores, threshold)
        results.append({
            "model": "DistilBERT-base-uncased",
            "condition": condition,
            "seed": SEED,
            "best_epoch": best_epoch,
            "threshold": threshold,
            "test_size": len(frame),
            **metrics,
        })
        prediction_frame = pd.DataFrame({
            "row_index": np.arange(len(frame)),
            "label": frame["label"].to_numpy(),
            "ai_score": scores,
            "threshold": threshold,
            "predicted_label": np.where(scores >= threshold, 0, 1),
        })
        prediction_frame.to_csv(OUTPUT_DIR / f"predictions_{condition}.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "test_results.csv", index=False)

    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "epochs_configured": EPOCHS,
            "best_epoch_by_validation_roc_auc": best_epoch,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "threshold_selection": "maximum balanced accuracy on validation set",
            "validation_metrics_at_selected_threshold": validation_metrics,
            "training_records": len(train_df),
            "validation_records": len(validation_df),
            "in_domain_test_records": len(test_in_df),
            "unseen_eli5_test_records": len(test_unseen_df),
            "device": torch.cuda.get_device_name(0),
            "precision": "FP32",
        }, f, indent=2)

    print("\n" + "=" * 65)
    print("DISTILBERT CROSS-DOMAIN TEST RESULTS")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("\nSaved outputs to:", OUTPUT_DIR)
    print("NOTE: This is a single-seed, one-epoch exploratory transformer run.")
    print("Do not describe it as a multi-seed or statistically validated result.")


if __name__ == "__main__":
    main()
