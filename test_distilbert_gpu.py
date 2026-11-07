
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"

# MAGE labels:
# 0 = AI-generated
# 1 = Human-written

ID2LABEL = {
    0: "AI-generated",
    1: "Human-written",
}

LABEL2ID = {
    "AI-generated": 0,
    "Human-written": 1,
}


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check PyTorch installation."
        )

    device = torch.device("cuda")

    print("=" * 55)
    print("DISTILBERT GPU TEST")
    print("=" * 55)

    print("GPU:", torch.cuda.get_device_name(0))

    print("\nDownloading/loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("\nDownloading/loading pretrained model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("\nMoving model to NVIDIA GPU...")

    model.to(device)
    model.eval()

    print("Model device:", next(model.parameters()).device)

    # Small test batch. These are test sentences only.
    test_texts = [
        "Artificial intelligence can generate text using patterns learned from large collections of documents.",
        "Yesterday I visited the local market and bought vegetables for dinner.",
    ]

    print("\nTokenizing test sentences...")

    inputs = tokenizer(
        test_texts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("Input shape:", inputs["input_ids"].shape)

    print("\nRunning a small GPU forward pass...")

    with torch.inference_mode():
        outputs = model(**inputs)

    print("Output shape:", outputs.logits.shape)

    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated(0) / (1024 ** 2),
            2
        ),
        "MB"
    )

    print(
        "GPU memory reserved:",
        round(
            torch.cuda.memory_reserved(0) / (1024 ** 2),
            2
        ),
        "MB"
    )

    print("\nSUCCESS: DistilBERT ran on your NVIDIA GPU.")
    print("This was a loading and inference test, not training.")


if __name__ == "__main__":
    main()
