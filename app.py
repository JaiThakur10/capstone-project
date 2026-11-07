
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / "milestone2"
    / "distilbert_cross_domain_seed42"
    / "best_model"
)

MODEL_NAME = "DistilBERT-base-uncased"
MAX_LENGTH = 128

# Selected on the validation set by maximum balanced accuracy.
# Source: seed-42 training terminal output.
AI_THRESHOLD = 0.526899

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

inference_lock = Lock()


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 55)
    print("STARTING AI TEXT DETECTION API")
    print("=" * 55)

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}"
        )

    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (MODEL_DIR / filename).exists()
    ]

    if missing_files:
        raise RuntimeError(
            f"Missing model files: {missing_files}"
        )

    print("Device:", DEVICE)
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    print("Loading fine-tuned DistilBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    if model.config.id2label.get(0) != "AI-generated":
        raise RuntimeError(
            "Unexpected class 0 mapping in model configuration."
        )

    if model.config.id2label.get(1) != "Human-written":
        raise RuntimeError(
            "Unexpected class 1 mapping in model configuration."
        )

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = DEVICE

    print("Model loaded successfully.")
    print("AI threshold:", AI_THRESHOLD)
    print("Maximum sequence length:", MAX_LENGTH)
    print("=" * 55)

    yield

    print("Shutting down API...")

    app.state.model = None
    app.state.tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="AI-Generated Text Detection API",
    description=(
        "Research prototype for detecting AI-generated "
        "and human-written text using fine-tuned DistilBERT."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class PredictionRequest(BaseModel):

    text: str = Field(
        min_length=20,
        max_length=12000,
        description="Text to analyse.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError("Text cannot be empty.")

        if len(value) < 20:
            raise ValueError(
                "Please enter at least 20 characters."
            )

        return value


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "AI Text Detection API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():

    model_loaded = getattr(
        app.state,
        "model",
        None,
    ) is not None

    return {
        "status": "healthy" if model_loaded else "not_ready",
        "model_loaded": model_loaded,
        "model": MODEL_NAME,
        "device": str(DEVICE),
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    model = getattr(app.state, "model", None)
    tokenizer = getattr(app.state, "tokenizer", None)

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    text = request.text

    try:

        # Count the original tokens before truncation.
        original_tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        original_token_count = len(original_tokens)

        # Match the maximum sequence length used in training.
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        tokens_processed = inputs["input_ids"].shape[1]

        truncated = original_token_count > tokens_processed

        inputs = {
            key: value.to(app.state.device)
            for key, value in inputs.items()
        }

        # Perform inference.
        with inference_lock:

            with torch.inference_mode():

                outputs = model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1,
                )[0]

        ai_score = float(probabilities[0].item())
        human_score = float(probabilities[1].item())

        # Apply the threshold selected on validation data.
        if ai_score >= AI_THRESHOLD:

            prediction = "AI-generated"
            predicted_score = ai_score

        else:

            prediction = "Human-written"
            predicted_score = human_score

        return {
            "prediction": prediction,

            "scores": {
                "ai_generated": round(ai_score, 6),
                "human_written": round(human_score, 6),
            },

            "predicted_class_score": round(
                predicted_score, 6
            ),

            "threshold": AI_THRESHOLD,

            "text_analysis": {
                "original_tokens": original_token_count,
                "tokens_processed": tokens_processed,
                "truncated": truncated,
                "maximum_tokens": MAX_LENGTH,
            },

            "model": MODEL_NAME,

            "device": str(app.state.device),

            "disclaimer": (
                "This result is a model estimate, not proof "
                "of authorship. AI-text detection can produce "
                "false positives and false negatives."
            ),
        }

    except HTTPException:
        raise

    except Exception as error:

        print("Prediction error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="An error occurred while analysing the text.",
        ) from error


# --------------------------------------------------
# RUN DIRECTLY (OPTIONAL)
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
    )
