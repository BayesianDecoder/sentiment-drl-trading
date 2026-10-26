"""
FastAPI ML service — live sentiment inference using the project's trained FinBERT-LoRA adapter.

Run:
    cd backend/fastapi_ml
    ../../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
import os
import time

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

import live_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_DIR = os.path.join(ROOT, "saved_models", "finbert_peft_all_artifacts_v2", "saved_models", "finbert_lora_adapter")
TOKENIZER_DIR = os.path.join(ROOT, "saved_models", "finbert_peft_all_artifacts_v2", "saved_models", "finbert_tokenizer")
BASE_MODEL = "ProsusAI/finbert"

app = FastAPI(title="Sentiment-DRL-Trading ML Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dashboard-only demo; restrict this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tokenizer = None
_id2label = None


def _load_model():
    global _model, _tokenizer, _id2label
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
    _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}


@app.on_event("startup")
async def startup():
    _load_model()
    live_pipeline.set_finbert(_model, _tokenizer, _id2label)
    asyncio.create_task(live_pipeline.poll_loop())


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict
    inference_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest):
    start = time.time()
    inputs = _tokenizer(req.text, return_tensors="pt", truncation=True, max_length=64)
    logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    probabilities = {_id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
    best_idx = int(probs.argmax())
    inference_ms = (time.time() - start) * 1000

    return PredictResponse(
        text=req.text,
        label=_id2label[best_idx],
        confidence=round(float(probs[best_idx]), 4),
        probabilities=probabilities,
        inference_ms=round(inference_ms, 2),
    )


@app.get("/live/status")
def live_status():
    """Last poll timestamp, whether NSE is currently open, and how often we poll."""
    return live_pipeline.get_live_status()


@app.get("/live/state")
def live_state():
    """Latest live prices, sentiment, A2C's current recommendation per stock, and fresh news."""
    return live_pipeline.get_live_state()


@app.get("/nlp-model-comparison")
def nlp_model_comparison():
    """Static final results from Part A's 7-architecture benchmark (see project README/MLflow)."""
    return {
        "models": [
            {"name": "Logistic Regression (tuned)", "accuracy": 0.81, "f1_weighted": 0.81, "category": "classical"},
            {"name": "SVM (tuned)", "accuracy": 0.81, "f1_weighted": 0.81, "category": "classical"},
            {"name": "Random Forest (tuned)", "accuracy": 0.78, "f1_weighted": 0.78, "category": "classical"},
            {"name": "LightGBM (tuned)", "accuracy": 0.79, "f1_weighted": 0.79, "category": "classical"},
            {"name": "XGBoost (tuned)", "accuracy": 0.79, "f1_weighted": 0.79, "category": "classical"},
            {"name": "TextCNN (tuned)", "accuracy": 0.7639, "f1_weighted": 0.7636, "category": "deep_learning"},
            {"name": "BiLSTM (tuned)", "accuracy": 0.7731, "f1_weighted": 0.7733, "category": "deep_learning"},
            {"name": "CNN-BiLSTM-Attention (tuned)", "accuracy": 0.7646, "f1_weighted": 0.7642, "category": "deep_learning"},
            {"name": "Transformer (scratch, tuned)", "accuracy": 0.7773, "f1_weighted": 0.7771, "category": "deep_learning"},
            {"name": "FinBERT + LoRA", "accuracy": 0.8654, "f1_weighted": 0.8652, "category": "transformer"},
        ]
    }
