from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.model_utils import analyze_digit, build_model_snapshot, export_model_snapshot, get_model, train_on_example


app = FastAPI(title='Ultimate Neural Network Visualizer ML Server', version='1.0.0')


class AnalyzePayload(BaseModel):
    pixels: list[list[float]] = Field(..., min_length=28, max_length=28)
    adjustments: dict[str, Any] | None = None
    includeBackprop: bool = True


class TrainPayload(AnalyzePayload):
    label: int = Field(..., ge=0, le=9)


@app.on_event('startup')
def warm_model() -> None:
    get_model()


@app.get('/health')
def health() -> dict[str, str]:
    get_model()
    return {'status': 'ok'}


@app.get('/model')
def model_snapshot() -> dict[str, Any]:
    snapshot = build_model_snapshot()
    export_model_snapshot()
    return snapshot


@app.post('/analyze')
def analyze(payload: AnalyzePayload) -> dict[str, Any]:
    return analyze_digit(payload.pixels, adjustments=payload.adjustments, include_backprop=payload.includeBackprop)


@app.post('/train-step')
def train_step(payload: TrainPayload) -> dict[str, Any]:
    return train_on_example(payload.pixels, label=payload.label)