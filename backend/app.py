from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import MongoManager
from predictor import DDOSPredictor
from routes.predict import build_router

app = FastAPI(title="AI DDoS Detection Dashboard API", version="1.0.0")

# Allow frontend communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.getenv("MODEL_PATH", "backend/model.pkl")
parquet_path = os.getenv("DATASET_PATH", "backend/test_preprocessed.parquet")
predictor = DDOSPredictor(model_path=model_path, parquet_path=parquet_path)
mongo = MongoManager()


@app.on_event("startup")
def startup_event() -> None:
    predictor.load()


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "ddos-detection-api"}


app.include_router(build_router(predictor, mongo))
