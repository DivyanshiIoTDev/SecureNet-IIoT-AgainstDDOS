from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from db import MongoManager
from predictor import DDOSPredictor

router = APIRouter(prefix="/api", tags=["prediction"])


class PredictRequest(BaseModel):
    ip_address: str = Field(..., min_length=7, max_length=45)


def build_router(predictor: DDOSPredictor, mongo: MongoManager) -> APIRouter:
    @router.post("/predict")
    def predict(payload: PredictRequest):
        try:
            prediction = predictor.predict_one()
            log_entry = mongo.log_prediction(payload.ip_address, prediction)
            logs = mongo.get_logs(payload.ip_address)
            return {
                "ip_address": payload.ip_address,
                "prediction": prediction,
                "timestamp": log_entry["timestamp"],
                "logs": logs,
            }
        except Exception as exc:  # Important surface-level API error handling.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router

