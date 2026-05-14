from __future__ import annotations

from datetime import datetime, UTC
import os

from pymongo import MongoClient
from pymongo.collection import Collection

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "ddos_dashboard")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "ip_predictions")


class MongoManager:
    def __init__(self) -> None:
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection: Collection = self.db[COLLECTION_NAME]

    def log_prediction(self, ip_address: str, prediction: str) -> dict:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"prediction": prediction, "timestamp": timestamp}

        # If IP already exists, append under logs. Otherwise, create new doc.
        self.collection.update_one(
            {"ip_address": ip_address},
            {"$push": {"logs": log_entry}, "$setOnInsert": {"ip_address": ip_address}},
            upsert=True,
        )
        return log_entry

    def get_logs(self, ip_address: str) -> list[dict]:
        doc = self.collection.find_one({"ip_address": ip_address}, {"_id": 0, "logs": 1})
        return doc.get("logs", []) if doc else []
