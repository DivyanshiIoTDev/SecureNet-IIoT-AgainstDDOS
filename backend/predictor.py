from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd

# Exact 36-feature order already used in your existing Api_calling_pkl.py.
SELECTED_FEATURES: List[str] = [
    "IAT", "Min", "Magnitue", "fin_flag_number", "psh_flag_number", "syn_flag_number",
    "Tot sum", "Protocol Type", "ICMP", "Header_Length", "rst_count", "Radius",
    "fin_count", "syn_count", "flow_duration", "Srate", "Number", "AVG", "Rate",
    "Variance", "HTTPS", "urg_count", "Duration", "Weight", "HTTP", "Max", "Tot size",
    "Covariance", "ack_count", "Std", "rst_flag_number", "UDP", "ack_flag_number",
    "SSH", "TCP", "LLC",
]


class DDOSPredictor:
    """Loads model and test dataset once, then predicts one row sequentially per request."""

    def __init__(self, model_path: str, parquet_path: str) -> None:
        self.model_path = Path(model_path)
        self.parquet_path = Path(parquet_path)
        self.model = None
        self.dataset = pd.DataFrame()
        self.current_index = 0

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.parquet_path}")

        self.model = joblib.load(self.model_path)
        self.dataset = pd.read_parquet(self.parquet_path)

        missing = [f for f in SELECTED_FEATURES if f not in self.dataset.columns]
        if missing:
            raise ValueError(f"Missing selected features in dataset: {missing}")

    def _next_row(self) -> pd.Series:
        """Returns exactly one row and moves pointer sequentially (wrap-around)."""
        if self.dataset.empty:
            raise ValueError("Loaded dataset is empty")

        row = self.dataset.iloc[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.dataset)
        return row

    def predict_one(self) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        row = self._next_row()

        # Build feature vector in exact training order for one sample.
        one_sample = pd.DataFrame([{f: row[f] for f in SELECTED_FEATURES}])
        one_sample = one_sample.apply(pd.to_numeric, errors="coerce")

        if one_sample.isnull().any().any():
            bad_cols = one_sample.columns[one_sample.isnull().any()].tolist()
            raise ValueError(f"Invalid numeric values in features: {bad_cols}")

        pred = self.model.predict(one_sample)
        pred_val = int(pred[0])
        return "ATTACK" if pred_val == 1 else "NORMAL"
