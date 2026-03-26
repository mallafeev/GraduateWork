from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree

from ml.feature_extraction import FEATURE_COLUMNS, extract_feature_rows


class ModelInferenceError(RuntimeError):
    pass


class RandomForestBootstrapPredictor:
    def __init__(self) -> None:
        self.model = None
        self.metadata = {}
        self.model_path = ""

    def load(self, model_path: str | Path, metadata_path: str | Path | None = None) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelInferenceError(f"Файл модели не найден: {model_path}")
        self.model = joblib.load(model_path)
        self.model_path = str(model_path)
        if metadata_path:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_tree(self, tree: Tree, alignment: MultipleSeqAlignment) -> List[dict]:
        if self.model is None:
            raise ModelInferenceError("Модель не загружена.")
        feature_rows = extract_feature_rows(tree, alignment)
        if not feature_rows:
            return []

        X = pd.DataFrame([row.features for row in feature_rows])
        # гарантируем порядок колонок
        for col in FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = pd.NA
        X = X[FEATURE_COLUMNS]

        preds = self.model.predict(X)
        results = []
        for row, pred in zip(feature_rows, preds):
            pred = float(pred)
            pred = max(0.0, min(100.0, pred))
            results.append({
                "node_id": row.node_id,
                "predicted_bootstrap": pred,
                **row.features,
            })
        return results
