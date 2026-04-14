from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .feature_constants import FEATURE_COLUMNS


class ModelInferenceError(RuntimeError):
    pass


class FeatureImportanceRepository:
    def load(self, model, model_path: Path) -> dict[str, float]:
        csv_path = model_path.with_name('feature_importance.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if {'feature', 'importance'}.issubset(df.columns):
                return {str(r['feature']): float(r['importance']) for _, r in df.iterrows()}
        if hasattr(model, 'feature_importances_'):
            values = list(getattr(model, 'feature_importances_'))
            return {col: float(values[idx]) for idx, col in enumerate(FEATURE_COLUMNS) if idx < len(values)}
        return {}


class MetadataRepository:
    def load(self, metadata_path: str | Path | None) -> tuple[dict, str]:
        if not metadata_path:
            return {}, ''
        metadata_file = Path(metadata_path).resolve()
        if not metadata_file.exists():
            return {}, ''
        data = json.loads(metadata_file.read_text(encoding='utf-8'))
        return data, str(metadata_file)


class SerializedModelLoader:
    def load(self, model_path: Path):
        return joblib.load(model_path)
