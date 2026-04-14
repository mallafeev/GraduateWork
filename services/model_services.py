from __future__ import annotations

from pathlib import Path

from ml.model_inference import RandomForestBootstrapPredictor


class ModelCatalogService:
    def create_metadata_payload(self, predictor: RandomForestBootstrapPredictor) -> dict:
        return predictor.metadata


class ModelLoadingService:
    def __init__(self, predictor: RandomForestBootstrapPredictor | None = None) -> None:
        self.predictor = predictor or RandomForestBootstrapPredictor()
        self.catalog_service = ModelCatalogService()

    def load(self, model_path: str | Path, metadata_path: str | Path | None = None) -> tuple[dict, bool]:
        loaded_now = self.predictor.load(model_path, metadata_path)
        return self.catalog_service.create_metadata_payload(self.predictor), loaded_now
