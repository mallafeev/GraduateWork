from __future__ import annotations

from pathlib import Path

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree

from domain.analysis_models import AnalysisArtifacts
from services.input_services import AlignmentInspectionService, AlignmentLoadingService, TreeInspectionService, TreeLoadingService
from services.tree_services import TreeBuildingService


class AnalysisService:
    def __init__(self) -> None:
        self._predictor = None
        self._model_loading_service = None
        self._bootstrap_prediction_service = None
        self.alignment_inspection_service = AlignmentInspectionService()
        self.tree_inspection_service = TreeInspectionService()
        self.alignment_loading_service = AlignmentLoadingService()
        self.tree_loading_service = TreeLoadingService()
        self.tree_building_service = TreeBuildingService()

    def _ensure_predictor_services(self) -> None:
        if self._predictor is None:
            from ml.model_inference import RandomForestBootstrapPredictor
            from services.model_services import ModelLoadingService
            from services.prediction_services import BootstrapPredictionService

            self._predictor = RandomForestBootstrapPredictor()
            self._model_loading_service = ModelLoadingService(self._predictor)
            self._bootstrap_prediction_service = BootstrapPredictionService(self._predictor)

    def inspect_alignment_file(self, path: str | Path) -> dict:
        return self.alignment_inspection_service.inspect(path)

    def inspect_tree_file(self, path: str | Path) -> dict:
        return self.tree_inspection_service.inspect(path)

    def load_alignment(self, path: str | Path) -> AnalysisArtifacts:
        return self.alignment_loading_service.load(path)

    def build_tree_from_alignment(self, alignment: MultipleSeqAlignment, method: str = 'Neighbor Joining', output_path: str | Path | None = None) -> AnalysisArtifacts:
        return self.tree_building_service.build_from_alignment(alignment, method=method, output_path=output_path)

    def load_tree(self, path: str | Path) -> AnalysisArtifacts:
        return self.tree_loading_service.load(path)

    def load_model(self, model_path: str | Path, metadata_path: str | Path | None = None) -> tuple[dict, bool]:
        self._ensure_predictor_services()
        return self._model_loading_service.load(model_path, metadata_path)

    def predict_bootstrap(self, tree: Tree, alignment: MultipleSeqAlignment) -> list[dict]:
        self._ensure_predictor_services()
        return self._bootstrap_prediction_service.predict(tree, alignment)


__all__ = ['AnalysisService', 'AnalysisArtifacts']
