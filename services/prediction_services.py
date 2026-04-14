from __future__ import annotations

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree

from ml.model_inference import RandomForestBootstrapPredictor


class BootstrapPredictionService:
    def __init__(self, predictor: RandomForestBootstrapPredictor | None = None) -> None:
        self.predictor = predictor or RandomForestBootstrapPredictor()

    def predict(self, tree: Tree, alignment: MultipleSeqAlignment) -> list[dict]:
        return self.predictor.predict_tree(tree, alignment)
