from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree

from ml.model_inference import RandomForestBootstrapPredictor
from parsers.nexus_parser import alignment_summary, load_alignment
from phylo.nj_builder import build_neighbor_joining_tree, load_tree, save_tree_newick
from phylo.tree_utils import build_node_rows, summarize_tree


@dataclass
class AnalysisArtifacts:
    alignment: MultipleSeqAlignment | None
    tree: Tree | None
    alignment_info: dict
    tree_info: dict
    node_rows: list[dict]


class AnalysisService:
    def __init__(self) -> None:
        self.predictor = RandomForestBootstrapPredictor()

    def load_alignment(self, path: str | Path) -> AnalysisArtifacts:
        alignment = load_alignment(path)
        return AnalysisArtifacts(
            alignment=alignment,
            tree=None,
            alignment_info=alignment_summary(alignment),
            tree_info={},
            node_rows=[],
        )

    def build_tree_from_alignment(self, alignment: MultipleSeqAlignment, output_path: str | Path | None = None) -> AnalysisArtifacts:
        tree = build_neighbor_joining_tree(alignment)
        if output_path is not None:
            save_tree_newick(tree, output_path)
        return AnalysisArtifacts(
            alignment=alignment,
            tree=tree,
            alignment_info=alignment_summary(alignment),
            tree_info=summarize_tree(tree),
            node_rows=build_node_rows(tree),
        )

    def load_tree(self, path: str | Path) -> AnalysisArtifacts:
        tree = load_tree(path)
        return AnalysisArtifacts(
            alignment=None,
            tree=tree,
            alignment_info={},
            tree_info=summarize_tree(tree),
            node_rows=build_node_rows(tree),
        )

    def load_model(self, model_path: str | Path, metadata_path: str | Path | None = None) -> dict:
        self.predictor.load(model_path, metadata_path)
        return self.predictor.metadata

    def predict_bootstrap(self, tree: Tree, alignment: MultipleSeqAlignment) -> list[dict]:
        return self.predictor.predict_tree(tree, alignment)
