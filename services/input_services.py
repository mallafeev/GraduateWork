from __future__ import annotations

from pathlib import Path

from domain.analysis_models import AnalysisArtifacts
from parsers.nexus_parser import alignment_summary, load_alignment, validate_alignment_file, validate_tree_file
from phylo.nj_builder import load_tree
from phylo.tree_utils import build_node_rows, summarize_tree


class AlignmentInspectionService:
    def inspect(self, path: str | Path) -> dict:
        return validate_alignment_file(path)


class TreeInspectionService:
    def inspect(self, path: str | Path) -> dict:
        return validate_tree_file(path)


class AlignmentLoadingService:
    def load(self, path: str | Path) -> AnalysisArtifacts:
        alignment = load_alignment(path)
        return AnalysisArtifacts(
            alignment=alignment,
            tree=None,
            alignment_info=alignment_summary(alignment),
            tree_info={},
            node_rows=[],
        )


class TreeLoadingService:
    def load(self, path: str | Path) -> AnalysisArtifacts:
        tree = load_tree(path)
        tree_info = summarize_tree(tree)
        tree_info.setdefault('build_method', 'Loaded from file')
        tree_info.setdefault('support_kind', 'observed')
        tree_info.setdefault('support_label', 'Bootstrap / support')
        return AnalysisArtifacts(
            alignment=None,
            tree=tree,
            alignment_info={},
            tree_info=tree_info,
            node_rows=build_node_rows(tree),
        )
