from __future__ import annotations

from pathlib import Path

from Bio.Align import MultipleSeqAlignment

from domain.analysis_models import AnalysisArtifacts
from parsers.nexus_parser import alignment_summary
from phylo.nj_builder import build_tree_by_method, save_tree_newick
from phylo.tree_utils import build_node_rows, summarize_tree


class TreeBuildMetadataAssembler:
    def assemble(self, build_result) -> dict:
        tree_info = summarize_tree(build_result.tree)
        tree_info['build_method'] = build_result.method
        if build_result.log_likelihood is not None:
            tree_info['log_likelihood'] = build_result.log_likelihood
        if build_result.support_label:
            tree_info['support_label'] = build_result.support_label
        if build_result.support_kind:
            tree_info['support_kind'] = build_result.support_kind
        if build_result.metadata:
            tree_info.update(build_result.metadata)
        return tree_info


class TreePersistenceService:
    def save_newick(self, tree, output_path: str | Path) -> Path:
        return save_tree_newick(tree, output_path)


class TreeBuildingService:
    def __init__(self) -> None:
        self.metadata_assembler = TreeBuildMetadataAssembler()
        self.persistence_service = TreePersistenceService()

    def build_from_alignment(self, alignment: MultipleSeqAlignment, method: str = 'Neighbor Joining', output_path: str | Path | None = None) -> AnalysisArtifacts:
        build_result = build_tree_by_method(alignment, method)
        if output_path is not None:
            self.persistence_service.save_newick(build_result.tree, output_path)
        tree_info = self.metadata_assembler.assemble(build_result)
        return AnalysisArtifacts(
            alignment=alignment,
            tree=build_result.tree,
            alignment_info=alignment_summary(alignment),
            tree_info=tree_info,
            node_rows=build_node_rows(build_result.tree, support_source=build_result.support_kind or 'observed'),
        )
