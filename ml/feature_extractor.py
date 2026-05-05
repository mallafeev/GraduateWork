# ml/feature_extractor.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree
from .alignment_statistics import AlignmentStatisticsCalculator
from .clade_statistics import CladeLeafCollector, CladeStatisticsCalculator, SubtreeBalanceCalculator
from .sequence_processing import AlignmentSequenceMapper

@dataclass
class NodeFeatureRow:
    node_id: str
    features: dict[str, float]

class NodeFeatureFactory:
    def create(self, node_id: str, features: dict[str, float]) -> NodeFeatureRow:
        return NodeFeatureRow(node_id=node_id, features=features)

class TreeFeatureExtractor:
    def __init__(self) -> None:
        self.sequence_mapper = AlignmentSequenceMapper()
        self.alignment_stats_calculator = AlignmentStatisticsCalculator()
        self.leaf_collector = CladeLeafCollector()
        self.clade_stats_calculator = CladeStatisticsCalculator()
        self.balance_calculator = SubtreeBalanceCalculator()
        self.row_factory = NodeFeatureFactory()

    def extract(self, tree: Tree, alignment: MultipleSeqAlignment) -> list[NodeFeatureRow]:
        seqs = self.sequence_mapper.to_dict(alignment)
        global_stats = self.alignment_stats_calculator.calculate(seqs)
        depths = tree.depths()
        
        # Считаем максимальную глубину дерева для нормализации
        max_depth = max(depths.values()) if depths else 1.0
        
        rows: list[NodeFeatureRow] = []
        for idx, clade in enumerate(tree.get_nonterminals(order='level'), start=1):
            rows.append(self._build_row(idx, clade, depths, seqs, global_stats, max_depth))
        return rows

    def _build_row(self, idx: int, clade, depths, seqs: dict[str, str], global_stats: dict[str, float], max_depth: float) -> NodeFeatureRow:
        branch_length = max(float(clade.branch_length or 0.0), 0.0)
        depth = float(depths.get(clade, 0.0))
        leaf_names = self.leaf_collector.get_leaf_names(clade)
        n_leaves = len(leaf_names)
        clade_stats = self.clade_stats_calculator.calculate(leaf_names, seqs)
        child_branch_lengths = [max(float(ch.branch_length or 0.0), 0.0) for ch in clade.clades]

        # Формируем ПОЛНЫЙ набор признаков (старые + новые)
        features = {
            # === СТАРЫЕ ПРИЗНАКИ (для моделей v1, v2) ===
            'branch_length': branch_length,
            'depth': depth,
            'n_leaves_subtree': float(n_leaves),
            'subtree_fraction': float(n_leaves / global_stats['taxa_count']) if global_stats['taxa_count'] else np.nan,
            'subtree_balance': float(self.balance_calculator.calculate(clade)),
            'mean_child_branch_length': float(np.mean(child_branch_lengths)) if child_branch_lengths else 0.0,
            'std_child_branch_length': float(np.std(child_branch_lengths)) if child_branch_lengths else 0.0,
            
            # === НОВЫЕ ПРИЗНАКИ (для модели v3) ===
            'n_children': float(len(clade.clades)),
            'clade_size_log': float(np.log1p(n_leaves)),
            'relative_branch_length': float(branch_length / max_depth) if max_depth > 0 else 0.0,
            'depth_ratio': float(depth / max_depth) if max_depth > 0 else 0.0,
            
            # === ГЛОБАЛЬНЫЕ И ЛОКАЛЬНЫЕ СТАТИСТИКИ ===
            **global_stats,
            **clade_stats,
        }
        return self.row_factory.create(node_id=f'N{idx}', features=features)