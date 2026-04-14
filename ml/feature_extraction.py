from __future__ import annotations

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree

from .alignment_statistics import AlignmentStatisticsCalculator
from .clade_statistics import CladeLeafCollector, CladeStatisticsCalculator, MeanPairwiseDistanceCalculator, SubtreeBalanceCalculator
from .feature_constants import DNA_PLUS_GAP, FEATURE_COLUMNS, VALID_DNA
from .feature_extractor import NodeFeatureRow, TreeFeatureExtractor
from .sequence_processing import AlignmentSequenceMapper, GapFractionCalculator, GCFractionCalculator, PairwiseDistanceCalculator, SequenceSanitizer

_sanitizer = SequenceSanitizer()
_mapper = AlignmentSequenceMapper()
_distance = PairwiseDistanceCalculator()
_gap = GapFractionCalculator()
_gc = GCFractionCalculator()
_alignment_stats = AlignmentStatisticsCalculator()
_leaf_collector = CladeLeafCollector()
_mpd = MeanPairwiseDistanceCalculator()
_clade_stats = CladeStatisticsCalculator()
_balance = SubtreeBalanceCalculator()
_extractor = TreeFeatureExtractor()


def sanitize_seq(seq: str) -> str:
    return _sanitizer.sanitize(seq)


def alignment_to_dict(alignment: MultipleSeqAlignment) -> dict[str, str]:
    return _mapper.to_dict(alignment)


def p_distance(seq1: str, seq2: str) -> float:
    return _distance.calculate(seq1, seq2)


def seq_gap_fraction(seq: str) -> float:
    return _gap.calculate(seq)


def seq_gc_fraction(seq: str) -> float:
    return _gc.calculate(seq)


def alignment_stats(seqs: dict[str, str]) -> dict[str, float]:
    return _alignment_stats.calculate(seqs)


def get_leaf_names(clade) -> list[str]:
    return _leaf_collector.get_leaf_names(clade)


def mean_pairwise_distance(names: list[str], seqs: dict[str, str]) -> float:
    return _mpd.calculate(names, seqs)


def clade_alignment_stats(names: list[str], seqs: dict[str, str]) -> dict[str, float]:
    return _clade_stats.calculate(names, seqs)


def subtree_balance(clade) -> float:
    return _balance.calculate(clade)


def extract_feature_rows(tree: Tree, alignment: MultipleSeqAlignment) -> list[NodeFeatureRow]:
    return _extractor.extract(tree, alignment)


__all__ = [
    'FEATURE_COLUMNS', 'VALID_DNA', 'DNA_PLUS_GAP', 'NodeFeatureRow',
    'sanitize_seq', 'alignment_to_dict', 'p_distance', 'seq_gap_fraction', 'seq_gc_fraction',
    'alignment_stats', 'get_leaf_names', 'mean_pairwise_distance', 'clade_alignment_stats',
    'subtree_balance', 'extract_feature_rows'
]
