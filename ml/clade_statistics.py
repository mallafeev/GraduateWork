from __future__ import annotations

import numpy as np

from .sequence_processing import GapFractionCalculator, GCFractionCalculator, PairwiseDistanceCalculator, SequenceSanitizer


class CladeLeafCollector:
    def get_leaf_names(self, clade) -> list[str]:
        return [leaf.name for leaf in clade.get_terminals() if leaf.name]


class MeanPairwiseDistanceCalculator:
    def __init__(self) -> None:
        self.sanitizer = SequenceSanitizer()
        self.distance_calculator = PairwiseDistanceCalculator()

    def calculate(self, names: list[str], seqs: dict[str, str]) -> float:
        if len(names) < 2:
            return 0.0
        vals = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s1 = self.sanitizer.sanitize(seqs.get(names[i], ''))
                s2 = self.sanitizer.sanitize(seqs.get(names[j], ''))
                if not s1 or not s2:
                    continue
                d = self.distance_calculator.calculate(s1, s2)
                if not np.isnan(d):
                    vals.append(d)
        return np.nan if not vals else float(np.mean(vals))


class CladeStatisticsCalculator:
    def __init__(self) -> None:
        self.gap_calculator = GapFractionCalculator()
        self.gc_calculator = GCFractionCalculator()
        self.sanitizer = SequenceSanitizer()
        self.distance_calculator = MeanPairwiseDistanceCalculator()

    def calculate(self, names: list[str], seqs: dict[str, str]) -> dict[str, float]:
        sub = {k: seqs[k] for k in names if k in seqs}
        if not sub:
            return {
                'gap_fraction_clade': np.nan,
                'gc_mean_clade': np.nan,
                'gc_std_clade': np.nan,
                'mean_pairwise_pdist_clade': np.nan,
            }
        gap_fraction_clade = float(np.nanmean([self.gap_calculator.calculate(self.sanitizer.sanitize(v)) for v in sub.values()]))
        gc_vals = [self.gc_calculator.calculate(self.sanitizer.sanitize(v)) for v in sub.values()]
        return {
            'gap_fraction_clade': gap_fraction_clade,
            'gc_mean_clade': float(np.nanmean(gc_vals)),
            'gc_std_clade': float(np.nanstd(gc_vals)),
            'mean_pairwise_pdist_clade': self.distance_calculator.calculate(list(sub.keys()), sub),
        }


class SubtreeBalanceCalculator:
    def calculate(self, clade) -> float:
        children = clade.clades
        if len(children) < 2:
            return 0.0
        sizes = [len(ch.get_terminals()) for ch in children]
        total = sum(sizes)
        return 0.0 if total == 0 else abs(sizes[0] - sizes[1]) / total
