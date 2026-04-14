from __future__ import annotations

import numpy as np

from .sequence_processing import GCFractionCalculator, GapFractionCalculator, SequenceSanitizer
from .feature_constants import VALID_DNA


class AlignmentStatisticsCalculator:
    def __init__(self) -> None:
        self.sanitizer = SequenceSanitizer()
        self.gap_calculator = GapFractionCalculator()
        self.gc_calculator = GCFractionCalculator()

    def calculate(self, seqs: dict[str, str]) -> dict[str, float]:
        items = [self.sanitizer.sanitize(s) for s in seqs.values()]
        if not items:
            return {
                'taxa_count': np.nan,
                'alignment_length': np.nan,
                'gap_fraction_global': np.nan,
                'gc_mean_global': np.nan,
                'gc_std_global': np.nan,
                'variable_site_fraction_global': np.nan,
            }
        n = len(items)
        length = max(len(s) for s in items)
        items = [s.ljust(length, '-') for s in items]
        gap_fraction_global = np.mean([self.gap_calculator.calculate(s) for s in items])
        gc_values = [self.gc_calculator.calculate(s) for s in items]
        variable_site_fraction = self._calculate_variable_site_fraction(items, length)
        return {
            'taxa_count': float(n),
            'alignment_length': float(length),
            'gap_fraction_global': float(gap_fraction_global),
            'gc_mean_global': float(np.nanmean(gc_values)),
            'gc_std_global': float(np.nanstd(gc_values)),
            'variable_site_fraction_global': float(variable_site_fraction),
        }

    def _calculate_variable_site_fraction(self, items: list[str], length: int) -> float:
        variable_sites = 0
        informative_positions = 0
        for i in range(length):
            col = [s[i] for s in items if s[i] in VALID_DNA]
            if len(col) < 2:
                continue
            informative_positions += 1
            if len(set(col)) > 1:
                variable_sites += 1
        return variable_sites / informative_positions if informative_positions else np.nan
