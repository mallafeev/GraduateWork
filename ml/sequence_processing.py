from __future__ import annotations

import numpy as np
from Bio.Align import MultipleSeqAlignment

from .feature_constants import DNA_PLUS_GAP, VALID_DNA


class SequenceSanitizer:
    def sanitize(self, seq: str) -> str:
        return ''.join(ch.upper() for ch in seq if ch.upper() in DNA_PLUS_GAP)


class AlignmentSequenceMapper:
    def to_dict(self, alignment: MultipleSeqAlignment) -> dict[str, str]:
        return {record.id: str(record.seq) for record in alignment}


class PairwiseDistanceCalculator:
    def calculate(self, seq1: str, seq2: str) -> float:
        mismatches = 0
        valid = 0
        for a, b in zip(seq1, seq2):
            if a in {'-', '?'} or b in {'-', '?'}:
                continue
            if a not in VALID_DNA or b not in VALID_DNA:
                continue
            valid += 1
            if a != b:
                mismatches += 1
        return np.nan if valid == 0 else mismatches / valid


class GapFractionCalculator:
    def calculate(self, seq: str) -> float:
        return np.nan if not seq else sum(ch in {'-', '?'} for ch in seq) / len(seq)


class GCFractionCalculator:
    def calculate(self, seq: str) -> float:
        letters = [ch for ch in seq if ch in VALID_DNA]
        return np.nan if not letters else sum(ch in {'G', 'C'} for ch in letters) / len(letters)
