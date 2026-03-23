from __future__ import annotations

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceMatrix


MISSING_CHARS = {"-", "?", "N"}


def p_distance(seq1: str, seq2: str) -> float:
    valid = 0
    diff = 0
    for a, b in zip(seq1.upper(), seq2.upper()):
        if a in MISSING_CHARS or b in MISSING_CHARS:
            continue
        valid += 1
        if a != b:
            diff += 1
    return (diff / valid) if valid else 0.0


def build_distance_matrix(alignment: MultipleSeqAlignment) -> DistanceMatrix:
    names = [record.id for record in alignment]
    matrix: list[list[float]] = []
    sequences = [str(record.seq) for record in alignment]

    for i in range(len(sequences)):
        row: list[float] = [0.0]
        for j in range(i):
            row.append(p_distance(sequences[i], sequences[j]))
        matrix.append(row)

    return DistanceMatrix(names=names, matrix=matrix)
