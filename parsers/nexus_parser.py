from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class NexusParseError(ValueError):
    """Raised when an input NEXUS file does not contain a readable alignment."""


def load_alignment(path: str | Path) -> MultipleSeqAlignment:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".fasta", ".fa", ".fas", ".aln"}:
        return AlignIO.read(str(path), "fasta")
    if suffix in {".nex", ".nexus", ".txt"}:
        return load_nexus_alignment(path)
    raise NexusParseError(f"Неподдерживаемый формат файла: {path.suffix}")


def load_nexus_alignment(path: str | Path) -> MultipleSeqAlignment:
    """Try Biopython first, then fall back to a lightweight parser for TreeBASE-like files."""
    path = Path(path)
    try:
        alignment = AlignIO.read(str(path), "nexus")
        _validate_alignment(alignment)
        return alignment
    except Exception:
        text = path.read_text(encoding="utf-8", errors="ignore")
        alignment = _parse_nexus_matrix(text)
        _validate_alignment(alignment)
        return alignment


def _validate_alignment(alignment: MultipleSeqAlignment) -> None:
    if len(alignment) == 0:
        raise NexusParseError("В файле не найдено ни одной последовательности.")
    lengths = {len(record.seq) for record in alignment}
    if len(lengths) != 1:
        raise NexusParseError("Последовательности не выровнены: длины различаются.")


def _parse_nexus_matrix(text: str) -> MultipleSeqAlignment:
    text = _strip_comments(text)
    matrix_match = re.search(r"\bMATRIX\b(.*?)\n\s*;", text, flags=re.IGNORECASE | re.DOTALL)
    if not matrix_match:
        raise NexusParseError("В NEXUS не найден блок MATRIX.")

    matrix_text = matrix_match.group(1)
    records: list[SeqRecord] = []
    for raw_line in matrix_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        taxon = parts[0]
        sequence = "".join(parts[1:]).upper()
        records.append(SeqRecord(Seq(sequence), id=taxon, name=taxon, description=""))

    if not records:
        raise NexusParseError("Не удалось извлечь последовательности из MATRIX.")
    return MultipleSeqAlignment(records)


def _strip_comments(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)


def alignment_summary(alignment: MultipleSeqAlignment) -> dict:
    taxa_count = len(alignment)
    length = alignment.get_alignment_length() if taxa_count else 0
    gap_count = 0
    total_count = taxa_count * length if taxa_count and length else 0
    for record in alignment:
        seq = str(record.seq)
        gap_count += sum(1 for ch in seq if ch in "-?")
    gap_fraction = (gap_count / total_count) if total_count else 0.0
    return {
        "taxa_count": taxa_count,
        "length": length,
        "gap_fraction": gap_fraction,
        "taxa": [record.id for record in alignment],
    }


def export_fasta(alignment: MultipleSeqAlignment, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    AlignIO.write(alignment, str(output_path), "fasta")
    return output_path
