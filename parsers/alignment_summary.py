from __future__ import annotations

from pathlib import Path

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment


class AlignmentSummaryBuilder:
    def build(self, alignment: MultipleSeqAlignment) -> dict:
        taxa_count = len(alignment)
        length = alignment.get_alignment_length() if taxa_count else 0
        gap_count = 0
        total_count = taxa_count * length if taxa_count and length else 0
        for record in alignment:
            seq = str(record.seq)
            gap_count += sum(1 for ch in seq if ch in '-?')
        gap_fraction = (gap_count / total_count) if total_count else 0.0
        return {
            'taxa_count': taxa_count,
            'length': length,
            'gap_fraction': gap_fraction,
            'taxa': [record.id for record in alignment],
        }


class AlignmentExporter:
    def export_fasta(self, alignment: MultipleSeqAlignment, output_path: str | Path) -> Path:
        output = Path(output_path)
        AlignIO.write(alignment, str(output), 'fasta')
        return output
