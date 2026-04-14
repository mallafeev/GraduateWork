from __future__ import annotations

import re
from pathlib import Path

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .file_kind import TextFileReader, CommentStripper
from .validation import AlignmentFileValidator, NexusParseError


class NexusMatrixParser:
    def parse(self, text: str) -> MultipleSeqAlignment:
        stripped = CommentStripper.strip(text)
        matrix_match = re.search(r"\bMATRIX\b(.*?)\n\s*;", stripped, flags=re.IGNORECASE | re.DOTALL)
        if not matrix_match:
            raise NexusParseError('В NEXUS не найден блок MATRIX.')
        matrix_text = matrix_match.group(1)
        records: list[SeqRecord] = []
        for raw_line in matrix_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('['):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            taxon = parts[0]
            sequence = ''.join(parts[1:]).upper()
            records.append(SeqRecord(Seq(sequence), id=taxon, name=taxon, description=''))
        if not records:
            raise NexusParseError('Не удалось извлечь последовательности из MATRIX.')
        return MultipleSeqAlignment(records)


class AlignmentIntegrityValidator:
    def validate(self, alignment: MultipleSeqAlignment) -> None:
        if len(alignment) == 0:
            raise NexusParseError('В файле не найдено ни одной последовательности.')
        lengths = {len(record.seq) for record in alignment}
        if len(lengths) != 1:
            raise NexusParseError('Последовательности не выровнены: длины различаются.')


class AlignmentLoader:
    def __init__(
        self,
        file_validator: AlignmentFileValidator | None = None,
        text_reader: TextFileReader | None = None,
        matrix_parser: NexusMatrixParser | None = None,
        integrity_validator: AlignmentIntegrityValidator | None = None,
    ) -> None:
        self.file_validator = file_validator or AlignmentFileValidator()
        self.text_reader = text_reader or TextFileReader()
        self.matrix_parser = matrix_parser or NexusMatrixParser()
        self.integrity_validator = integrity_validator or AlignmentIntegrityValidator()

    def load(self, path: str | Path) -> MultipleSeqAlignment:
        file_path = Path(path)
        self.file_validator.validate(file_path)
        suffix = file_path.suffix.lower()
        if suffix in {'.fasta', '.fa', '.fas', '.aln'}:
            alignment = AlignIO.read(str(file_path), 'fasta')
        elif suffix in {'.nex', '.nexus', '.nexorg', '.txt'}:
            alignment = self._load_nexus(file_path)
        else:
            raise NexusParseError(f'Неподдерживаемый формат файла: {file_path.suffix}')
        self.integrity_validator.validate(alignment)
        return alignment

    def _load_nexus(self, path: Path) -> MultipleSeqAlignment:
        try:
            return AlignIO.read(str(path), 'nexus')
        except Exception:
            return self.matrix_parser.parse(self.text_reader.read(path))
