from __future__ import annotations

import re
from pathlib import Path

from .file_kind import CommentStripper, FileKindDetector, TextFileReader


class NexusParseError(ValueError):
    """Raised when an input file does not contain a readable alignment or tree."""


class AlignmentFileValidator:
    def __init__(self, detector: FileKindDetector | None = None, reader: TextFileReader | None = None) -> None:
        self.detector = detector or FileKindDetector()
        self.reader = reader or TextFileReader()

    def validate(self, path: str | Path) -> dict:
        file_path = Path(path)
        kind = self.detector.detect(file_path)
        if kind == 'alignment_fasta':
            return self._validate_fasta(file_path, kind)
        return self._validate_nexus(file_path, kind)

    def _validate_fasta(self, path: Path, kind: str) -> dict:
        text = self.reader.read(path)
        headers = [line for line in text.splitlines() if line.strip().startswith('>')]
        if not headers:
            raise NexusParseError("FASTA-файл должен содержать заголовки, начинающиеся с символа '>'.")
        return {
            'kind': kind,
            'format': 'FASTA',
            'required_blocks': ['>taxon', 'sequence'],
            'present_blocks': ['>taxon'],
            'is_valid': True,
        }

    def _validate_nexus(self, path: Path, kind: str) -> dict:
        text = CommentStripper.strip(self.reader.read(path))
        upper = text.upper()
        present = []
        for block in ['BEGIN TAXA', 'BEGIN DATA', 'BEGIN CHARACTERS', 'MATRIX', 'DIMENSIONS']:
            if block in upper:
                present.append(block)
        has_data_block = ('BEGIN DATA' in upper) or ('BEGIN CHARACTERS' in upper)
        if not has_data_block:
            raise NexusParseError('Для NEXUS-выравнивания нужен блок BEGIN DATA или BEGIN CHARACTERS.')
        if 'MATRIX' not in upper:
            raise NexusParseError('Для NEXUS-выравнивания нужен блок MATRIX с последовательностями.')
        return {
            'kind': kind,
            'format': 'NEXUS alignment',
            'required_blocks': ['BEGIN DATA/CHARACTERS', 'MATRIX'],
            'present_blocks': present,
            'is_valid': True,
        }


class TreeFileValidator:
    def __init__(self, detector: FileKindDetector | None = None, reader: TextFileReader | None = None) -> None:
        self.detector = detector or FileKindDetector()
        self.reader = reader or TextFileReader()

    def validate(self, path: str | Path) -> dict:
        file_path = Path(path)
        kind = self.detector.detect(file_path)
        text = CommentStripper.strip(self.reader.read(file_path))
        upper = text.upper()
        if kind == 'tree_nexus':
            return self._validate_nexus_tree(kind, upper)
        return self._validate_newick(kind, text)

    def _validate_nexus_tree(self, kind: str, upper: str) -> dict:
        present = []
        if 'BEGIN TAXA' in upper:
            present.append('BEGIN TAXA')
        if 'BEGIN TREES' in upper:
            present.append('BEGIN TREES')
        if re.search(r'\bTREE\b', upper):
            present.append('TREE')
        if 'BEGIN TREES' not in upper:
            raise NexusParseError('Для NEXUS-дерева нужен блок BEGIN TREES.')
        if not re.search(r'\bTREE\b', upper):
            raise NexusParseError('В NEXUS-файле дерева должна быть хотя бы одна строка TREE ... = (...).')
        return {
            'kind': kind,
            'format': 'NEXUS tree',
            'required_blocks': ['BEGIN TREES', 'TREE'],
            'present_blocks': present,
            'is_valid': True,
        }

    def _validate_newick(self, kind: str, text: str) -> dict:
        compact = ''.join(line.strip() for line in text.splitlines())
        if '(' not in compact or ')' not in compact or ';' not in compact:
            raise NexusParseError("Newick-файл дерева должен содержать скобочную запись и заканчиваться символом ';'.")
        return {
            'kind': kind,
            'format': 'Newick tree',
            'required_blocks': ['(', ')', ';'],
            'present_blocks': ['(', ')', ';'],
            'is_valid': True,
        }
