from __future__ import annotations

import re
from pathlib import Path


class TextFileReader:
    def read(self, path: str | Path) -> str:
        return Path(path).read_text(encoding='utf-8', errors='ignore')


class CommentStripper:
    @staticmethod
    def strip(text: str) -> str:
        return re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)


class FileKindDetector:
    def __init__(self, reader: TextFileReader | None = None) -> None:
        self.reader = reader or TextFileReader()

    def detect(self, path: str | Path) -> str:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix in {'.fasta', '.fa', '.fas', '.aln'}:
            return 'alignment_fasta'
        text = CommentStripper.strip(self.reader.read(file_path)).upper()
        if 'BEGIN TREES' in text:
            return 'tree_nexus'
        if 'BEGIN DATA' in text or 'BEGIN CHARACTERS' in text or 'MATRIX' in text:
            return 'alignment_nexus'
        return 'tree_newick'
