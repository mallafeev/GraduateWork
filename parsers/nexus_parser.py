from __future__ import annotations

from pathlib import Path

from Bio.Align import MultipleSeqAlignment

from .alignment_loader import AlignmentLoader
from .alignment_summary import AlignmentExporter, AlignmentSummaryBuilder
from .file_kind import CommentStripper, FileKindDetector, TextFileReader
from .validation import AlignmentFileValidator, NexusParseError, TreeFileValidator

_reader = TextFileReader()
_detector = FileKindDetector(_reader)
_alignment_validator = AlignmentFileValidator(_detector, _reader)
_tree_validator = TreeFileValidator(_detector, _reader)
_alignment_loader = AlignmentLoader(_alignment_validator, _reader)
_summary_builder = AlignmentSummaryBuilder()
_exporter = AlignmentExporter()


def _read_text(path: str | Path) -> str:
    return _reader.read(path)


def _strip_comments(text: str) -> str:
    return CommentStripper.strip(text)


def detect_file_kind(path: str | Path) -> str:
    return _detector.detect(path)


def validate_alignment_file(path: str | Path) -> dict:
    return _alignment_validator.validate(path)


def validate_tree_file(path: str | Path) -> dict:
    return _tree_validator.validate(path)


def load_alignment(path: str | Path) -> MultipleSeqAlignment:
    return _alignment_loader.load(path)


def load_nexus_alignment(path: str | Path) -> MultipleSeqAlignment:
    return _alignment_loader.load(path)


def alignment_summary(alignment: MultipleSeqAlignment) -> dict:
    return _summary_builder.build(alignment)


def export_fasta(alignment: MultipleSeqAlignment, output_path: str | Path) -> Path:
    return _exporter.export_fasta(alignment, output_path)


__all__ = [
    'NexusParseError',
    '_read_text',
    '_strip_comments',
    'detect_file_kind',
    'validate_alignment_file',
    'validate_tree_file',
    'load_alignment',
    'load_nexus_alignment',
    'alignment_summary',
    'export_fasta',
]
