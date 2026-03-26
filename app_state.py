from dataclasses import dataclass, field
from typing import Optional

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree


@dataclass
class AppState:
    tree_file: str = ""
    alignment_file: str = ""
    model_file: str = ""
    tree_loaded: bool = False
    alignment_loaded: bool = False
    model_loaded: bool = False
    tree_built: bool = False
    analysis_completed: bool = False
    input_validated: bool = False

    alignment: Optional[MultipleSeqAlignment] = None
    tree: Optional[Tree] = None
    source_format: str = ""
    last_error: str = ""
    model_metadata: dict = field(default_factory=dict)
    node_rows: list[dict] = field(default_factory=list)
