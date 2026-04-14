from dataclasses import dataclass, field

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree


@dataclass
class AnalysisArtifacts:
    alignment: MultipleSeqAlignment | None
    tree: Tree | None
    alignment_info: dict = field(default_factory=dict)
    tree_info: dict = field(default_factory=dict)
    node_rows: list[dict] = field(default_factory=list)


@dataclass
class TreeBuildResultView:
    tree: Tree
    method: str
    log_likelihood: float | None = None
    support_label: str | None = None
    support_kind: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelLoadResult:
    metadata: dict = field(default_factory=dict)
    loaded_now: bool = False
