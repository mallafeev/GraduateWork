from __future__ import annotations

from pathlib import Path

from Bio import Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor

from .distance import build_distance_matrix


class TreeBuildError(RuntimeError):
    pass


def build_neighbor_joining_tree(alignment: MultipleSeqAlignment) -> Tree:
    if len(alignment) < 3:
        raise TreeBuildError("Для построения NJ-дерева нужно минимум 3 таксона.")

    distance_matrix = build_distance_matrix(alignment)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(distance_matrix)
    tree.rooted = False
    _normalize_branch_lengths(tree)
    return tree


def save_tree_newick(tree: Tree, path: str | Path) -> Path:
    path = Path(path)
    Phylo.write(tree, str(path), "newick")
    return path


def load_tree(path: str | Path) -> Tree:
    path = Path(path)
    return Phylo.read(str(path), "newick")


def _normalize_branch_lengths(tree: Tree) -> None:
    for clade in tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 0.0
