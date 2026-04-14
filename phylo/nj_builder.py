from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable

from Bio import Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor

from .distance import build_distance_matrix
from .external_inference import run_iqtree_ml


DNA_STATES = ("A", "C", "G", "T")
STATE_TO_INDEX = {state: idx for idx, state in enumerate(DNA_STATES)}
MISSING_CHARS = {"-", "?", "N", "X"}


class TreeBuildError(RuntimeError):
    pass


@dataclass
class TreeBuildResult:
    tree: Tree
    method: str
    log_likelihood: float | None = None
    support_label: str | None = None
    support_kind: str | None = None
    metadata: dict | None = None


@dataclass
class _LikelihoodContext:
    sequences: dict[str, str]
    n_sites: int


@dataclass
class _BayesConfig:
    iterations: int = 160
    burn_in: int = 40
    sample_every: int = 4
    branch_move_prob: float = 0.35
    temperature: float = 1.0


_BRANCH_PRIOR_RATE = 10.0
_RNG = random.Random(42)


def build_neighbor_joining_tree(alignment: MultipleSeqAlignment) -> Tree:
    if len(alignment) < 3:
        raise TreeBuildError("Для построения NJ-дерева нужно минимум 3 таксона.")

    distance_matrix = build_distance_matrix(alignment)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(distance_matrix)
    tree.rooted = False
    _normalize_branch_lengths(tree)
    return tree


def build_tree_by_method(alignment: MultipleSeqAlignment, method: str) -> TreeBuildResult:
    method_key = method.strip().lower()
    if method_key in {"neighbor joining", "nj"}:
        tree = build_neighbor_joining_tree(alignment)
        return TreeBuildResult(tree=tree, method="Neighbor Joining")
    if method_key in {"maximum likelihood", "ml"}:
        return build_maximum_likelihood_tree(alignment)
    raise TreeBuildError(f"Неизвестный метод построения дерева: {method}")


def build_maximum_likelihood_tree(alignment: MultipleSeqAlignment) -> TreeBuildResult:
    external_result = run_iqtree_ml(alignment)
    _normalize_branch_lengths(external_result.tree)
    return TreeBuildResult(
        tree=external_result.tree,
        method="Maximum Likelihood (IQ-TREE)",
        log_likelihood=external_result.log_likelihood,
        support_label=external_result.support_label,
        support_kind=external_result.support_kind,
        metadata=external_result.metadata or {},
    )


def save_tree_newick(tree: Tree, path: str | Path) -> Path:
    path = Path(path)
    Phylo.write(tree, str(path), "newick")
    return path


def load_tree(path: str | Path) -> Tree:
    path = Path(path)
    suffix = path.suffix.lower()
    parse_order = ["newick"]
    if suffix in {".nex", ".nexus", ".txt", ".tree"}:
        parse_order = ["nexus", "newick"]
    elif suffix in {".nwk", ".newick"}:
        parse_order = ["newick", "nexus"]
    last_exc = None
    for fmt in parse_order:
        try:
            tree = Phylo.read(str(path), fmt)
            _normalize_branch_lengths(tree)
            return tree
        except Exception as exc:
            last_exc = exc
    raise TreeBuildError(f"Не удалось прочитать файл дерева '{path.name}': {last_exc}")


# Legacy helpers are kept below because other parts of the prototype may still import them.
def _prepare_context(alignment: MultipleSeqAlignment) -> _LikelihoodContext:
    sequences = {record.id: str(record.seq).upper() for record in alignment}
    lengths = {len(seq) for seq in sequences.values()}
    if len(lengths) != 1:
        raise TreeBuildError("Для ML нужны выровненные последовательности одинаковой длины.")
    return _LikelihoodContext(sequences=sequences, n_sites=next(iter(lengths), 0))


def _tree_log_likelihood(tree: Tree, ctx: _LikelihoodContext) -> float:
    _normalize_branch_lengths(tree)
    if ctx.n_sites == 0:
        return 0.0

    cache: dict[tuple[float, str], tuple[float, float, float, float]] = {}

    def postorder_vectors(clade: Clade) -> list[list[float]]:
        if clade.is_terminal():
            seq = ctx.sequences.get(clade.name or "")
            if seq is None:
                raise TreeBuildError(f"Таксон '{clade.name}' отсутствует в выравнивании.")
            vectors: list[list[float]] = []
            for ch in seq:
                if ch in STATE_TO_INDEX:
                    row = [0.0, 0.0, 0.0, 0.0]
                    row[STATE_TO_INDEX[ch]] = 1.0
                else:
                    row = [1.0, 1.0, 1.0, 1.0]
                vectors.append(row)
            return vectors

        child_vectors = [postorder_vectors(child) for child in clade.clades]
        result = [[1.0, 1.0, 1.0, 1.0] for _ in range(ctx.n_sites)]
        for child, vectors in zip(clade.clades, child_vectors):
            matrix = _jc_transition_tuple(float(max(child.branch_length or 0.0, 1e-6)), cache)
            for site in range(ctx.n_sites):
                target = result[site]
                child_vec = vectors[site]
                updated = [0.0, 0.0, 0.0, 0.0]
                for parent_state in range(4):
                    s = 0.0
                    row_offset = parent_state * 4
                    for child_state in range(4):
                        s += matrix[row_offset + child_state] * child_vec[child_state]
                    updated[parent_state] = target[parent_state] * s
                result[site] = updated
        return result

    root_vectors = postorder_vectors(tree.root)
    log_likelihood = 0.0
    for site_vec in root_vectors:
        site_prob = 0.25 * sum(site_vec)
        log_likelihood += math.log(max(site_prob, 1e-300))
    return log_likelihood


def _jc_transition_tuple(branch_length: float, cache: dict[tuple[float, str], tuple[float, ...]]) -> tuple[float, ...]:
    key = (round(branch_length, 6), "jc69")
    if key in cache:
        return cache[key]
    exp_term = math.exp(-4.0 * branch_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 - 0.25 * exp_term
    values = []
    for i in range(4):
        for j in range(4):
            values.append(p_same if i == j else p_diff)
    cache[key] = tuple(values)
    return cache[key]


def _hill_climb_nni(tree: Tree, ctx: _LikelihoodContext, rounds: int = 3) -> tuple[Tree, float, int]:
    best_tree = _copy_tree(tree)
    best_ll = _tree_log_likelihood(best_tree, ctx)
    improvements = 0

    for _ in range(rounds):
        improved_this_round = False
        for proposal in _generate_nni_neighbors(best_tree):
            proposal_ll = _tree_log_likelihood(proposal, ctx)
            if proposal_ll > best_ll + 1e-9:
                best_tree = proposal
                best_ll = proposal_ll
                improvements += 1
                improved_this_round = True
                break
        if not improved_this_round:
            break
    return best_tree, best_ll, improvements


def _generate_nni_neighbors(tree: Tree) -> Iterable[Tree]:
    parent_map = _build_parent_map(tree)
    for child, parent in list(parent_map.items()):
        if parent is None:
            continue
        if len(parent.clades) != 2 or len(child.clades) != 2:
            continue
        sibling = parent.clades[0] if parent.clades[1] is child else parent.clades[1]
        for child_idx in range(2):
            proposal = _copy_tree(tree)
            proposal_parent_map = _build_parent_map(proposal)
            proposal_child = _locate_matching_clade(child, proposal, proposal_parent_map)
            proposal_parent = proposal_parent_map.get(proposal_child)
            if proposal_parent is None or len(proposal_parent.clades) != 2 or len(proposal_child.clades) != 2:
                continue
            proposal_sibling = proposal_parent.clades[0] if proposal_parent.clades[1] is proposal_child else proposal_parent.clades[1]
            moving_child = proposal_child.clades[child_idx]
            proposal_parent.clades = [proposal_child if c is proposal_sibling else c for c in proposal_parent.clades]
            proposal_child.clades = [proposal_sibling if c is moving_child else c for c in proposal_child.clades]
            proposal_parent.clades = [proposal_child if c is proposal_child else moving_child for c in proposal_parent.clades]
            _normalize_branch_lengths(proposal)
            yield proposal


def _propose_tree(tree: Tree) -> tuple[Tree, str]:
    if _RNG.random() < 0.35:
        neighbors = list(_generate_nni_neighbors(tree))
        if neighbors:
            return _RNG.choice(neighbors), "nni"
    proposal = _copy_tree(tree)
    clades = [c for c in proposal.find_clades() if c is not proposal.root]
    if clades:
        target = _RNG.choice(clades)
        current = float(max(target.branch_length or 0.05, 1e-4))
        factor = math.exp(_RNG.uniform(-0.35, 0.35))
        target.branch_length = max(1e-4, current * factor)
    return proposal, "branch"


def _copy_tree(tree: Tree) -> Tree:
    handle = StringIO()
    Phylo.write(tree, handle, "newick")
    handle.seek(0)
    copied = Phylo.read(handle, "newick")
    copied.rooted = tree.rooted
    return copied


def _build_parent_map(tree: Tree) -> dict[Clade, Clade | None]:
    parent_map: dict[Clade, Clade | None] = {tree.root: None}
    for parent in tree.find_clades(order="level"):
        for child in parent.clades:
            parent_map[child] = parent
    return parent_map


def _locate_matching_clade(target: Clade, tree: Tree, parent_map: dict[Clade, Clade | None]) -> Clade:
    target_signature = _clade_signature(target)
    for clade in parent_map:
        if _clade_signature(clade) == target_signature:
            return clade
    raise TreeBuildError("Не удалось сопоставить внутренний узел при перестройке дерева.")


def _clade_signature(clade: Clade) -> tuple[str, ...]:
    if clade.is_terminal():
        return (clade.name or "",)
    return tuple(sorted(name for child in clade.clades for name in _clade_signature(child)))


def _normalize_branch_lengths(tree: Tree) -> None:
    for clade in tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 0.0
        elif clade.branch_length < 0:
            clade.branch_length = 0.0
