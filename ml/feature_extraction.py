from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree

FEATURE_COLUMNS = [
    "branch_length",
    "depth",
    "n_leaves_subtree",
    "subtree_fraction",
    "subtree_balance",
    "mean_child_branch_length",
    "std_child_branch_length",
    "taxa_count",
    "alignment_length",
    "gap_fraction_global",
    "gc_mean_global",
    "gc_std_global",
    "variable_site_fraction_global",
    "gap_fraction_clade",
    "gc_mean_clade",
    "gc_std_clade",
    "mean_pairwise_pdist_clade",
]

VALID_DNA = set("ACGT")
DNA_PLUS_GAP = set("ACGT-?")


def sanitize_seq(seq: str) -> str:
    return "".join(ch.upper() for ch in seq if ch.upper() in DNA_PLUS_GAP)


def alignment_to_dict(alignment: MultipleSeqAlignment) -> Dict[str, str]:
    return {record.id: str(record.seq) for record in alignment}


def p_distance(seq1: str, seq2: str) -> float:
    mismatches = 0
    valid = 0
    for a, b in zip(seq1, seq2):
        if a in {"-", "?"} or b in {"-", "?"}:
            continue
        if a not in VALID_DNA or b not in VALID_DNA:
            continue
        valid += 1
        if a != b:
            mismatches += 1
    return np.nan if valid == 0 else mismatches / valid


def seq_gap_fraction(seq: str) -> float:
    return np.nan if not seq else sum(ch in {"-", "?"} for ch in seq) / len(seq)


def seq_gc_fraction(seq: str) -> float:
    letters = [ch for ch in seq if ch in VALID_DNA]
    return np.nan if not letters else sum(ch in {"G", "C"} for ch in letters) / len(letters)


def alignment_stats(seqs: Dict[str, str]) -> Dict[str, float]:
    items = [sanitize_seq(s) for s in seqs.values()]
    if not items:
        return {
            "taxa_count": np.nan,
            "alignment_length": np.nan,
            "gap_fraction_global": np.nan,
            "gc_mean_global": np.nan,
            "gc_std_global": np.nan,
            "variable_site_fraction_global": np.nan,
        }

    n = len(items)
    L = max(len(s) for s in items)
    items = [s.ljust(L, "-") for s in items]

    gap_fraction_global = np.mean([seq_gap_fraction(s) for s in items])
    gc_values = [seq_gc_fraction(s) for s in items]
    gc_mean_global = float(np.nanmean(gc_values))
    gc_std_global = float(np.nanstd(gc_values))

    variable_sites = 0
    informative_positions = 0
    for i in range(L):
        col = [s[i] for s in items if s[i] in VALID_DNA]
        if len(col) < 2:
            continue
        informative_positions += 1
        if len(set(col)) > 1:
            variable_sites += 1

    variable_site_fraction = variable_sites / informative_positions if informative_positions else np.nan

    return {
        "taxa_count": float(n),
        "alignment_length": float(L),
        "gap_fraction_global": float(gap_fraction_global),
        "gc_mean_global": gc_mean_global,
        "gc_std_global": gc_std_global,
        "variable_site_fraction_global": float(variable_site_fraction),
    }


def get_leaf_names(clade) -> List[str]:
    return [leaf.name for leaf in clade.get_terminals() if leaf.name]


def mean_pairwise_distance(names: List[str], seqs: Dict[str, str]) -> float:
    if len(names) < 2:
        return 0.0
    vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1 = sanitize_seq(seqs.get(names[i], ""))
            s2 = sanitize_seq(seqs.get(names[j], ""))
            if not s1 or not s2:
                continue
            d = p_distance(s1, s2)
            if not np.isnan(d):
                vals.append(d)
    return np.nan if not vals else float(np.mean(vals))


def clade_alignment_stats(names: List[str], seqs: Dict[str, str]) -> Dict[str, float]:
    sub = {k: seqs[k] for k in names if k in seqs}
    if not sub:
        return {
            "gap_fraction_clade": np.nan,
            "gc_mean_clade": np.nan,
            "gc_std_clade": np.nan,
            "mean_pairwise_pdist_clade": np.nan,
        }

    gap_fraction_clade = float(np.nanmean([seq_gap_fraction(sanitize_seq(v)) for v in sub.values()]))
    gc_vals = [seq_gc_fraction(sanitize_seq(v)) for v in sub.values()]
    gc_mean_clade = float(np.nanmean(gc_vals))
    gc_std_clade = float(np.nanstd(gc_vals))
    mpd = mean_pairwise_distance(list(sub.keys()), sub)
    return {
        "gap_fraction_clade": gap_fraction_clade,
        "gc_mean_clade": gc_mean_clade,
        "gc_std_clade": gc_std_clade,
        "mean_pairwise_pdist_clade": mpd,
    }


def subtree_balance(clade) -> float:
    children = clade.clades
    if len(children) < 2:
        return 0.0
    sizes = [len(ch.get_terminals()) for ch in children]
    total = sum(sizes)
    return 0.0 if total == 0 else abs(sizes[0] - sizes[1]) / total


@dataclass
class NodeFeatureRow:
    node_id: str
    features: Dict[str, float]


def extract_feature_rows(tree: Tree, alignment: MultipleSeqAlignment) -> List[NodeFeatureRow]:
    seqs = alignment_to_dict(alignment)
    global_stats = alignment_stats(seqs)
    depths = tree.depths()
    rows: List[NodeFeatureRow] = []
    node_idx = 0

    for clade in tree.get_nonterminals(order="level"):
        branch_length = max(float(clade.branch_length or 0.0), 0.0)
        leaf_names = get_leaf_names(clade)
        n_leaves = len(leaf_names)
        clade_stats = clade_alignment_stats(leaf_names, seqs)
        child_branch_lengths = [max(float(ch.branch_length or 0.0), 0.0) for ch in clade.clades]

        feats = {
            "branch_length": branch_length,
            "depth": float(depths.get(clade, 0.0)),
            "n_leaves_subtree": float(n_leaves),
            "subtree_fraction": float(n_leaves / global_stats["taxa_count"]) if global_stats["taxa_count"] else np.nan,
            "subtree_balance": float(subtree_balance(clade)),
            "mean_child_branch_length": float(np.mean(child_branch_lengths)) if child_branch_lengths else 0.0,
            "std_child_branch_length": float(np.std(child_branch_lengths)) if child_branch_lengths else 0.0,
            **global_stats,
            **clade_stats,
        }
        rows.append(NodeFeatureRow(node_id=f"N{node_idx+1}", features=feats))
        node_idx += 1
    return rows
