from __future__ import annotations

from io import StringIO
from typing import Iterable

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree


def tree_to_ascii(tree: Tree) -> str:
    buf = StringIO()
    Phylo.draw_ascii(tree, file=buf)
    return buf.getvalue()


def summarize_tree(tree: Tree) -> dict:
    terminals = tree.get_terminals()
    nonterminals = tree.get_nonterminals()
    total_branch_length = sum((clade.branch_length or 0.0) for clade in tree.find_clades())
    support_values = [float(clade.confidence) for clade in nonterminals if getattr(clade, 'confidence', None) is not None]
    return {
        "taxa_count": len(terminals),
        "internal_nodes": len(nonterminals),
        "total_branch_length": total_branch_length,
        "is_rooted": bool(tree.rooted),
        "has_support": bool(support_values),
        "support_min": min(support_values) if support_values else None,
        "support_max": max(support_values) if support_values else None,
    }


def iter_internal_nodes(tree: Tree) -> Iterable[Clade]:
    for clade in tree.get_nonterminals(order="level"):
        yield clade


def build_node_rows(tree: Tree, support_source: str | None = None) -> list[dict]:
    rows: list[dict] = []
    depths = tree.depths()
    if not max(depths.values(), default=0):
        depths = tree.depths(unit_branch_lengths=True)
    for idx, clade in enumerate(iter_internal_nodes(tree), start=1):
        descendants = clade.get_terminals()
        support = getattr(clade, "confidence", None)
        inferred_source = support_source or ("observed" if support is not None else "none")
        rows.append(
            {
                "node_id": f"N{idx}",
                "name": clade.name or "<internal>",
                "branch_length": float(max(clade.branch_length or 0.0, 0.0)),
                "depth": float(depths.get(clade, 0.0)),
                "leaf_count": len(descendants),
                "bootstrap": float(support) if support is not None else None,
                "predicted_bootstrap": None,
                "support_source": inferred_source if support is not None or support_source else "none",
                "descendants": ", ".join(t.name or "?" for t in descendants[:6]) + (" ..." if len(descendants) > 6 else ""),
                "feature_influences": [],
                "feature_influence_text": "",
            }
        )
    return rows


def build_clade_node_id_map(tree: Tree) -> dict[Clade, str]:
    mapping: dict[Clade, str] = {}
    for idx, clade in enumerate(iter_internal_nodes(tree), start=1):
        mapping[clade] = f"N{idx}"
    return mapping
