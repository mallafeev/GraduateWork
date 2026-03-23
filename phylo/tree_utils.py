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
    return {
        "taxa_count": len(terminals),
        "internal_nodes": len(nonterminals),
        "total_branch_length": total_branch_length,
        "is_rooted": bool(tree.rooted),
    }


def iter_internal_nodes(tree: Tree) -> Iterable[Clade]:
    for clade in tree.get_nonterminals(order="level"):
        yield clade


def build_node_rows(tree: Tree) -> list[dict]:
    rows: list[dict] = []
    depths = tree.depths()
    for idx, clade in enumerate(iter_internal_nodes(tree), start=1):
        descendants = clade.get_terminals()
        support = getattr(clade, "confidence", None)
        rows.append(
            {
                "node_id": f"N{idx}",
                "name": clade.name or "<internal>",
                "branch_length": float(clade.branch_length or 0.0),
                "depth": float(depths.get(clade, 0.0)),
                "leaf_count": len(descendants),
                "bootstrap": float(support) if support is not None else None,
                "descendants": ", ".join(t.name or "?" for t in descendants[:6]) + (" ..." if len(descendants) > 6 else ""),
            }
        )
    return rows
