from __future__ import annotations

from copy import deepcopy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Bio import Phylo
from Bio.Phylo.BaseTree import Tree


class PhyloTreeCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(12, 8), tight_layout=False)
        super().__init__(self.figure)
        self.setParent(parent)
        self._tree: Tree | None = None
        self._show_branch_lengths = True

    def set_tree(self, tree: Tree | None, show_branch_lengths: bool = True):
        self._tree = tree
        self._show_branch_lengths = show_branch_lengths
        self.redraw()

    def set_show_branch_lengths(self, value: bool):
        self._show_branch_lengths = value
        self.redraw()

    def redraw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()

        if self._tree is None:
            ax.text(0.5, 0.5, "Дерево ещё не загружено", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.draw_idle()
            return

        tree = deepcopy(self._tree)
        tree.ladderize(reverse=True)
        self._clear_internal_labels(tree)

        taxa_count = max(1, len(tree.get_terminals()))
        fig_height = max(8.0, min(22.0, taxa_count * 0.42))
        self.figure.set_size_inches(14, fig_height, forward=True)

        def branch_labeler(clade):
            if not self._show_branch_lengths:
                return None
            branch_length = getattr(clade, "branch_length", None)
            if branch_length is None:
                return None
            if abs(branch_length) < 1e-12:
                return None
            return f"{branch_length:.3f}"

        Phylo.draw(
            tree,
            axes=ax,
            do_show=False,
            show_confidence=False,
            label_func=lambda clade: clade.name if clade.is_terminal() else None,
            branch_labels=branch_labeler,
        )

        ax.set_title("Neighbor Joining дерево", fontsize=15)
        ax.set_xlabel("Эволюционная дистанция", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=10)

        for text in ax.texts:
            x, _ = text.get_position()
            s = text.get_text()
            if not s:
                continue
            if self._show_branch_lengths and self._looks_like_branch_length(s):
                text.set_fontsize(8)
                text.set_alpha(0.8)
            else:
                text.set_fontsize(10)

        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
        self.draw_idle()

    @staticmethod
    def _clear_internal_labels(tree: Tree) -> None:
        for clade in tree.get_nonterminals():
            clade.name = None

    @staticmethod
    def _looks_like_branch_length(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False
