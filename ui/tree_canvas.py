from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from Bio.Phylo.BaseTree import Clade, Tree
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QWheelEvent, QMouseEvent


@dataclass
class LayoutNode:
    x: float
    y: float


class PhyloTreeCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(16, 10), tight_layout=False)
        super().__init__(self.figure)
        self.setParent(parent)
        self._tree: Tree | None = None
        self._show_branch_lengths = False
        self._prediction_map: dict[str, float] = {}
        self._show_predictions = True
        self._node_id_map: Dict[Clade, str] = {}
        self._zoom = 1.0
        self._leaf_spacing = 1.0
        self._base_dpi = self.figure.dpi
        self._dragging = False
        self._last_mouse_pos = QPointF()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def set_tree(
        self,
        tree: Tree | None,
        show_branch_lengths: bool = False,
        prediction_map: dict[str, float] | None = None,
        show_predictions: bool = True,
        node_id_map: Dict[Clade, str] | None = None,
    ):
        self._tree = tree
        self._show_branch_lengths = show_branch_lengths
        self._prediction_map = prediction_map or {}
        self._show_predictions = show_predictions
        self._node_id_map = node_id_map or {}
        self.redraw()

    def set_leaf_spacing(self, value: float):
        self._leaf_spacing = max(0.6, min(3.0, float(value)))
        self.redraw()

    def set_zoom(self, value: float):
        self._zoom = max(0.4, min(4.0, float(value)))
        self.redraw()

    def zoom_in(self):
        self.set_zoom(self._zoom * 1.15)

    def zoom_out(self):
        self.set_zoom(self._zoom / 1.15)

    def fit_view(self):
        self._zoom = 1.0
        self.redraw()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            parent = self.parent()
            # parent is QScrollArea viewport widget wrapper; canvas parent is scroll area widget container use parentWidget if needed
            scroll = self.parentWidget().parentWidget() if self.parentWidget() is not None and self.parentWidget().parentWidget() is not None else None
            if scroll is not None and hasattr(scroll, 'horizontalScrollBar'):
                dx = event.pos().x() - self._last_mouse_pos.x()
                dy = event.pos().y() - self._last_mouse_pos.y()
                scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().value() - int(dx))
                scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().value() - int(dy))
                self._last_mouse_pos = event.pos()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def redraw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()

        if self._tree is None:
            ax.text(0.5, 0.5, "Дерево ещё не загружено", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.draw_idle()
            return

        tree = self._tree
        tree.ladderize(reverse=True)
        depths = tree.depths()
        if not max(depths.values(), default=0):
            depths = tree.depths(unit_branch_lengths=True)

        terminals = tree.get_terminals()
        spacing = self._leaf_spacing
        y_positions: dict[Clade, float] = {term: float((len(terminals) - i) * spacing) for i, term in enumerate(terminals)}

        def assign_internal_y(clade: Clade) -> float:
            if clade in y_positions:
                return y_positions[clade]
            child_ys = [assign_internal_y(ch) for ch in clade.clades]
            y_positions[clade] = sum(child_ys) / len(child_ys)
            return y_positions[clade]

        assign_internal_y(tree.root)
        layout = {clade: LayoutNode(float(depths.get(clade, 0.0)), float(y_positions[clade])) for clade in tree.find_clades(order="preorder")}

        max_x = max((n.x for n in layout.values()), default=1.0)
        min_y = min((n.y for n in layout.values()), default=0.0)
        max_y = max((n.y for n in layout.values()), default=1.0)
        taxa_count = max(1, len(terminals))
        max_label_len = max((len(t.name or "?") for t in terminals), default=10)

        fig_height = max(8.0, min(80.0, (taxa_count * 0.36 * spacing + 2.0) * self._zoom))
        fig_width = max(14.0, min(80.0, (9.0 + max_x * 11.0 + max_label_len * 0.14) * self._zoom))
        self.figure.set_size_inches(fig_width, fig_height, forward=True)
        px_w = int(fig_width * self.figure.dpi)
        px_h = int(fig_height * self.figure.dpi)
        self.setMinimumSize(px_w, px_h)
        self.resize(px_w, px_h)

        label_gap = max(0.02, max_x * 0.01)
        x_label_line_end = max_x + label_gap
        x_label_text = x_label_line_end + max(0.003, max_x * 0.003)
        x_pred_offset = max(0.008, max_x * 0.008)

        def pred_color(pred: float | None) -> str:
            if pred is None:
                return "#444"
            if pred < 50:
                return "#c0392b"
            if pred < 75:
                return "#d68910"
            return "#1e8449"

        # draw branches and internal nodes
        for parent in tree.find_clades(order="preorder"):
            p = layout[parent]
            if parent.clades:
                ys = [layout[ch].y for ch in parent.clades]
                ax.plot([p.x, p.x], [min(ys), max(ys)], color="#222", lw=1.4, solid_capstyle="round", zorder=2)

            node_id = self._node_id_map.get(parent)
            pred = self._prediction_map.get(node_id) if node_id else None
            node_color = pred_color(pred) if self._show_predictions else "#222"
            if parent is not tree.root:
                ax.scatter([p.x], [p.y], s=16, color=node_color, zorder=4)
                if self._show_predictions and pred is not None:
                    ax.text(
                        p.x + x_pred_offset,
                        p.y + 0.14 * spacing,
                        f"{pred:.0f}",
                        fontsize=max(7, int(8 * self._zoom)),
                        color=node_color,
                        va="bottom",
                        ha="left",
                        zorder=5,
                    )

            for child in parent.clades:
                c = layout[child]
                child_id = self._node_id_map.get(child)
                child_pred = self._prediction_map.get(child_id) if child_id else None
                color = pred_color(child_pred) if self._show_predictions else "#222"
                ax.plot([p.x, c.x], [c.y, c.y], color=color, lw=1.6, solid_capstyle="round", zorder=2)

                if self._show_branch_lengths:
                    branch_length = child.branch_length if child.branch_length is not None else 0.0
                    if abs(branch_length) > 1e-12:
                        mid_x = (p.x + c.x) / 2
                        ax.text(
                            mid_x,
                            c.y + 0.17 * spacing,
                            f"{max(float(branch_length), 0.0):.3f}",
                            fontsize=max(6, int(7 * self._zoom)),
                            color="#555",
                            ha="center",
                            va="bottom",
                            zorder=5,
                        )

        for term in terminals:
            pos = layout[term]
            # extend terminal branch to near label for readability
            ax.plot([pos.x, x_label_line_end], [pos.y, pos.y], color="#666", lw=0.9, alpha=0.9, zorder=1)
            ax.text(
                x_label_text,
                pos.y,
                term.name or "?",
                fontsize=max(8, int(10 * self._zoom)),
                ha="left",
                va="center",
                color="#111",
                zorder=6,
            )

        ax.set_title("Филогенетическое дерево", fontsize=max(12, int(15 * self._zoom)))
        ax.set_xlim(-max(0.02, max_x * 0.03), x_label_text + max(0.04, max_label_len * 0.018))
        ax.set_ylim(min_y - 1 * spacing, max_y + 1 * spacing)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.margins(x=0, y=0)
        self.figure.subplots_adjust(left=0.02, right=0.985, top=0.95, bottom=0.03)
        self.draw_idle()
