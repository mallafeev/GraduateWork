from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from Bio.Phylo.BaseTree import Clade, Tree
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QToolTip


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
        self._support_map: dict[str, float] = {}
        self._show_support_values = False
        self._support_label = "Bootstrap"
        self._node_tooltip_map: dict[str, str] = {}
        self._node_screen_points: list[tuple[float, float, str]] = []
        self._node_data_points: list[tuple[float, float, str]] = []
        self._zoom = 1.0
        self._leaf_spacing = 1.0
        self._dragging = False
        self._last_mouse_pos = QPointF()
        self._axes = None
        self._base_xlim: tuple[float, float] | None = None
        self._base_ylim: tuple[float, float] | None = None
        self._layout_cache: dict[Clade, LayoutNode] = {}
        self._terminals_cache: list[Clade] = []
        self._layout_key: tuple | None = None
        self._suspend_tooltips = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def set_tree(
        self,
        tree: Tree | None,
        show_branch_lengths: bool = False,
        prediction_map: dict[str, float] | None = None,
        show_predictions: bool = True,
        node_id_map: Dict[Clade, str] | None = None,
        support_map: dict[str, float] | None = None,
        show_support_values: bool = False,
        support_label: str = "Bootstrap",
        node_tooltip_map: dict[str, str] | None = None,
    ):
        old_tree = self._tree
        self._tree = tree
        self._show_branch_lengths = show_branch_lengths
        self._prediction_map = prediction_map or {}
        self._show_predictions = show_predictions
        self._node_id_map = node_id_map or {}
        self._support_map = support_map or {}
        self._show_support_values = show_support_values
        self._support_label = support_label
        self._node_tooltip_map = node_tooltip_map or {}
        if old_tree is not tree:
            self._layout_key = None
            self._layout_cache = {}
            self._terminals_cache = []
            self._zoom = 1.0
        self.redraw()

    def set_leaf_spacing(self, value: float):
        value = max(0.6, min(3.0, float(value)))
        if abs(value - self._leaf_spacing) < 1e-9:
            return
        self._leaf_spacing = value
        self._layout_key = None
        self.redraw()

    def set_zoom(self, value: float):
        self._zoom = max(0.4, min(4.0, float(value)))
        self._apply_zoom()

    def zoom_in(self):
        self.set_zoom(self._zoom * 1.15)

    def zoom_out(self):
        self.set_zoom(self._zoom / 1.15)

    def fit_view(self):
        self._zoom = 1.0
        self._apply_zoom(reset=True)

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
            scroll = self.parentWidget().parentWidget() if self.parentWidget() is not None and self.parentWidget().parentWidget() is not None else None
            if scroll is not None and hasattr(scroll, 'horizontalScrollBar'):
                dx = event.pos().x() - self._last_mouse_pos.x()
                dy = event.pos().y() - self._last_mouse_pos.y()
                scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().value() - int(dx))
                scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().value() - int(dy))
                self._last_mouse_pos = event.pos()
                event.accept()
                return
        else:
            tooltip = self._find_node_tooltip(event.pos().x(), event.pos().y())
            if tooltip:
                QToolTip.showText(event.globalPos(), tooltip, self)
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _find_node_tooltip(self, x: float, y: float) -> str | None:
        if not self._node_screen_points or self._suspend_tooltips:
            return None
        threshold = 12.0
        best_text = None
        best_dist = None
        for px, py, node_id in self._node_screen_points:
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 <= threshold ** 2 and (best_dist is None or d2 < best_dist):
                best_dist = d2
                best_text = self._node_tooltip_map.get(node_id)
        return best_text

    def _layout_signature(self) -> tuple:
        tree_id = id(self._tree)
        return (tree_id, round(self._leaf_spacing, 4))

    def _ensure_layout_cache(self):
        if self._tree is None:
            self._layout_cache = {}
            self._terminals_cache = []
            return
        key = self._layout_signature()
        if key == self._layout_key and self._layout_cache:
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
        self._layout_cache = {
            clade: LayoutNode(float(depths.get(clade, 0.0)), float(y_positions[clade]))
            for clade in tree.find_clades(order="preorder")
        }
        self._terminals_cache = terminals
        self._layout_key = key

    def _apply_zoom(self, reset: bool = False):
        if self._axes is None or self._base_xlim is None or self._base_ylim is None:
            return
        ax = self._axes
        if reset:
            ax.set_xlim(*self._base_xlim)
            ax.set_ylim(*self._base_ylim)
            self.draw_idle()
            return

        base_xlim = self._base_xlim
        base_ylim = self._base_ylim
        full_w = base_xlim[1] - base_xlim[0]
        full_h = base_ylim[1] - base_ylim[0]
        zoom = max(0.4, min(4.0, self._zoom))
        new_w = full_w / zoom
        new_h = full_h / zoom
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        center_x = (cur_xlim[0] + cur_xlim[1]) / 2 if cur_xlim else (base_xlim[0] + base_xlim[1]) / 2
        center_y = (cur_ylim[0] + cur_ylim[1]) / 2 if cur_ylim else (base_ylim[0] + base_ylim[1]) / 2
        x0 = max(base_xlim[0], center_x - new_w / 2)
        x1 = min(base_xlim[1], center_x + new_w / 2)
        y0 = max(base_ylim[0], center_y - new_h / 2)
        y1 = min(base_ylim[1], center_y + new_h / 2)
        if (x1 - x0) < new_w:
            if x0 <= base_xlim[0]:
                x1 = min(base_xlim[1], x0 + new_w)
            else:
                x0 = max(base_xlim[0], x1 - new_w)
        if (y1 - y0) < new_h:
            if y0 <= base_ylim[0]:
                y1 = min(base_ylim[1], y0 + new_h)
            else:
                y0 = max(base_ylim[0], y1 - new_h)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        self.draw_idle()

    def redraw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._axes = ax
        self._node_screen_points = []
        self._node_data_points = []

        if self._tree is None:
            ax.text(0.5, 0.5, "Дерево ещё не загружено", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self._base_xlim = None
            self._base_ylim = None
            self.draw_idle()
            return

        self._ensure_layout_cache()
        tree = self._tree
        layout = self._layout_cache
        terminals = self._terminals_cache
        spacing = self._leaf_spacing

        max_x = max((n.x for n in layout.values()), default=1.0)
        min_y = min((n.y for n in layout.values()), default=0.0)
        max_y = max((n.y for n in layout.values()), default=1.0)
        taxa_count = max(1, len(terminals))
        max_label_len = max((len(t.name or "?") for t in terminals), default=10)

        fig_height = max(8.0, min(70.0, taxa_count * 0.30 * spacing + 2.0))
        fig_width = max(14.0, min(70.0, 9.0 + max_x * 10.0 + max_label_len * 0.14))
        self.figure.set_size_inches(fig_width, fig_height, forward=True)
        px_w = int(fig_width * self.figure.dpi)
        px_h = int(fig_height * self.figure.dpi)
        self.setMinimumSize(px_w, px_h)
        self.resize(px_w, px_h)

        label_gap = max(0.02, max_x * 0.01)
        x_label_line_end = max_x + label_gap
        x_label_text = x_label_line_end + max(0.003, max_x * 0.003)
        x_pred_offset = max(0.008, max_x * 0.008)

        show_node_text = len(terminals) <= 250
        show_leaf_text = len(terminals) <= 600
        self._suspend_tooltips = False

        def pred_color(pred: float | None) -> str:
            if pred is None:
                return "#444"
            if pred < 50:
                return "#c0392b"
            if pred < 75:
                return "#d68910"
            return "#1e8449"

        for parent in tree.find_clades(order="preorder"):
            p = layout[parent]
            if parent.clades:
                ys = [layout[ch].y for ch in parent.clades]
                ax.plot([p.x, p.x], [min(ys), max(ys)], color="#222", lw=1.4, solid_capstyle="round", zorder=2)

            node_id = self._node_id_map.get(parent)
            pred = self._prediction_map.get(node_id) if node_id else None
            node_color = pred_color(pred) if self._show_predictions else "#222"
            if parent is not tree.root:
                ax.scatter([p.x], [p.y], s=18, color=node_color, zorder=4)
                if node_id:
                    self._node_data_points.append((float(p.x), float(p.y), node_id))
                if show_node_text and self._show_predictions and pred is not None:
                    ax.text(p.x + x_pred_offset, p.y + 0.14 * spacing, f"RF {pred:.0f}", fontsize=8, color=node_color, va="bottom", ha="left", zorder=5)
                if show_node_text and self._show_support_values and node_id is not None:
                    support_val = self._support_map.get(node_id)
                    if support_val is not None:
                        ax.text(p.x + x_pred_offset, p.y - 0.16 * spacing, f"PP {support_val:.0f}", fontsize=8, color="#2c3e50", va="top", ha="left", zorder=5)

            for child in parent.clades:
                c = layout[child]
                child_id = self._node_id_map.get(child)
                child_pred = self._prediction_map.get(child_id) if child_id else None
                color = pred_color(child_pred) if self._show_predictions else "#222"
                ax.plot([p.x, c.x], [c.y, c.y], color=color, lw=1.6, solid_capstyle="round", zorder=2)
                if self._show_branch_lengths and show_node_text:
                    branch_length = child.branch_length if child.branch_length is not None else 0.0
                    if abs(branch_length) > 1e-12:
                        mid_x = (p.x + c.x) / 2
                        ax.text(mid_x, c.y + 0.17 * spacing, f"{max(float(branch_length), 0.0):.3f}", fontsize=7, color="#555", ha="center", va="bottom", zorder=5)

        if show_leaf_text:
            for term in terminals:
                pos = layout[term]
                ax.plot([pos.x, x_label_line_end], [pos.y, pos.y], color="#666", lw=0.9, alpha=0.9, zorder=1)
                ax.text(x_label_text, pos.y, term.name or "?", fontsize=9, ha="left", va="center", color="#111", zorder=6)
        else:
            for term in terminals:
                pos = layout[term]
                ax.plot([pos.x, x_label_line_end], [pos.y, pos.y], color="#666", lw=0.9, alpha=0.9, zorder=1)

        title = "Филогенетическое дерево"
        if self._show_support_values:
            title += f" — {self._support_label}"
        if not show_leaf_text:
            title += " (упрощённый режим)"
        ax.set_title(title, fontsize=13)
        self._base_xlim = (-max(0.02, max_x * 0.03), x_label_text + max(0.04, max_label_len * 0.018))
        self._base_ylim = (min_y - 1 * spacing, max_y + 1 * spacing)
        ax.set_xlim(*self._base_xlim)
        ax.set_ylim(*self._base_ylim)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.margins(x=0, y=0)
        self.figure.subplots_adjust(left=0.02, right=0.985, top=0.95, bottom=0.03)
        self._apply_zoom(reset=self._zoom == 1.0)
        self.draw_idle()

    def draw_idle(self, *args, **kwargs):
        super().draw_idle(*args, **kwargs)
        self._refresh_node_screen_points()

    def _refresh_node_screen_points(self):
        if self._axes is None or not self._node_data_points:
            return
        transformed = []
        for x, y, node_id in self._node_data_points:
            disp = self._axes.transData.transform((x, y))
            transformed.append((float(disp[0]), float(self.height() - disp[1]), node_id))
        self._node_screen_points = transformed
