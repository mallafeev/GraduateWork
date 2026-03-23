from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
)

from app_state import AppState
from services.analysis_service import AnalysisArtifacts, AnalysisService
from ui.tree_canvas import PhyloTreeCanvas

APP_TITLE = "Система оценки достоверности филогенетических деревьев"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.service = AnalysisService()
        self.setWindowTitle(APP_TITLE)
        self.resize(1450, 920)
        self._build_ui()
        self._connect_signals()
        self._refresh_state_view()
        self._append_log("Приложение запущено.")

    def _build_ui(self):
        self._build_actions()
        self._build_menu()
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        header = self._build_header()
        outer.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        self.nav_list = QListWidget()
        self.nav_list.setMaximumWidth(260)
        self.nav_list.addItems([
            "1. Подготовка данных",
            "2. Построение дерева",
            "3. Результаты дерева",
            "4. Визуализация",
            "5. Экспорт",
        ])
        self.nav_list.setCurrentRow(0)
        splitter.addWidget(self.nav_list)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setSizes([220, 1180])

        self.tabs.addTab(self._create_input_tab(), "Подготовка данных")
        self.tabs.addTab(self._create_build_tree_tab(), "Построение дерева")
        self.tabs.addTab(self._create_results_tab(), "Результаты дерева")
        self.tabs.addTab(self._create_visual_tab(), "Визуализация")
        self.tabs.addTab(self._create_export_tab(), "Экспорт")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_actions(self):
        self.action_load_tree = QAction("Загрузить дерево", self)
        self.action_load_alignment = QAction("Загрузить NEXUS/FASTA", self)
        self.action_validate = QAction("Проверить данные", self)
        self.action_build_tree = QAction("Построить NJ-дерево", self)
        self.action_export_tree = QAction("Сохранить дерево", self)
        self.action_about = QAction("О программе", self)
        self.action_exit = QAction("Выход", self)

    def _build_menu(self):
        menu_file = self.menuBar().addMenu("Файл")
        menu_file.addAction(self.action_load_alignment)
        menu_file.addAction(self.action_load_tree)
        menu_file.addSeparator()
        menu_file.addAction(self.action_export_tree)
        menu_file.addSeparator()
        menu_file.addAction(self.action_exit)

        menu_analysis = self.menuBar().addMenu("Phylo")
        menu_analysis.addAction(self.action_validate)
        menu_analysis.addAction(self.action_build_tree)

        menu_help = self.menuBar().addMenu("Справка")
        menu_help.addAction(self.action_about)

    def _build_toolbar(self):
        toolbar = QToolBar("Основные действия")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for action in [
            self.action_load_alignment,
            self.action_load_tree,
            self.action_validate,
            self.action_build_tree,
            self.action_export_tree,
        ]:
            toolbar.addAction(action)

    def _build_header(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        title = QLabel(APP_TITLE)
        font = QFont()
        font.setPointSize(15)
        font.setBold(True)
        title.setFont(font)
        subtitle = QLabel(
            "Тест."
        )
        subtitle.setStyleSheet("color: #555;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame

    def _create_input_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        top_grid = QGridLayout()

        box_alignment = QGroupBox("Загрузка выравнивания")
        alignment_layout = QVBoxLayout(box_alignment)
        self.alignment_path_edit = QLineEdit()
        self.alignment_path_edit.setPlaceholderText("Путь к файлу .nex / .nexus / .fasta")
        self.btn_load_alignment = QPushButton("Загрузить NEXUS/FASTA")
        alignment_layout.addWidget(self.alignment_path_edit)
        alignment_layout.addWidget(self.btn_load_alignment)

        box_tree = QGroupBox("Загрузка готового дерева")
        tree_layout = QVBoxLayout(box_tree)
        self.tree_path_edit = QLineEdit()
        self.tree_path_edit.setPlaceholderText("Путь к файлу дерева .nwk")
        self.btn_load_tree = QPushButton("Загрузить дерево .nwk")
        tree_layout.addWidget(self.tree_path_edit)
        tree_layout.addWidget(self.btn_load_tree)

        box_validation = QGroupBox("Сводка входных данных")
        validation_layout = QVBoxLayout(box_validation)
        self.validation_notes = QTextEdit()
        self.validation_notes.setReadOnly(True)
        self.btn_validate = QPushButton("Проверить входные данные")
        validation_layout.addWidget(self.validation_notes)
        validation_layout.addWidget(self.btn_validate)

        top_grid.addWidget(box_alignment, 0, 0)
        top_grid.addWidget(box_tree, 0, 1)
        top_grid.addWidget(box_validation, 1, 0, 1, 2)
        layout.addLayout(top_grid)

        state_box = QGroupBox("Состояние данных")
        form = QFormLayout(state_box)
        self.lbl_tree_loaded = QLabel("-")
        self.lbl_alignment_loaded = QLabel("-")
        self.lbl_validated = QLabel("-")
        self.lbl_source_format = QLabel("-")
        form.addRow("Дерево загружено:", self.lbl_tree_loaded)
        form.addRow("Выравнивание загружено:", self.lbl_alignment_loaded)
        form.addRow("Проверка выполнена:", self.lbl_validated)
        form.addRow("Формат входа:", self.lbl_source_format)
        layout.addWidget(state_box)
        layout.addStretch(1)
        return page

    def _create_build_tree_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        config_box = QGroupBox("Параметры построения дерева")
        form = QFormLayout(config_box)
        self.combo_tree_method = QComboBox()
        self.combo_tree_method.addItems(["Neighbor Joining"])
        self.combo_distance = QComboBox()
        self.combo_distance.addItems(["p-distance (pairwise deletion)"])
        form.addRow("Метод построения:", self.combo_tree_method)
        form.addRow("Метрика расстояния:", self.combo_distance)
        layout.addWidget(config_box)

        buttons = QHBoxLayout()
        self.btn_build_tree = QPushButton("Построить NJ-дерево")
        self.btn_save_tree = QPushButton("Сохранить построенное дерево")
        buttons.addWidget(self.btn_build_tree)
        buttons.addWidget(self.btn_save_tree)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        preview_box = QGroupBox("Сводка дерева")
        preview_layout = QVBoxLayout(preview_box)
        self.tree_build_preview = QTextEdit()
        self.tree_build_preview.setReadOnly(True)
        preview_layout.addWidget(self.tree_build_preview)
        layout.addWidget(preview_box, stretch=1)

        state_box = QGroupBox("Состояние построения")
        form2 = QFormLayout(state_box)
        self.lbl_tree_built = QLabel("-")
        self.lbl_tree_source = QLabel("Не определён")
        form2.addRow("Дерево построено:", self.lbl_tree_built)
        form2.addRow("Источник дерева:", self.lbl_tree_source)
        layout.addWidget(state_box)
        return page

    def _create_results_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels([
            "ID узла",
            "Имя",
            "Длина ветви",
            "Глубина",
            "Листьев",
            "Bootstrap",
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table, stretch=1)

        details_box = QGroupBox("Подробности по узлам")
        details_layout = QVBoxLayout(details_box)
        self.result_details = QTextEdit()
        self.result_details.setReadOnly(True)
        details_layout.addWidget(self.result_details)
        layout.addWidget(details_box)
        return page

    def _create_visual_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        controls = QHBoxLayout()
        self.chk_show_branch_lengths = QCheckBox("Показывать длины ветвей")
        self.chk_show_branch_lengths.setChecked(True)
        controls.addWidget(self.chk_show_branch_lengths)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.tree_canvas = PhyloTreeCanvas()
        layout.addWidget(self.tree_canvas, stretch=1)

        return page

    def _create_export_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.export_log = QPlainTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setPlaceholderText("История сохранения дерева и промежуточных файлов.")
        layout.addWidget(self.export_log)
        return page

    def _connect_signals(self):
        self.action_load_alignment.triggered.connect(self.load_alignment)
        self.action_load_tree.triggered.connect(self.load_tree)
        self.action_validate.triggered.connect(self.validate_input)
        self.action_build_tree.triggered.connect(self.build_tree)
        self.action_export_tree.triggered.connect(self.export_tree)
        self.action_about.triggered.connect(self.show_about)
        self.action_exit.triggered.connect(self.close)

        self.btn_load_alignment.clicked.connect(self.load_alignment)
        self.btn_load_tree.clicked.connect(self.load_tree)
        self.btn_validate.clicked.connect(self.validate_input)
        self.btn_build_tree.clicked.connect(self.build_tree)
        self.btn_save_tree.clicked.connect(self.export_tree)
        self.nav_list.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.results_table.itemSelectionChanged.connect(self.show_selected_node_details)
        self.chk_show_branch_lengths.toggled.connect(self._update_tree_view)

    def load_alignment(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор файла выравнивания",
            "",
            "Alignment files (*.nex *.nexus *.fasta *.fa *.fas *.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            artifacts = self.service.load_alignment(path)
        except Exception as exc:
            self._error(f"Не удалось загрузить выравнивание: {exc}")
            return

        self.state.alignment_file = path
        self.state.alignment_loaded = True
        self.state.alignment = artifacts.alignment
        self.state.source_format = Path(path).suffix.lower()
        self.alignment_path_edit.setText(path)
        self._fill_alignment_summary(artifacts)
        self._append_log(f"Загружено выравнивание: {path}")
        self.status_bar.showMessage("Выравнивание загружено", 4000)
        self._refresh_state_view()

    def load_tree(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор файла дерева",
            "",
            "Tree files (*.nwk *.newick *.tree *.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            artifacts = self.service.load_tree(path)
        except Exception as exc:
            self._error(f"Не удалось загрузить дерево: {exc}")
            return

        self.state.tree_file = path
        self.state.tree_loaded = True
        self.state.tree_built = True
        self.state.tree = artifacts.tree
        self.state.tree_ascii = artifacts.tree_ascii
        self.state.node_rows = artifacts.node_rows
        self.tree_path_edit.setText(path)
        self.lbl_tree_source.setText("Загружено из файла")
        self._apply_tree_artifacts(artifacts)
        self._append_log(f"Загружено дерево: {path}")
        self.status_bar.showMessage("Файл дерева загружен", 4000)
        self._refresh_state_view()

    def validate_input(self):
        lines: list[str] = []
        if self.state.alignment is not None:
            aln = self.state.alignment
            lines.append(f"Выравнивание: {len(aln)} таксонов, {aln.get_alignment_length()} позиций.")
        else:
            lines.append("Выравнивание не загружено.")

        if self.state.tree is not None:
            lines.append(f"Дерево: {len(self.state.tree.get_terminals())} терминальных узлов.")
        else:
            lines.append("Дерево не загружено.")

        if self.state.alignment is not None and self.state.tree is not None:
            aln_taxa = {record.id for record in self.state.alignment}
            tree_taxa = {clade.name for clade in self.state.tree.get_terminals()}
            common = len(aln_taxa & tree_taxa)
            lines.append(f"Совпадающих таксонов между alignment и tree: {common}/{len(aln_taxa)}.")

        self.state.input_validated = self.state.alignment is not None or self.state.tree is not None
        self.validation_notes.setPlainText("\n".join(lines))
        self._append_log("Проверка входных данных выполнена.")
        self._refresh_state_view()

    def build_tree(self):
        if self.state.alignment is None:
            self._error("Для построения дерева сначала загрузите NEXUS/FASTA файл с выравниванием.")
            return
        try:
            artifacts = self.service.build_tree_from_alignment(self.state.alignment)
        except Exception as exc:
            self._error(f"Не удалось построить дерево: {exc}")
            return

        self.state.tree = artifacts.tree
        self.state.tree_built = True
        self.state.tree_loaded = True
        self.state.tree_ascii = artifacts.tree_ascii
        self.state.node_rows = artifacts.node_rows
        self.lbl_tree_source.setText("Построено по выравниванию")
        self._apply_tree_artifacts(artifacts)
        self.tabs.setCurrentIndex(2)
        self.nav_list.setCurrentRow(2)
        self._append_log("Построено NJ-дерево по alignment.")
        self.status_bar.showMessage("NJ-дерево построено", 4000)
        self._refresh_state_view()

    def export_tree(self):
        if self.state.tree is None:
            self._error("Нет дерева для сохранения.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить дерево",
            "tree_built.nwk",
            "Newick (*.nwk *.newick);;All files (*.*)",
        )
        if not path:
            return
        try:
            from phylo.nj_builder import save_tree_newick

            save_tree_newick(self.state.tree, path)
        except Exception as exc:
            self._error(f"Не удалось сохранить дерево: {exc}")
            return
        self.export_log.appendPlainText(f"[{self._now()}] Сохранено дерево: {path}")
        self._append_log(f"Дерево сохранено в {path}")
        self.status_bar.showMessage("Дерево сохранено", 4000)

    def show_selected_node_details(self):
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.state.node_rows):
            return
        info = self.state.node_rows[row]
        bootstrap = "-" if info["bootstrap"] is None else f"{info['bootstrap']:.3f}"
        self.result_details.setPlainText(
            f"Узел: {info['node_id']}\n"
            f"Имя: {info['name']}\n"
            f"Длина ветви: {info['branch_length']:.6f}\n"
            f"Глубина: {info['depth']:.6f}\n"
            f"Количество листьев: {info['leaf_count']}\n"
            f"Bootstrap: {bootstrap}\n"
            f"Листья поддерева: {info['descendants']}"
        )

    def _fill_alignment_summary(self, artifacts: AnalysisArtifacts):
        info = artifacts.alignment_info
        self.validation_notes.setPlainText(
            f"Alignment загружен успешно.\n"
            f"Таксонов: {info.get('taxa_count', 0)}\n"
            f"Длина выравнивания: {info.get('length', 0)}\n"
            f"Доля gap/missing: {info.get('gap_fraction', 0.0):.4f}"
        )

    def _apply_tree_artifacts(self, artifacts: AnalysisArtifacts):
        tree_info = artifacts.tree_info
        self.tree_build_preview.setPlainText(
            f"Метод: Neighbor Joining\n"
            f"Терминальных узлов: {tree_info.get('taxa_count', 0)}\n"
            f"Внутренних узлов: {tree_info.get('internal_nodes', 0)}\n"
            f"Суммарная длина ветвей: {tree_info.get('total_branch_length', 0.0):.6f}\n"
            f"Rooted: {'Да' if tree_info.get('is_rooted') else 'Нет'}"
        )
        self.tree_canvas.set_tree(artifacts.tree, self.chk_show_branch_lengths.isChecked())
        self._populate_node_rows(artifacts.node_rows)

    def _populate_node_rows(self, rows: list[dict]):
        self.results_table.setRowCount(len(rows))
        for row_idx, info in enumerate(rows):
            values = [
                info["node_id"],
                info["name"],
                f"{info['branch_length']:.6f}",
                f"{info['depth']:.6f}",
                str(info["leaf_count"]),
                "-" if info["bootstrap"] is None else f"{info['bootstrap']:.3f}",
            ]
            for col_idx, value in enumerate(values):
                self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        if rows:
            self.results_table.selectRow(0)
            self.show_selected_node_details()

    def _update_tree_view(self):
        self.tree_canvas.set_tree(self.state.tree, self.chk_show_branch_lengths.isChecked())

    def _refresh_state_view(self):
        self.lbl_tree_loaded.setText("Да" if self.state.tree_loaded else "Нет")
        self.lbl_alignment_loaded.setText("Да" if self.state.alignment_loaded else "Нет")
        self.lbl_validated.setText("Да" if self.state.input_validated else "Нет")
        self.lbl_tree_built.setText("Да" if self.state.tree_built else "Нет")
        self.lbl_source_format.setText(self.state.source_format or "-")

    def _append_log(self, text: str):
        timestamped = f"[{self._now()}] {text}"
        if hasattr(self, "export_log"):
            self.export_log.appendPlainText(timestamped)

    def _error(self, text: str):
        self.state.last_error = text
        QMessageBox.critical(self, "Ошибка", text)
        self.status_bar.showMessage(text, 6000)

    def _now(self):
        return datetime.now().strftime("%H:%M:%S")

    def show_about(self):
        QMessageBox.about(
            self,
            "О программе",
            "Рабочий прототип для диплома:\n"
            "- загрузка NEXUS/FASTA;\n"
            "- построение NJ-дерева;\n"
            "- просмотр длины ветвей и внутренних узлов;\n"
            "- экспорт дерева в Newick."
        )


def run():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
