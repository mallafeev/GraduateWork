from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
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
    QScrollArea,
    QProgressDialog,
    QSlider,
)
from app_state import AppState
from services.analysis_service import AnalysisArtifacts, AnalysisService
from phylo.tree_utils import build_clade_node_id_map
from ui.tree_canvas import PhyloTreeCanvas
APP_TITLE = "Система оценки достоверности филогенетических деревьев"


class TreeBuildWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, alignment, method: str):
        super().__init__()
        self.alignment = alignment
        self.method = method

    def run(self):
        try:
            service = AnalysisService()
            artifacts = service.build_tree_from_alignment(self.alignment, method=self.method)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(artifacts)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.service = AnalysisService()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 940)
        self.preloaded_models = self._discover_preloaded_models()
        self._build_tree_thread = None
        self._build_tree_worker = None
        self._build_tree_busy = None
        self._build_ui()
        self._connect_signals()
        self._refresh_state_view()
        self._append_log("Приложение запущено.")
    def _discover_preloaded_models(self):
        base = Path(__file__).resolve().parents[1]
        models = []
        seen = set()
        preferred = [
            ("RF model v1", base / "models" / "model_v1.pkl"),
            ("RF model v1 (ml_outputs_v1)", base / "ml_outputs_v1" / "model_v1.pkl"),
            ("RF model v1 nested", base / "ml_outputs_v1" / "ml_outputs_v1" / "model_v1.pkl"),
            ("RF model v2 (IQ-TREE)", base / "ml2" / "ml_outputs_v1" / "model_v2.pkl"),
        ]
        for label, path in preferred:
            path = path.resolve()
            if not path.exists():
                continue
            resolved = str(path)
            if resolved in seen:
                continue
            meta = path.with_name("model_metadata.json")
            models.append({
                "label": label,
                "path": resolved,
                "metadata": str(meta) if meta.exists() else None,
            })
            seen.add(resolved)
        return models
    def _build_ui(self):
        self._build_actions()
        self._build_menu()
        self._build_toolbar()
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addWidget(self._build_header())
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)
        self.nav_list = QListWidget()
        self.nav_list.setMaximumWidth(260)
        self.nav_list.addItems([
            "1. Подготовка данных",
            "2. Построение дерева",
            "3. Результаты анализа",
            "4. Визуализация",
            "5. Экспорт",
        ])
        self.nav_list.setCurrentRow(0)
        splitter.addWidget(self.nav_list)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setSizes([220, 1200])
        self.tabs.addTab(self._create_input_tab(), "Подготовка данных")
        self.tabs.addTab(self._create_build_tree_tab(), "Построение дерева")
        self.tabs.addTab(self._create_results_tab(), "Результаты анализа")
        self.tabs.addTab(self._create_visual_tab(), "Визуализация")
        self.tabs.addTab(self._create_export_tab(), "Экспорт")
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    def _build_actions(self):
        self.action_load_tree = QAction("Загрузить/визуализировать дерево", self)
        self.action_load_alignment = QAction("Загрузить NEXUS/FASTA", self)
        self.action_load_model = QAction("Загрузить модель", self)
        self.action_validate = QAction("Проверить данные", self)
        self.action_build_tree = QAction("Построить дерево", self)
        self.action_visualize_tree = QAction("Визуализировать дерево из файла", self)
        self.action_predict = QAction("Предсказать bootstrap", self)
        self.action_export_tree = QAction("Сохранить дерево", self)
        self.action_about = QAction("О программе", self)
        self.action_exit = QAction("Выход", self)
    def _build_menu(self):
        menu_file = self.menuBar().addMenu("Файл")
        menu_file.addAction(self.action_load_alignment)
        menu_file.addAction(self.action_load_tree)
        menu_file.addAction(self.action_load_model)
        menu_file.addSeparator()
        menu_file.addAction(self.action_export_tree)
        menu_file.addSeparator()
        menu_file.addAction(self.action_exit)
        menu_analysis = self.menuBar().addMenu("Phylo")
        menu_analysis.addAction(self.action_validate)
        menu_analysis.addAction(self.action_build_tree)
        menu_analysis.addAction(self.action_predict)
        menu_analysis.addAction(self.action_visualize_tree)
        menu_help = self.menuBar().addMenu("Справка")
        menu_help.addAction(self.action_about)
    def _build_toolbar(self):
        toolbar = QToolBar("Основные действия")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for action in [
            self.action_load_alignment,
            self.action_load_tree,
            self.action_load_model,
            self.action_validate,
            self.action_build_tree,
            self.action_predict,
            self.action_visualize_tree,
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
        subtitle = QLabel("Загрузка NEXUS/FASTA, построение деревьев NJ/ML и ML-предсказание поддержки для внутренних узлов.")
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
        self.tree_path_edit.setPlaceholderText("Путь к файлу дерева .nex / .nexus / .nwk / .newick")
        self.btn_load_tree = QPushButton("Загрузить и визуализировать дерево")
        tree_layout.addWidget(self.tree_path_edit)
        tree_layout.addWidget(self.btn_load_tree)
        box_model = QGroupBox("Модель Random Forest")
        model_layout = QVBoxLayout(box_model)
        self.combo_model_select = QComboBox()
        self.combo_model_select.addItem("Выберите модель...", None)
        for item in self.preloaded_models:
            self.combo_model_select.addItem(item["label"], item)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("Путь к выбранной модели")
        self.btn_load_model = QPushButton("Загрузить выбранную модель")
        model_layout.addWidget(self.combo_model_select)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.btn_load_model)
        box_validation = QGroupBox("Сводка входных данных")
        validation_layout = QVBoxLayout(box_validation)
        self.validation_notes = QTextEdit()
        self.validation_notes.setReadOnly(True)
        self.btn_validate = QPushButton("Проверить входные данные")
        validation_layout.addWidget(self.validation_notes)
        validation_layout.addWidget(self.btn_validate)
        top_grid.addWidget(box_alignment, 0, 0)
        top_grid.addWidget(box_tree, 0, 1)
        top_grid.addWidget(box_model, 0, 2)
        top_grid.addWidget(box_validation, 1, 0, 1, 3)
        layout.addLayout(top_grid)
        state_box = QGroupBox("Состояние данных")
        form = QFormLayout(state_box)
        self.lbl_tree_loaded = QLabel("-")
        self.lbl_alignment_loaded = QLabel("-")
        self.lbl_model_loaded = QLabel("-")
        self.lbl_validated = QLabel("-")
        self.lbl_source_format = QLabel("-")
        form.addRow("Дерево загружено:", self.lbl_tree_loaded)
        form.addRow("Выравнивание загружено:", self.lbl_alignment_loaded)
        form.addRow("Модель загружена:", self.lbl_model_loaded)
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
        self.combo_tree_method.addItems(["Neighbor Joining", "Maximum Likelihood"])
        self.combo_distance = QComboBox()
        self.combo_distance.addItems(["p-distance (pairwise deletion)"])
        form.addRow("Метод построения:", self.combo_tree_method)
        form.addRow("Метрика расстояния:", self.combo_distance)
        layout.addWidget(config_box)
        buttons = QHBoxLayout()
        self.btn_build_tree = QPushButton("Построить дерево")
        self.btn_visualize_tree = QPushButton("Визуализировать дерево из файла")
        self.btn_predict = QPushButton("Предсказать bootstrap")
        self.btn_save_tree = QPushButton("Сохранить дерево")
        buttons.addWidget(self.btn_build_tree)
        buttons.addWidget(self.btn_visualize_tree)
        buttons.addWidget(self.btn_predict)
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
        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels([
            "ID узла",
            "Длина ветви",
            "Глубина",
            "Листьев",
            "Поддержка узла",
            "Predicted RF",
            "Источник",
            "Листья поддерева",
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
        self.chk_show_branch_lengths.setChecked(False)
        self.chk_show_predictions = QCheckBox("Подсветка по predicted bootstrap")
        self.chk_show_predictions.setChecked(True)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_out = QPushButton("-")
        self.btn_fit_tree = QPushButton("Fit")
        self.leaf_spacing_slider = QSlider(Qt.Horizontal)
        self.leaf_spacing_slider.setMinimum(6)
        self.leaf_spacing_slider.setMaximum(24)
        self.leaf_spacing_slider.setValue(10)
        self.leaf_spacing_slider.setTracking(False)
        controls.addWidget(self.chk_show_branch_lengths)
        controls.addWidget(self.chk_show_predictions)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Масштаб:"))
        controls.addWidget(self.btn_zoom_out)
        controls.addWidget(self.btn_zoom_in)
        controls.addWidget(self.btn_fit_tree)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Интервал листьев:"))
        controls.addWidget(self.leaf_spacing_slider, stretch=1)
        layout.addLayout(controls)
        self.tree_canvas = PhyloTreeCanvas()
        self.tree_scroll = QScrollArea()
        self.tree_scroll.setWidget(self.tree_canvas)
        self.tree_scroll.setWidgetResizable(False)
        self.tree_scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.tree_scroll, stretch=1)
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
        self.action_load_model.triggered.connect(self.load_model)
        self.action_validate.triggered.connect(self.validate_input)
        self.action_build_tree.triggered.connect(self.build_tree)
        self.action_visualize_tree.triggered.connect(self.load_tree)
        self.action_predict.triggered.connect(self.predict_bootstrap)
        self.action_export_tree.triggered.connect(self.export_tree)
        self.action_about.triggered.connect(self.show_about)
        self.action_exit.triggered.connect(self.close)
        self.btn_load_alignment.clicked.connect(self.load_alignment)
        self.btn_load_tree.clicked.connect(self.load_tree)
        self.btn_load_model.clicked.connect(self.load_model)
        self.combo_model_select.currentIndexChanged.connect(self._sync_selected_model_path)
        self.btn_validate.clicked.connect(self.validate_input)
        self.btn_build_tree.clicked.connect(self.build_tree)
        self.btn_visualize_tree.clicked.connect(self.load_tree)
        self.btn_predict.clicked.connect(self.predict_bootstrap)
        self.btn_save_tree.clicked.connect(self.export_tree)
        self.nav_list.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.results_table.itemSelectionChanged.connect(self.show_selected_node_details)
        self.chk_show_branch_lengths.toggled.connect(self._update_tree_view)
        self.chk_show_predictions.toggled.connect(self._update_tree_view)
        self.btn_zoom_in.clicked.connect(self._zoom_in_tree)
        self.btn_zoom_out.clicked.connect(self._zoom_out_tree)
        self.btn_fit_tree.clicked.connect(self._fit_tree)
        self.leaf_spacing_slider.sliderReleased.connect(self._apply_leaf_spacing)
    def _show_busy(self, title: str, text: str) -> QProgressDialog:
        dlg = QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        return dlg
    def _hide_busy(self, dlg: QProgressDialog | None):
        QApplication.restoreOverrideCursor()
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()
        QApplication.processEvents()
    def _set_build_controls_enabled(self, enabled: bool):
        self.btn_build_tree.setEnabled(enabled)
        self.action_build_tree.setEnabled(enabled)
        self.combo_tree_method.setEnabled(enabled)
        self.combo_distance.setEnabled(enabled)

    def _start_background_tree_build(self, method: str):
        self._build_tree_busy = self._show_busy("Построение дерева", "ML-дерево строится в фоне. Окно остаётся отзывчивым, пожалуйста подождите...")
        self.status_bar.showMessage("Строится ML-дерево...", 0)
        self._set_build_controls_enabled(False)
        self._build_tree_thread = QThread(self)
        self._build_tree_worker = TreeBuildWorker(self.state.alignment, method)
        self._build_tree_worker.moveToThread(self._build_tree_thread)
        self._build_tree_thread.started.connect(self._build_tree_worker.run)
        self._build_tree_worker.finished.connect(self._on_background_tree_built)
        self._build_tree_worker.failed.connect(self._on_background_tree_build_failed)
        self._build_tree_worker.finished.connect(self._build_tree_thread.quit)
        self._build_tree_worker.failed.connect(self._build_tree_thread.quit)
        self._build_tree_thread.finished.connect(self._build_tree_worker.deleteLater)
        self._build_tree_thread.finished.connect(self._build_tree_thread.deleteLater)
        self._build_tree_thread.start()

    def _on_background_tree_built(self, artifacts):
        self._hide_busy(self._build_tree_busy)
        self._build_tree_busy = None
        self._set_build_controls_enabled(True)
        self._build_tree_thread = None
        self._build_tree_worker = None
        self._apply_built_tree_artifacts(artifacts)

    def _on_background_tree_build_failed(self, error_text: str):
        self._hide_busy(self._build_tree_busy)
        self._build_tree_busy = None
        self._set_build_controls_enabled(True)
        self._build_tree_thread = None
        self._build_tree_worker = None
        self._error(f"Не удалось построить дерево: {error_text}")

    def _apply_built_tree_artifacts(self, artifacts: AnalysisArtifacts):
        self.state.tree = artifacts.tree
        self.state.tree_built = True
        self.state.tree_loaded = True
        self.state.node_rows = artifacts.node_rows
        self.state.tree_method = artifacts.tree_info.get("build_method", self.combo_tree_method.currentText())
        self.state.tree_support_kind = artifacts.tree_info.get("support_kind", "observed")
        self.state.tree_support_label = artifacts.tree_info.get("support_label", "Bootstrap")
        self.lbl_tree_source.setText("Построено по выравниванию")
        self._apply_tree_artifacts(artifacts)
        self.tabs.setCurrentIndex(3)
        self.nav_list.setCurrentRow(3)
        self._append_log(f"Построено дерево методом: {self.state.tree_method}.")
        self.status_bar.showMessage(f"Дерево построено: {self.state.tree_method}", 4000)
        self._refresh_state_view()

    def load_alignment(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбор файла выравнивания", "", "Alignment files (*.nex *.nexus *.nexorg *.fasta *.fa *.fas *.txt);;All files (*.*)")
        if not path:
            return
        try:
            self.service.inspect_alignment_file(path)
            artifacts = self.service.load_alignment(path)
        except Exception as exc:
            self._error(f"Не удалось загрузить выравнивание: {exc}")
            return
        self.state.alignment_file = path
        self.state.alignment_loaded = True
        self.state.alignment = artifacts.alignment
        self.state.source_format = f"alignment:{Path(path).suffix.lower()}"
        self.alignment_path_edit.setText(path)
        self._fill_alignment_summary(artifacts)
        self._append_log(f"Загружено выравнивание: {path}")
        self.status_bar.showMessage("Выравнивание загружено", 4000)
        self._refresh_state_view()
    def load_tree(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбор файла дерева", "", "Tree files (*.nex *.nexus *.nwk *.newick *.tree *.txt);;All files (*.*)")
        if not path:
            return
        try:
            self.service.inspect_tree_file(path)
            artifacts = self.service.load_tree(path)
        except Exception as exc:
            self._error(f"Не удалось загрузить дерево: {exc}")
            return
        self.state.tree_file = path
        self.state.tree_loaded = True
        self.state.tree_built = True
        self.state.tree = artifacts.tree
        self.state.node_rows = artifacts.node_rows
        self.state.tree_method = artifacts.tree_info.get("build_method", "Loaded from file")
        self.state.tree_support_kind = artifacts.tree_info.get("support_kind", "observed")
        self.state.tree_support_label = artifacts.tree_info.get("support_label", "Bootstrap")
        self.tree_path_edit.setText(path)
        self.lbl_tree_source.setText("Загружено из файла и визуализировано")
        self._apply_tree_artifacts(artifacts)
        self.tabs.setCurrentIndex(3)
        self.nav_list.setCurrentRow(3)
        self._append_log(f"Загружено и визуализировано дерево: {path}")
        self.status_bar.showMessage("Файл дерева загружен и визуализирован", 4000)
        self._refresh_state_view()
    def _sync_selected_model_path(self):
        data = self.combo_model_select.currentData()
        if isinstance(data, dict):
            self.model_path_edit.setText(data.get("path", ""))
        else:
            self.model_path_edit.clear()
    def load_model(self):
        data = self.combo_model_select.currentData()
        if not isinstance(data, dict):
            self._error("Сначала выберите модель в выпадающем списке.")
            return
        model_path = data["path"]
        metadata_path = data.get("metadata")
        busy = self._show_busy("Загрузка модели", "Загружается модель Random Forest...")
        try:
            metadata, loaded_now = self.service.load_model(model_path, metadata_path)
        except Exception as exc:
            self._hide_busy(busy)
            self._error(f"Не удалось загрузить модель: {exc}")
            return
        self._hide_busy(busy)
        self.state.model_file = model_path
        self.state.model_loaded = True
        self.state.model_metadata = metadata
        self.model_path_edit.setText(model_path)
        if loaded_now:
            self._append_log(f"Загружена модель: {model_path}")
            self.status_bar.showMessage("Модель загружена", 4000)
        else:
            self._append_log(f"Модель уже была в памяти: {model_path}")
            self.status_bar.showMessage("Модель уже загружена, повторная загрузка не требуется", 5000)
        self._refresh_state_view()
    def validate_input(self):
        lines = []
        self.state.input_validated = False
        if self.state.alignment_file:
            try:
                meta = self.service.inspect_alignment_file(self.state.alignment_file)
                lines.append(f"Выравнивание: файл корректный ({meta['format']}).")
                lines.append("Найдены блоки: " + ", ".join(meta.get("present_blocks", [])))
                if self.state.alignment is not None:
                    aln = self.state.alignment
                    lines.append(f"Таксонов: {len(aln)}, длина выравнивания: {aln.get_alignment_length()}.")
            except Exception as exc:
                lines.append(f"Выравнивание: ошибка проверки — {exc}")
        else:
            lines.append("Выравнивание не загружено.")
        if self.state.tree_file:
            try:
                meta = self.service.inspect_tree_file(self.state.tree_file)
                lines.append(f"Дерево: файл корректный ({meta['format']}).")
                lines.append("Найдены блоки: " + ", ".join(meta.get("present_blocks", [])))
                if self.state.tree is not None:
                    lines.append(f"Терминальных узлов: {len(self.state.tree.get_terminals())}.")
            except Exception as exc:
                lines.append(f"Дерево: ошибка проверки — {exc}")
        else:
            lines.append("Дерево не загружено.")
        if self.state.model_loaded:
            lines.append("Модель Random Forest загружена.")
        else:
            lines.append("Модель Random Forest не загружена.")
        if self.state.alignment is not None and self.state.tree is not None:
            aln_taxa = {record.id for record in self.state.alignment}
            tree_taxa = {clade.name for clade in self.state.tree.get_terminals()}
            common = len(aln_taxa & tree_taxa)
            lines.append(f"Совпадающих таксонов между alignment и tree: {common}/{len(aln_taxa)}.")
        self.state.input_validated = any("файл корректный" in line for line in lines)
        self.validation_notes.setPlainText("\n".join(lines))
        self._append_log("Проверка входных данных выполнена.")
        self._refresh_state_view()
    def build_tree(self):
        if self.state.alignment is None:
            self._error("Для построения дерева сначала загрузите NEXUS/FASTA файл с выравниванием.")
            return
        method = self.combo_tree_method.currentText()
        if method == "Maximum Likelihood":
            self._start_background_tree_build(method)
            return
        try:
            artifacts = self.service.build_tree_from_alignment(self.state.alignment, method=method)
        except Exception as exc:
            self._error(f"Не удалось построить дерево: {exc}")
            return
        self._apply_built_tree_artifacts(artifacts)

    def predict_bootstrap(self):
        if self.state.tree is None or self.state.alignment is None:
            self._error("Для предсказания нужен и alignment, и tree.")
            return
        if not self.state.model_loaded:
            self._error("Сначала выберите и загрузите модель Random Forest.")
            return
        busy = self._show_busy("Предсказание bootstrap", "Модель считает predicted bootstrap для внутренних узлов...")
        try:
            predictions = self.service.predict_bootstrap(self.state.tree, self.state.alignment)
        except Exception as exc:
            self._hide_busy(busy)
            self._error(f"Не удалось выполнить предсказание: {exc}")
            return
        self._hide_busy(busy)
        pred_map = {row["node_id"]: row["predicted_bootstrap"] for row in predictions}
        pred_rows = {row["node_id"]: row for row in predictions}
        for row in self.state.node_rows:
            pred_info = pred_rows.get(row["node_id"], {})
            row["predicted_bootstrap"] = pred_info.get("predicted_bootstrap")
            row["feature_influences"] = pred_info.get("feature_influences", [])
            row["feature_influence_text"] = pred_info.get("feature_influence_text", "")
            if row["predicted_bootstrap"] is not None:
                row["support_source"] = "predicted_rf"
        self.state.analysis_completed = True
        self._populate_node_rows(self.state.node_rows)
        self._update_tree_view()
        self.tabs.setCurrentIndex(2)
        self.nav_list.setCurrentRow(2)
        self._append_log("Выполнено ML-предсказание bootstrap для внутренних узлов и рассчитаны влияющие параметры.")
        self.status_bar.showMessage("Predicted bootstrap рассчитан", 4000)
        self._refresh_state_view()
    def export_tree(self):
        if self.state.tree is None:
            self._error("Нет дерева для сохранения.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить дерево", "tree_built.nwk", "Newick (*.nwk *.newick);;All files (*.*)")
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
        support = "-" if info.get("bootstrap") is None else f"{info['bootstrap']:.3f}"
        predicted = "-" if info.get("predicted_bootstrap") is None else f"{info['predicted_bootstrap']:.2f}"
        influence = info.get("feature_influence_text") or "Нет данных о влияющих параметрах."
        self.result_details.setPlainText(
            f"Узел: {info['node_id']}\n"
            f"Длина ветви: {info['branch_length']:.6f}\n"
            f"Глубина: {info['depth']:.6f}\n"
            f"Количество листьев: {info['leaf_count']}\n"
            f"{self.state.tree_support_label}: {support}\n"
            f"Predicted bootstrap (RF): {predicted}\n"
            f"Источник: {info.get('support_source', '-')}\n"
            f"Листья поддерева: {info['descendants']}\n\n"
            f"{influence}"
        )
    def _fill_alignment_summary(self, artifacts: AnalysisArtifacts):
        info = artifacts.alignment_info
        self.validation_notes.setPlainText(
            f"Alignment загружен успешно.\n"
            f"Таксонов: {info.get('taxa_count', 0)}\n"
            f"Длина выравнивания: {info.get('length', 0)}\n"
            f"Доля gap/missing: {info.get('gap_fraction', 0.0):.4f}\n"
            f"Перед анализом можно запустить проверку структуры файла."
        )
    def _apply_tree_artifacts(self, artifacts: AnalysisArtifacts):
        tree_info = artifacts.tree_info
        support_min = tree_info.get('support_min')
        support_max = tree_info.get('support_max')
        support_range = '-' if support_min is None or support_max is None else f"{support_min:.2f} .. {support_max:.2f}"
        self.tree_build_preview.setPlainText(
            f"Метод: {tree_info.get('build_method', self.state.tree_method)}\n"
            f"Терминальных узлов: {tree_info.get('taxa_count', 0)}\n"
            f"Внутренних узлов: {tree_info.get('internal_nodes', 0)}\n"
            f"Суммарная длина ветвей: {tree_info.get('total_branch_length', 0.0):.6f}\n"
            f"Rooted: {'Да' if tree_info.get('is_rooted') else 'Нет'}\n"
            f"Поддержка узлов: {tree_info.get('support_label', self.state.tree_support_label)}\n"
            f"Диапазон support: {support_range}"
        )
        self._populate_node_rows(artifacts.node_rows)
        self._update_tree_view()
    def _populate_node_rows(self, rows: list[dict]):
        self.results_table.setRowCount(len(rows))
        for row_idx, info in enumerate(rows):
            values = [
                info['node_id'],
                f"{info['branch_length']:.6f}",
                f"{info['depth']:.6f}",
                str(info['leaf_count']),
                '-' if info.get('bootstrap') is None else f"{info['bootstrap']:.3f}",
                '-' if info.get('predicted_bootstrap') is None else f"{info['predicted_bootstrap']:.2f}",
                info.get('support_source', '-'),
                info['descendants'],
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                tooltip = info.get('feature_influence_text')
                if tooltip:
                    item.setToolTip(tooltip)
                if col_idx == 5 and info.get('predicted_bootstrap') is not None:
                    pred = float(info['predicted_bootstrap'])
                    if pred < 50:
                        item.setBackground(Qt.red)
                    elif pred < 75:
                        item.setBackground(Qt.yellow)
                    else:
                        item.setBackground(Qt.green)
                self.results_table.setItem(row_idx, col_idx, item)
        if rows:
            self.results_table.selectRow(0)
            self.show_selected_node_details()
    def _update_tree_view(self):
        prediction_map = {row['node_id']: row['predicted_bootstrap'] for row in self.state.node_rows if row.get('predicted_bootstrap') is not None}
        support_map = {row['node_id']: row['bootstrap'] for row in self.state.node_rows if row.get('bootstrap') is not None}
        show_support = self.state.tree_support_kind == "ml_ultrafast_bootstrap"
        node_id_map = build_clade_node_id_map(self.state.tree) if self.state.tree is not None else {}
        node_tooltip_map = {row['node_id']: row.get('feature_influence_text', '') for row in self.state.node_rows if row.get('feature_influence_text')}
        self.tree_canvas.set_tree(
            self.state.tree,
            self.chk_show_branch_lengths.isChecked(),
            prediction_map,
            self.chk_show_predictions.isChecked(),
            node_id_map,
            support_map=support_map,
            show_support_values=show_support,
            support_label=self.state.tree_support_label,
            node_tooltip_map=node_tooltip_map,
        )
    def _zoom_in_tree(self):
        self.tree_canvas.zoom_in()
    def _zoom_out_tree(self):
        self.tree_canvas.zoom_out()
    def _fit_tree(self):
        self.tree_canvas.fit_view()
    def _apply_leaf_spacing(self):
        self.tree_canvas.set_leaf_spacing(self.leaf_spacing_slider.value() / 10.0)
    def _refresh_state_view(self):
        self.lbl_tree_loaded.setText("Да" if self.state.tree_loaded else "Нет")
        self.lbl_alignment_loaded.setText("Да" if self.state.alignment_loaded else "Нет")
        self.lbl_model_loaded.setText("Да" if self.state.model_loaded else "Нет")
        self.lbl_validated.setText("Да" if self.state.input_validated else "Нет")
        self.lbl_tree_built.setText("Да" if self.state.tree_built else "Нет")
        self.lbl_source_format.setText(self.state.source_format or "-")
    def _append_log(self, text: str):
        timestamped = f"[{self._now()}] {text}"
        if hasattr(self, 'export_log'):
            self.export_log.appendPlainText(timestamped)
    def _error(self, text: str):
        self.state.last_error = text
        QMessageBox.critical(self, "Ошибка", text)
        self.status_bar.showMessage(text, 6000)
    def _now(self):
        return datetime.now().strftime("%H:%M:%S")
    def show_about(self):
        QMessageBox.about(self, "О программе", "Рабочий прототип для диплома:\n- загрузка NEXUS/FASTA;\n- построение NJ-дерева;\n- предсказание bootstrap моделью Random Forest;\n- визуализация дерева с подсветкой по достоверности.")
def run():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
