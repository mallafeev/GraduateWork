from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree
from .explanation import FeatureExplanationBuilder
from .feature_constants import FEATURE_COLUMNS
from .feature_extractor import TreeFeatureExtractor
from .model_loader import FeatureImportanceRepository, MetadataRepository, ModelInferenceError, SerializedModelLoader

class FeatureFrameBuilder:
    def build(self, feature_rows, expected_columns: list[str] | None = None) -> pd.DataFrame:
        X = pd.DataFrame([row.features for row in feature_rows])
        target_cols = expected_columns or FEATURE_COLUMNS
        # 1. Вычисляем производные признаки "на лету", если они есть в ожидаемом списке
        X = self._compute_derived_features(X, target_cols)
        # 2. Приводим таблицу к точному списку признаков модели (лишние удаляются, недостающие = NaN)
        return X.reindex(columns=target_cols, fill_value=np.nan)

    @staticmethod
    def _compute_derived_features(X: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
        if "bl_cv" in expected_cols and "bl_cv" not in X.columns:
            X["bl_cv"] = X["std_child_branch_length"] / (X["mean_child_branch_length"] + 1e-6)
        if "gap_var_clade" in expected_cols and "gap_var_clade" not in X.columns:
            X["gap_var_clade"] = (X["gap_fraction_clade"] - X["gap_fraction_global"]).abs()
        if "gc_extreme" in expected_cols and "gc_extreme" not in X.columns:
            X["gc_extreme"] = ((X["gc_mean_clade"] - X["gc_mean_global"]).abs() > 0.15).astype(float)
        if "tiny_clade" in expected_cols and "tiny_clade" not in X.columns:
            X["tiny_clade"] = (X["n_leaves_subtree"] <= 3).astype(float)
        if "deep_node" in expected_cols and "deep_node" not in X.columns:
            max_depth = X["depth"].max()
            if pd.isna(max_depth) or max_depth <= 0:
                max_depth = 1.0
            X["deep_node"] = X["depth"] / (max_depth + 1e-6)
        return X


class PredictionExecutor:
    def predict(self, model, X: pd.DataFrame):
        try:
            return model.predict(X)
        except Exception:
            # Фоллбэк для пайплайнов: вручную импутим и предсказываем
            if hasattr(model, 'named_steps') and 'rf' in model.named_steps:
                rf = model.named_steps['rf']
                imputer = model.named_steps.get('imputer')
                X_num = X.copy().astype(float)
                if imputer is not None and hasattr(imputer, 'statistics_'):
                    stats = list(imputer.statistics_)
                    for idx, col in enumerate(X_num.columns):
                        fill = stats[idx] if idx < len(stats) else 0.0
                        X_num[col] = X_num[col].fillna(fill)
                else:
                    X_num = X_num.fillna(X_num.median(numeric_only=True))
                return rf.predict(X_num)
            raise

class PredictionResultAssembler:
    def assemble(self, feature_rows, preds, explanations) -> list[dict]:
        results = []
        for row, pred, explanation in zip(feature_rows, preds, explanations):
            pred = max(0.0, min(100.0, float(pred)))
            results.append({
                'node_id': row.node_id,
                'predicted_bootstrap': pred,
                'feature_influences': explanation['items'],
                'feature_influence_text': explanation['text'],
                **row.features,
            })
        return results

class RandomForestBootstrapPredictor:
    def __init__(self) -> None:
        self.model = None
        self.metadata = {}
        self.model_path = ''
        self.metadata_path = ''
        self.feature_importance_map: dict[str, float] = {}
        self.model_loader = SerializedModelLoader()
        self.metadata_repository = MetadataRepository()
        self.importance_repository = FeatureImportanceRepository()
        self.feature_extractor = TreeFeatureExtractor()
        self.frame_builder = FeatureFrameBuilder()
        self.prediction_executor = PredictionExecutor()
        self.explanation_builder = FeatureExplanationBuilder()
        self.result_assembler = PredictionResultAssembler()

    def load(self, model_path: str | Path, metadata_path: str | Path | None = None) -> bool:
        resolved_model_path = Path(model_path).resolve()
        if not resolved_model_path.exists():
            raise ModelInferenceError(f'Файл модели не найден: {resolved_model_path}')
        metadata, resolved_metadata = self.metadata_repository.load(metadata_path)
        if self.model is not None and self.model_path == str(resolved_model_path) and self.metadata_path == resolved_metadata:
            return False
        self.model = self.model_loader.load(resolved_model_path)
        self.model_path = str(resolved_model_path)
        self.metadata_path = resolved_metadata
        self.feature_importance_map = self.importance_repository.load(self.model, resolved_model_path)
        self.metadata = metadata
        self.metadata.setdefault('feature_importances', self.feature_importance_map)
        return True

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_tree(self, tree: Tree, alignment: MultipleSeqAlignment) -> list[dict]:
        if self.model is None:
            raise ModelInferenceError('Модель не загружена.')
        
        feature_rows = self.feature_extractor.extract(tree, alignment)
        if not feature_rows:
            return []
            
        # 🔑 Динамически берём список признаков из метаданных загруженной модели
        # Если metadata нет или в нём нет ключа, используем fallback из feature_constants.py
        expected_cols = self.metadata.get("feature_columns", FEATURE_COLUMNS)
        
        X = self.frame_builder.build(feature_rows, expected_columns=expected_cols)
        preds = self.prediction_executor.predict(self.model, X)
        explanations = self.explanation_builder.build(X, self.feature_importance_map)
        return self.result_assembler.assemble(feature_rows, preds, explanations)