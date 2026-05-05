# ml/predictor_core.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree
from .explanation import FeatureExplanationBuilder
from .feature_constants import FEATURE_COLUMNS  # Оставляем как фоллбэк
from .feature_extractor import TreeFeatureExtractor
from .model_loader import FeatureImportanceRepository, MetadataRepository, ModelInferenceError, SerializedModelLoader

class FeatureFrameBuilder:
    # Принимаем список колонок динамически!
    def build(self, feature_rows, feature_columns: list[str]) -> pd.DataFrame:
        X = pd.DataFrame([row.features for row in feature_rows])
        # 1. Добавляем недостающие колонки как NaN (если экстрактор их не сгенерировал)
        for col in feature_columns:
            if col not in X.columns:
                X[col] = pd.NA
        # 2. Выбираем ТОЛЬКО те колонки, которые ожидает модель (отбрасываем лишние)
        return X[feature_columns]

class PredictionExecutor:
    def predict(self, model, X: pd.DataFrame):
        try:
            return model.predict(X)
        except Exception:
            # Фоллбэк для пайплайнов с импутером (как в старых версиях)
            if hasattr(model, 'named_steps') and 'rf' in model.named_steps:
                rf = model.named_steps['rf']
                imputer = model.named_steps.get('imputer')
                X_num = X.astype(float).copy()
                if imputer is not None and hasattr(imputer, 'statistics_'):
                    stats = list(imputer.statistics_)
                    for idx, col in enumerate(X_num.columns):
                        fill_value = stats[idx] if idx < len(stats) else 0.0
                        X_num[col] = X_num[col].fillna(fill_value)
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
        
        # Добавляем поле для хранения ожидаемых колонок
        self.expected_feature_columns = FEATURE_COLUMNS 

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
        self.metadata = metadata
        
        # === КЛЮЧЕВОЙ МОМЕНТ: Читаем колонки из метаданных ===
        self.expected_feature_columns = metadata.get("feature_columns")
        if not self.expected_feature_columns:
            # Если в метаданных нет списка (старая модель), используем константы по умолчанию
            self.expected_feature_columns = FEATURE_COLUMNS

        self.feature_importance_map = self.importance_repository.load(self.model, resolved_model_path)
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
        
        # Передаем ожидаемые колонки в билдер
        X = self.frame_builder.build(feature_rows, self.expected_feature_columns)
        
        preds = self.prediction_executor.predict(self.model, X)
        explanations = self.explanation_builder.build(X, self.feature_importance_map)
        return self.result_assembler.assemble(feature_rows, preds, explanations)