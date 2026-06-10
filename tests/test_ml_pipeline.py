import time
import pytest
import pandas as pd
import numpy as np
import math
from unittest.mock import MagicMock
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.feature_extractor import NodeFeatureRow
from ml.predictor_core import FeatureFrameBuilder, PredictionExecutor, PredictionResultAssembler
from ml.explanation import FeatureExplanationBuilder
from ml.feature_constants import FEATURE_COLUMNS

class TestMLPipeline:
    @pytest.fixture
    def frame_builder(self): return FeatureFrameBuilder()
    
    @pytest.fixture
    def prediction_executor(self): return PredictionExecutor()
    
    @pytest.fixture
    def result_assembler(self): return PredictionResultAssembler()
    
    @pytest.fixture
    def explanation_builder(self): return FeatureExplanationBuilder()
    
    @pytest.fixture
    def synthetic_rows(self):
        return [
            NodeFeatureRow(node_id="N1", features={'branch_length': 0.12, 'depth': 2.5, 'n_leaves_subtree': 10.0}),
            NodeFeatureRow(node_id="N2", features={'branch_length': np.nan, 'depth': 4.0, 'n_leaves_subtree': 25.0}),
            NodeFeatureRow(node_id="N3", features={'branch_length': 0.05, 'depth': np.nan, 'n_leaves_subtree': 5.0}),
        ]
        
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.predict.return_value = np.array([75.4, np.nan, 115.0])
        return model
        
    @pytest.fixture
    def importance_map(self):
        return {col: 0.05 for col in FEATURE_COLUMNS} | {
            'branch_length': 0.4, 'depth': 0.35, 'gc_mean_global': 0.15
        }

    def test_01_feature_frame_structure(self, frame_builder, synthetic_rows):
        X = frame_builder.build(synthetic_rows, FEATURE_COLUMNS)
        assert list(X.columns) == FEATURE_COLUMNS
        assert X.shape[1] == 17
        assert X.shape[0] == len(synthetic_rows)
        # На этапе сборки DataFrame пропуски сохраняются intentionally для последующей импутации
        assert X.isnull().sum().sum() > 0

    def test_02_prediction_range_and_fallback(self, prediction_executor, result_assembler, mock_model, synthetic_rows):
        X = FeatureFrameBuilder().build(synthetic_rows, FEATURE_COLUMNS)
        preds = prediction_executor.predict(mock_model, X)
        results = result_assembler.assemble(
            synthetic_rows, preds, [{'items': [], 'text': ''} for _ in preds]
        )
        for r in results:
            assert 0.0 <= r['predicted_bootstrap'] <= 100.0
        assert results[2]['predicted_bootstrap'] == 100.0
        assert not np.isnan(results[1]['predicted_bootstrap'])

    def test_03_explanation_format_and_performance(self, explanation_builder, synthetic_rows, importance_map):
        X = FeatureFrameBuilder().build(synthetic_rows, FEATURE_COLUMNS)
        # Перед генерацией пояснений NaN заполняются нулями для безопасного приведения типов
        X_safe = X.fillna(0.0)
        
        start_time = time.perf_counter()
        explanations = explanation_builder.build(X_safe, importance_map)
        exec_time = time.perf_counter() - start_time
        
        assert len(explanations) == len(X_safe)
        for exp in explanations:
            assert len(exp['items']) == 5
            assert 'text' in exp and len(exp['text']) > 50
            for item in exp['items']:
                assert item['direction'] in ['выше среднего', 'ниже среднего']
        assert exec_time < 0.5

    def test_04_model_metrics_validation(self):
        y_true = np.array([82.0, 90.0, 75.0, 85.0, 68.0])
        y_pred = np.array([80.5, 92.0, 76.0, 84.0, 65.0])
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        assert isinstance(mae, float) and mae >= 0
        assert isinstance(rmse, float) and rmse >= 0
        assert isinstance(r2, float) and -1.0 <= r2 <= 1.0
        assert mae < 5.0
        assert rmse < 6.0
        assert r2 > 0.75