import sys
from pathlib import Path
import pytest

# Добавляем корень проекта в пути поиска модулей
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Хук, который срабатывает ПОСЛЕ завершения всех тестов и выводит сводную таблицу."""
    v1_mae = getattr(pytest, 'model_v1_mae', None)
    v1_rmse = getattr(pytest, 'model_v1_rmse', None)
    v1_r2 = getattr(pytest, 'model_v1_r2', None)

    v2_mae = getattr(pytest, 'model_v2_mae', None)
    v2_rmse = getattr(pytest, 'model_v2_rmse', None)
    v2_r2 = getattr(pytest, 'model_v2_r2', None)

    v3_mae = getattr(pytest, 'model_v3_mae', None)
    v3_rmse = getattr(pytest, 'model_v3_rmse', None)
    v3_r2 = getattr(pytest, 'model_v3_r2', None)

    # Если метрики не собраны (тесты упали), ничего не выводим
    if any(v is None for v in [v1_mae, v1_r2, v2_mae, v2_r2, v3_mae, v3_r2]):
        return

    terminalreporter.write_sep("=", "СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК МОДЕЛЕЙ")
    header = f"{'Модель':<8} | {'Алгоритм':<18} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}"
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * 65)
    terminalreporter.write_line(f"{'v1':<8} | {'Random Forest':<18} | {v1_mae:<8.3f} | {v1_rmse:<8.3f} | {v1_r2:<8.4f}")
    terminalreporter.write_line(f"{'v2':<8} | {'Random Forest v2':<18} | {v2_mae:<8.3f} | {v2_rmse:<8.3f} | {v2_r2:<8.4f}")
    terminalreporter.write_line(f"{'v3':<8} | {'Gradient Boosting':<18} | {v3_mae:<8.3f} | {v3_rmse:<8.3f} | {v3_r2:<8.4f}")
    terminalreporter.write_sep("=", "")