import os
import sys
from datetime import datetime

def extract_code_lines(root_dir, output_file="project_code_lines.txt", skip_empty=True):
    """
    Проходит по папкам, собирает только .py и .csv файлы,
    добавляет статистику в начало файла и безопасно фильтрует пустые строки/комментарии.
    """
    # ✅ БЕЛЫЙ СПИСОК: сканируем только эти расширения
    ALLOWED_EXTS = {".py"}
    
    # ❌ Папки, которые нужно полностью пропускать
    ignore_dirs = {
        "__pycache__", 
        ".pytest_cache", 
        "iqtree-3.1.1.-Windows", 
        "MrBayes-3.2.7.-WIN"
    }

    file_count = 0
    line_count = 0
    collected_lines = []

    print("🔍 Сканирование директории...")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Удаляем игнорируемые папки "на лету", чтобы os.walk в них не заходил
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for filename in filenames:
            _, ext = os.path.splitext(filename)
            
            # Пропускаем всё, кроме разрешённых расширений
            if ext.lower() not in ALLOWED_EXTS:
                continue

            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if skip_empty:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            # ⚠️ Комментарии убираем ТОЛЬКО в .py файлах, 
                            # чтобы не сломать CSV (где # иногда используется для заголовков)
                            if ext.lower() == ".py" and stripped.startswith(("#", "REM")):
                                continue
                        collected_lines.append(line)
                        line_count += 1
                file_count += 1
            except Exception as e:
                print(f"⚠️ Ошибка чтения {filepath}: {e}", file=sys.stderr)

    print(f"✅ Найдено файлов: {file_count}, строк: {line_count}")
    print("💾 Запись результата...")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("=" * 60 + "\n")
        out.write("📊 СТАТИСТИКА ИЗВЛЕЧЕНИЯ КОДА\n")
        out.write("=" * 60 + "\n")
        out.write(f"📁 Путь к проекту: {os.path.abspath(root_dir)}\n")
        out.write(f"📅 Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"📄 Обработано файлов: {file_count}\n")
        out.write(f"📝 Извлечено строк: {line_count}\n")
        out.write(f"🔍 Фильтр расширений: {', '.join(sorted(ALLOWED_EXTS))}\n")
        out.write("=" * 60 + "\n\n")

        for line in collected_lines:
            out.write(line)

    print(f"🎉 Готово! Результат сохранён в: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    # 🔧 НАСТРОЙКИ:
    PROJECT_DIR = "."  # "." = текущая папка, где лежит скрипт
    OUTPUT_FILE = "project_code_lines.txt"
    
    extract_code_lines(PROJECT_DIR, OUTPUT_FILE, skip_empty=True)