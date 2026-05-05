# save_project.py
from pathlib import Path

def dump_project_for_ai(root_dir: str = ".", output_file: str = "project_dump.txt"):
    root = Path(root_dir)
    output = Path(output_file)
    
    # Папки, которые точно не нужно анализировать
    SKIP_DIRS = {"__pycache__", ".venv", "venv", "env", ".git", "node_modules", ".idea", ".vscode", "build", "dist"}
    
    print(f"🔍 Сканирую: {root.absolute()}")
    
    with output.open("w", encoding="utf-8") as f:
        # Сортируем для стабильного порядка
        py_files = sorted(root.rglob("*.py"))
        count = 0
        
        for py_file in py_files:
            # Пропускаем файлы внутри мусорных папок
            if any(part in SKIP_DIRS for part in py_file.parts):
                continue
                
            rel_path = py_file.relative_to(root)
            
            # Чёткие разделители, которые ИИ легко распознаёт
            f.write(f"\n{'='*70}\n")
            f.write(f"=== FILE: {rel_path} ===\n")
            f.write(f"{'='*70}\n\n")
            
            try:
                # errors="replace" предотвратит крах на странных символах
                content = py_file.read_text(encoding="utf-8", errors="replace")
                f.write(content)
            except Exception as e:
                f.write(f"[⚠️ ОШИБКА ЧТЕНИЯ: {e}]\n")
                
            f.write(f"\n{'='*70}\n")
            f.write(f"=== END OF FILE: {rel_path} ===\n")
            f.write(f"{'='*70}\n\n")
            count += 1
            
    print(f"✅ Готово! Обработано файлов: {count}")
    print(f"📄 Результат сохранён: {output.absolute()}")

if __name__ == "__main__":
    # Запускать из корня проекта
    dump_project_for_ai()