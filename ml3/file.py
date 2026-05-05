import os

root_folder = "C:/GitHubMain/GraduateWork/ml3/datasets" 
deleted_count = 0

print(root_folder)
for root, dirs, files in os.walk(root_folder):
    if "tree.contree" in files:
        file_path = os.path.join(root, "tree.contree")
        try:
            os.remove(file_path)
            print(f"Удалён: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Ошибка при удалении {file_path}: {e}")

print(f"\nУдалено файлов: {deleted_count}")