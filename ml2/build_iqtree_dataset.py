from pathlib import Path
import shutil

# ПУТИ
alignments_dir = Path("datasets")
trees_dir = Path("trees")
output_dir = Path("datasets_iqtree")
output_dir.mkdir(parents=True, exist_ok=True)

valid_ext = [".nex", ".nexus", ".nexorg", ".fasta", ".fa"]

datasets = []
for f in alignments_dir.glob("*"):
    if f.suffix.lower() in valid_ext:
        datasets.append(f)

datasets = sorted(datasets)

print(f"Найдено alignment-файлов: {len(datasets)}")

counter = 1

for aln in datasets:
    name = aln.stem
    tree_file = trees_dir / f"{name}.contree"

    if not tree_file.exists():
        print(f"❌ Нет дерева для: {name}")
        continue

    dt_folder = output_dir / f"dt{counter:03d}"
    dt_folder.mkdir(exist_ok=True)

    # копируем alignment (оставляем оригинальное имя)
    shutil.copy(aln, dt_folder / aln.name)

    # копируем дерево
    shutil.copy(tree_file, dt_folder / "tree.contree")

    print(f"✅ Создан: {dt_folder}")

    counter += 1

print("\n🎉 Готово!")