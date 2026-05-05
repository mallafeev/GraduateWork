from pathlib import Path
import subprocess

# ПУТИ
IQTREE_PATH = r"iqtree-3.1.1-Windows\bin\iqtree3.exe"

input_dir = Path("ml2/datasets")
output_dir = Path("ml2/trees")
output_dir.mkdir(parents=True, exist_ok=True)

# ФОРМАТЫ
valid_ext = [".nex", ".nexus", ".nexorg", ".fasta", ".fa"]

files = [f for f in input_dir.glob("*") if f.suffix.lower() in valid_ext]

print(f"Найдено файлов: {len(files)}")

for aln in files:
    name = aln.stem
    prefix = output_dir / name

    cmd = [
        IQTREE_PATH,
        "-s", str(aln),      # ВАЖНО для твоих nexorg
        "-m", "MFP",
        "-B", "1000", 
        "-bnni",
        "-T", "AUTO",
        "--prefix", str(prefix),
        "--redo"
    ]

    print(f"\nЗапуск: {aln.name}")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Готово: {name}")
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка: {name}")