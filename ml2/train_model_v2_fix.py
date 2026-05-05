
from __future__ import annotations

import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


def parse_fasta(text: str) -> Dict[str, str]:
    seqs = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].strip()
            seqs[current] = []
        else:
            if current is None:
                continue
            seqs[current].append(line.replace(" ", ""))
    return {k: "".join(v) for k, v in seqs.items()}


def _extract_nexus_taxa(text: str) -> List[str]:
    m = re.search(r"BEGIN\s+TAXA;(.+?)END;", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    block = m.group(1)
    tm = re.search(r"TAXLABELS(.+?);", block, flags=re.IGNORECASE | re.DOTALL)
    if not tm:
        return []
    tokens = re.split(r"\s+", tm.group(1).strip())
    return [t.strip() for t in tokens if t.strip()]


def _parse_nexus_matrix_block(block: str, known_taxa: List[str]) -> Dict[str, str]:
    mm = re.search(r"\bMATRIX\b(.+?);", block, flags=re.IGNORECASE | re.DOTALL)
    if not mm:
        return {}
    matrix_text = mm.group(1)

    seqs = defaultdict(list)
    known_taxa_sorted = sorted(known_taxa, key=len, reverse=True)

    for raw_line in matrix_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("["):
            continue

        matched = False
        for taxon in known_taxa_sorted:
            if stripped.startswith(taxon):
                rest = stripped[len(taxon):].strip().replace(" ", "")
                if rest:
                    seqs[taxon].append(rest)
                matched = True
                break

        if not matched:
            parts = stripped.split()
            if len(parts) >= 2:
                taxon = parts[0]
                seq = "".join(parts[1:]).replace(" ", "")
                seqs[taxon].append(seq)

    return {k: "".join(v) for k, v in seqs.items()}


def parse_nexus_alignment(text: str) -> Dict[str, str]:
    known_taxa = _extract_nexus_taxa(text)
    char_blocks = re.findall(r"BEGIN\s+CHARACTERS;(.+?)END;", text, flags=re.IGNORECASE | re.DOTALL)
    if not char_blocks:
        raise ValueError("В файле не найдено BEGIN CHARACTERS.")

    merged = defaultdict(list)
    for block in char_blocks:
        part = _parse_nexus_matrix_block(block, known_taxa)
        for taxon, seq in part.items():
            merged[taxon].append(seq)

    if not merged:
        raise ValueError("Не удалось извлечь последовательности из MATRIX.")
    return {k: "".join(v) for k, v in merged.items()}


def load_alignment_file(path: str | Path) -> Dict[str, str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    upper = text.upper()
    if "#NEXUS" in upper or "BEGIN CHARACTERS;" in upper:
        return parse_nexus_alignment(text)
    if text.lstrip().startswith(">"):
        return parse_fasta(text)
    raise ValueError(f"Неподдерживаемый формат alignment: {path}")


def clean_newick(tree_str: str) -> str:
    return re.sub(r"\[.*?\]", "", tree_str.strip()).strip()


def extract_newick_from_text(text: str) -> str:
    text = text.strip()
    tree_match = re.search(r"\bTREE\b\s+[^=]+=\s*(\(.*?;)", text, flags=re.IGNORECASE | re.DOTALL)
    if tree_match:
        return clean_newick(tree_match.group(1))
    first_paren = text.find("(")
    last_semicolon = text.rfind(";")
    if first_paren != -1 and last_semicolon != -1 and last_semicolon > first_paren:
        return clean_newick(text[first_paren:last_semicolon + 1])
    raise ValueError("Не удалось извлечь Newick из файла дерева.")


def load_tree_file(path: str | Path):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    newick = extract_newick_from_text(text)
    tree = Phylo.read(io.StringIO(newick), "newick")
    return tree, newick


VALID_DNA = set("ACGT")
DNA_PLUS_GAP = set("ACGT-?")


def sanitize_seq(seq: str) -> str:
    return "".join(ch.upper() for ch in seq if ch.upper() in DNA_PLUS_GAP)


def p_distance(seq1: str, seq2: str) -> float:
    mismatches = 0
    valid = 0
    for a, b in zip(seq1, seq2):
        if a in {"-", "?"} or b in {"-", "?"}:
            continue
        if a not in VALID_DNA or b not in VALID_DNA:
            continue
        valid += 1
        if a != b:
            mismatches += 1
    return np.nan if valid == 0 else mismatches / valid


def seq_gap_fraction(seq: str) -> float:
    return np.nan if not seq else sum(ch in {"-", "?"} for ch in seq) / len(seq)


def seq_gc_fraction(seq: str) -> float:
    letters = [ch for ch in seq if ch in VALID_DNA]
    return np.nan if not letters else sum(ch in {"G", "C"} for ch in letters) / len(letters)


def alignment_stats(seqs: Dict[str, str]) -> Dict[str, float]:
    items = [sanitize_seq(s) for s in seqs.values()]
    if not items:
        return {
            "taxa_count": np.nan,
            "alignment_length": np.nan,
            "gap_fraction_global": np.nan,
            "gc_mean_global": np.nan,
            "gc_std_global": np.nan,
            "variable_site_fraction_global": np.nan,
        }

    n = len(items)
    L = max(len(s) for s in items)
    items = [s.ljust(L, "-") for s in items]

    gap_fraction_global = np.mean([seq_gap_fraction(s) for s in items])
    gc_values = [seq_gc_fraction(s) for s in items]
    gc_mean_global = float(np.nanmean(gc_values))
    gc_std_global = float(np.nanstd(gc_values))

    variable_sites = 0
    informative_positions = 0
    for i in range(L):
        col = [s[i] for s in items if s[i] in VALID_DNA]
        if len(col) < 2:
            continue
        informative_positions += 1
        if len(set(col)) > 1:
            variable_sites += 1

    variable_site_fraction = variable_sites / informative_positions if informative_positions else np.nan

    return {
        "taxa_count": float(n),
        "alignment_length": float(L),
        "gap_fraction_global": float(gap_fraction_global),
        "gc_mean_global": gc_mean_global,
        "gc_std_global": gc_std_global,
        "variable_site_fraction_global": float(variable_site_fraction),
    }


def get_leaf_names(clade) -> List[str]:
    return [leaf.name for leaf in clade.get_terminals() if leaf.name]


def mean_pairwise_distance(names: List[str], seqs: Dict[str, str]) -> float:
    if len(names) < 2:
        return 0.0
    vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1 = sanitize_seq(seqs.get(names[i], ""))
            s2 = sanitize_seq(seqs.get(names[j], ""))
            if not s1 or not s2:
                continue
            d = p_distance(s1, s2)
            if not np.isnan(d):
                vals.append(d)
    return np.nan if not vals else float(np.mean(vals))


def clade_alignment_stats(names: List[str], seqs: Dict[str, str]) -> Dict[str, float]:
    sub = {k: seqs[k] for k in names if k in seqs}
    if not sub:
        return {
            "gap_fraction_clade": np.nan,
            "gc_mean_clade": np.nan,
            "gc_std_clade": np.nan,
            "mean_pairwise_pdist_clade": np.nan,
        }

    gap_fraction_clade = float(np.nanmean([seq_gap_fraction(sanitize_seq(v)) for v in sub.values()]))
    gc_vals = [seq_gc_fraction(sanitize_seq(v)) for v in sub.values()]
    gc_mean_clade = float(np.nanmean(gc_vals))
    gc_std_clade = float(np.nanstd(gc_vals))
    mpd = mean_pairwise_distance(list(sub.keys()), sub)

    return {
        "gap_fraction_clade": gap_fraction_clade,
        "gc_mean_clade": gc_mean_clade,
        "gc_std_clade": gc_std_clade,
        "mean_pairwise_pdist_clade": mpd,
    }


def subtree_balance(clade) -> float:
    children = clade.clades
    if len(children) < 2:
        return 0.0
    sizes = [len(ch.get_terminals()) for ch in children]
    total = sum(sizes)
    return 0.0 if total == 0 else abs(sizes[0] - sizes[1]) / total


def extract_node_features(tree, seqs: Dict[str, str], dataset_id: str) -> pd.DataFrame:
    global_stats = alignment_stats(seqs)
    depths = tree.depths()
    rows = []
    node_idx = 0

    for clade in tree.find_clades(order="level"):
        if clade.is_terminal():
            continue

        conf = clade.confidence
        if conf is None:
            continue

        bootstrap = float(conf)
        if bootstrap <= 1.0:
            bootstrap *= 100.0

        branch_length = max(float(clade.branch_length or 0.0), 0.0)
        leaf_names = get_leaf_names(clade)
        n_leaves = len(leaf_names)
        clade_stats = clade_alignment_stats(leaf_names, seqs)

        child_branch_lengths = [max(float(ch.branch_length or 0.0), 0.0) for ch in clade.clades]

        rows.append({
            "dataset_id": dataset_id,
            "node_id": f"{dataset_id}_node_{node_idx}",
            "target_bootstrap": bootstrap,
            "branch_length": branch_length,
            "depth": float(depths.get(clade, 0.0)),
            "n_leaves_subtree": float(n_leaves),
            "subtree_fraction": float(n_leaves / global_stats["taxa_count"]) if global_stats["taxa_count"] else np.nan,
            "subtree_balance": float(subtree_balance(clade)),
            "mean_child_branch_length": float(np.mean(child_branch_lengths)) if child_branch_lengths else 0.0,
            "std_child_branch_length": float(np.std(child_branch_lengths)) if child_branch_lengths else 0.0,
            **global_stats,
            **clade_stats,
        })
        node_idx += 1

    return pd.DataFrame(rows)


ALIGNMENT_EXTS = {".nex", ".nexus", ".nexorg", ".fasta", ".fa", ".fas", ".txt"}
TREE_HINTS = ("tree", "bootstrap", "nwk", "newick", "con_", "majrule")


def detect_alignment_file(ds_path: Path) -> Path | None:
    candidates = [f for f in ds_path.iterdir() if f.is_file() and f.suffix.lower() in ALIGNMENT_EXTS]
    candidates = sorted(candidates, key=lambda p: (p.suffix.lower() not in {".nex", ".nexus", ".nexorg"}, p.name.lower()))
    return candidates[0] if candidates else None


def detect_tree_file(ds_path: Path, alignment_file: Path | None) -> Path | None:
    files = [f for f in ds_path.iterdir() if f.is_file()]
    if alignment_file is not None:
        files = [f for f in files if f.resolve() != alignment_file.resolve()]

    scored = []
    for f in files:
        name_l = f.name.lower()
        score = 0
        if any(h in name_l for h in TREE_HINTS):
            score += 3
        if f.suffix.lower() in {".nwk", ".newick", ".tree"}:
            score += 3
        if f.suffix == "":
            score += 1
        scored.append((score, f))

    scored.sort(key=lambda x: (-x[0], x[1].name.lower()))
    for _, f in scored:
        try:
            load_tree_file(f)
            return f
        except Exception:
            continue
    return None


def build_feature_table(datasets_root: str | Path):
    datasets_root = Path(datasets_root)
    all_frames = []
    audit = []

    for ds_path in sorted(datasets_root.iterdir()):
        if not ds_path.is_dir():
            continue

        dataset_id = ds_path.name
        alignment_file = detect_alignment_file(ds_path)
        tree_file = detect_tree_file(ds_path, alignment_file)

        status = {
            "dataset_id": dataset_id,
            "alignment_file": str(alignment_file) if alignment_file else None,
            "tree_file": str(tree_file) if tree_file else None,
            "ok": False,
            "error": None,
        }

        try:
            if alignment_file is None:
                raise FileNotFoundError("Не найден alignment-файл")
            if tree_file is None:
                raise FileNotFoundError("Не найден tree-файл")

            seqs = load_alignment_file(alignment_file)
            tree, _ = load_tree_file(tree_file)
            df = extract_node_features(tree, seqs, dataset_id)

            if df.empty:
                raise ValueError("Не извлечено ни одного внутреннего узла с bootstrap")

            all_frames.append(df)
            status["ok"] = True
            status["n_rows"] = int(len(df))

        except Exception as e:
            status["error"] = str(e)

        audit.append(status)

    if not all_frames:
        audit_df = pd.DataFrame(audit)
        raise RuntimeError("Не удалось собрать ни одной строки датасета. Проверь dataset_audit:\n" + audit_df.to_string(index=False))

    return pd.concat(all_frames, ignore_index=True), audit


FEATURE_COLUMNS = [
    "branch_length",
    "depth",
    "n_leaves_subtree",
    "subtree_fraction",
    "subtree_balance",
    "mean_child_branch_length",
    "std_child_branch_length",
    "taxa_count",
    "alignment_length",
    "gap_fraction_global",
    "gc_mean_global",
    "gc_std_global",
    "variable_site_fraction_global",
    "gap_fraction_clade",
    "gc_mean_clade",
    "gc_std_clade",
    "mean_pairwise_pdist_clade",
]


def train_model(df: pd.DataFrame):
    df = df.copy()

    df = df.dropna(subset=["target_bootstrap"])

    df = df.fillna(0)
    X = df[FEATURE_COLUMNS].copy()
    y = df["target_bootstrap"].copy()
    groups = df["dataset_id"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "n_datasets_total": int(groups.nunique()),
        "n_datasets_train": int(g_train.nunique()),
        "n_datasets_test": int(g_test.nunique()),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "feature_columns": FEATURE_COLUMNS,
        "train_datasets": sorted(g_train.unique().tolist()),
        "test_datasets": sorted(g_test.unique().tolist()),
    }

    rf = model.named_steps["rf"]
    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    pred_df = pd.DataFrame(index=X_test.index)

    pred_df["y_true"] = y_test
    pred_df["y_pred"] = pred

    # безопасно добавляем dataset_id
    pred_df["dataset_id"] = df.loc[X_test.index, "dataset_id"]

    return model, metrics, feature_importance, pred_df


def main():
    project_root = Path.cwd()
    datasets_root = project_root / "datasets_iqtree"
    output_dir = project_root / "ml_outputs_v1"
    output_dir.mkdir(exist_ok=True)

    print(f"[INFO] datasets root: {datasets_root}")
    print(f"[INFO] output dir:    {output_dir}")

    df, audit = build_feature_table(datasets_root)
    df.to_csv(output_dir / "node_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(output_dir / "dataset_audit.csv", index=False, encoding="utf-8-sig")

    model, metrics, feature_importance, pred_df = train_model(df)

    joblib.dump(model, output_dir / "model_v1.pkl")
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== TRAIN DONE ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nTop feature importance:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
