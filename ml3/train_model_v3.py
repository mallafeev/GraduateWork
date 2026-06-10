from __future__ import annotations
import io
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ==================== PARSE FUNCTIONS (без изменений) ====================
def parse_fasta(text: str) -> Dict[str, str]:
    seqs = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line: continue
        if line.startswith(">"):
            current = line[1:].strip()
            seqs[current] = []
        else:
            if current is None: continue
            seqs[current].append(line.replace(" ", ""))
    return {k: "".join(v) for k, v in seqs.items()}

def _extract_nexus_taxa(text: str) -> List[str]:
    m = re.search(r"BEGIN\s+TAXA;(.+?)END;", text, flags=re.IGNORECASE | re.DOTALL)
    if not m: return []
    block = m.group(1)
    tm = re.search(r"TAXLABELS(.+?);", block, flags=re.IGNORECASE | re.DOTALL)
    if not tm: return []
    return [t.strip() for t in re.split(r"\s+", tm.group(1).strip()) if t.strip()]

def _parse_nexus_matrix_block(block: str, known_taxa: List[str]) -> Dict[str, str]:
    mm = re.search(r"\bMATRIX\b(.+?);", block, flags=re.IGNORECASE | re.DOTALL)
    if not mm: return {}
    seqs = defaultdict(list)
    known_taxa_sorted = sorted(known_taxa, key=len, reverse=True)
    for raw_line in mm.group(1).splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("["): continue
        matched = False
        for taxon in known_taxa_sorted:
            if stripped.startswith(taxon):
                rest = stripped[len(taxon):].strip().replace(" ", "")
                if rest: seqs[taxon].append(rest)
                matched = True
                break
        if not matched:
            parts = stripped.split()
            if len(parts) >= 2: seqs[parts[0]].append("".join(parts[1:]).replace(" ", ""))
    return {k: "".join(v) for k, v in seqs.items()}

def parse_nexus_alignment(text: str) -> Dict[str, str]:
    known_taxa = _extract_nexus_taxa(text)
    char_blocks = re.findall(r"BEGIN\s+CHARACTERS;(.+?)END;", text, flags=re.IGNORECASE | re.DOTALL)
    if not char_blocks: raise ValueError("В файле не найдено BEGIN CHARACTERS.")
    merged = defaultdict(list)
    for block in char_blocks:
        part = _parse_nexus_matrix_block(block, known_taxa)
        for taxon, seq in part.items(): merged[taxon].append(seq)
    if not merged: raise ValueError("Не удалось извлечь последовательности из MATRIX.")
    return {k: "".join(v) for k, v in merged.items()}

def load_alignment_file(path: str | Path) -> Dict[str, str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    upper = text.upper()
    if "#NEXUS" in upper or "BEGIN CHARACTERS;" in upper: return parse_nexus_alignment(text)
    if text.lstrip().startswith(">"): return parse_fasta(text)
    raise ValueError(f"Неподдерживаемый формат alignment: {path}")

def clean_newick(tree_str: str) -> str:
    return re.sub(r"\[.*?\]", "", tree_str.strip()).strip()

def extract_newick_from_text(text: str) -> str:
    text = text.strip()
    tree_match = re.search(r"\bTREE\b\s+[^=]+=\s*(\(.*?;)", text, flags=re.IGNORECASE | re.DOTALL)
    if tree_match: return clean_newick(tree_match.group(1))
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

# ==================== FEATURE ENGINEERING (FIXED) ====================
VALID_DNA = set("ACGT")
DNA_PLUS_GAP = set("ACGT-?")

def sanitize_seq(seq: str) -> str:
    return "".join(ch.upper() for ch in seq if ch.upper() in DNA_PLUS_GAP)

def p_distance(seq1: str, seq2: str) -> float:
    mismatches = valid = 0
    for a, b in zip(seq1, seq2):
        if a in {"-", "?"} or b in {"-", "?"}: continue
        if a not in VALID_DNA or b not in VALID_DNA: continue
        valid += 1
        if a != b: mismatches += 1
    return np.nan if valid == 0 else mismatches / valid

def seq_gap_fraction(seq: str) -> float:
    return np.nan if not seq else sum(ch in {"-", "?"} for ch in seq) / len(seq)

def seq_gc_fraction(seq: str) -> float:
    letters = [ch for ch in seq if ch in VALID_DNA]
    return np.nan if not letters else sum(ch in {"G", "C"} for ch in letters) / len(letters)

def alignment_stats(seqs: Dict[str, str]) -> Dict[str, float]:
    items = [sanitize_seq(s) for s in seqs.values()]
    if not items:
        return {"taxa_count": np.nan, "alignment_length": np.nan, "gap_fraction_global": np.nan,
                "gc_mean_global": np.nan, "gc_std_global": np.nan, "variable_site_fraction_global": np.nan}
    n = len(items)
    L = max(len(s) for s in items)
    items = [s.ljust(L, "-") for s in items]
    gap_fraction_global = np.mean([seq_gap_fraction(s) for s in items])
    gc_values = [seq_gc_fraction(s) for s in items]
    variable_sites = informative_positions = 0
    for i in range(L):
        col = [s[i] for s in items if s[i] in VALID_DNA]
        if len(col) < 2: continue
        informative_positions += 1
        if len(set(col)) > 1: variable_sites += 1
    return {
        "taxa_count": float(n), "alignment_length": float(L),
        "gap_fraction_global": float(gap_fraction_global),
        "gc_mean_global": float(np.nanmean(gc_values)),
        "gc_std_global": float(np.nanstd(gc_values)),
        "variable_site_fraction_global": float(variable_sites / informative_positions if informative_positions else np.nan)
    }

def mean_pairwise_distance(names: List[str], seqs: Dict[str, str]) -> float:
    if len(names) < 2: return 0.0
    vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1, s2 = sanitize_seq(seqs.get(names[i], "")), sanitize_seq(seqs.get(names[j], ""))
            if not s1 or not s2: continue
            d = p_distance(s1, s2)
            if not np.isnan(d): vals.append(d)
    return np.nan if not vals else float(np.mean(vals))

def clade_alignment_stats(names: List[str], seqs: Dict[str, str]) -> Dict[str, float]:
    sub = {k: seqs[k] for k in names if k in seqs}
    if not sub: return {"gap_fraction_clade": np.nan, "gc_mean_clade": np.nan, "gc_std_clade": np.nan, "mean_pairwise_pdist_clade": np.nan}
    return {
        "gap_fraction_clade": float(np.nanmean([seq_gap_fraction(sanitize_seq(v)) for v in sub.values()])),
        "gc_mean_clade": float(np.nanmean([seq_gc_fraction(sanitize_seq(v)) for v in sub.values()])),
        "gc_std_clade": float(np.nanstd([seq_gc_fraction(sanitize_seq(v)) for v in sub.values()])),
        "mean_pairwise_pdist_clade": mean_pairwise_distance(list(sub.keys()), sub)
    }

def subtree_balance(clade) -> float:
    children = clade.clades
    if len(children) < 2: return 0.0
    sizes = np.array([len(ch.get_terminals()) for ch in children])
    total = sizes.sum()
    if total == 0: return 0.0
    return float(np.std(sizes) / (total / len(children)))

def extract_node_features(tree, seqs: Dict[str, str], dataset_id: str) -> pd.DataFrame:
    global_stats = alignment_stats(seqs)
    depths = tree.depths()
    max_depth = max(depths.values()) if depths else 1.0
    rows = []
    node_idx = 0
    for clade in tree.find_clades(order="level"):
        if clade.is_terminal(): continue
        conf = clade.confidence
        if conf is None: continue
        bootstrap = float(conf)
        if bootstrap <= 1.0: bootstrap *= 100.0
        
        branch_length = max(float(clade.branch_length or 0.0), 0.0)
        depth = float(depths.get(clade, 0.0))
        n_leaves = len([leaf.name for leaf in clade.get_terminals() if leaf.name])
        clade_stats = clade_alignment_stats([leaf.name for leaf in clade.get_terminals() if leaf.name], seqs)
        child_bl = [max(float(ch.branch_length or 0.0), 0.0) for ch in clade.clades]
        
        rows.append({
            "dataset_id": dataset_id,
            "node_id": f"{dataset_id}_node_{node_idx}",
            "target_bootstrap": bootstrap,
            "branch_length": branch_length,
            "depth": depth,
            "subtree_fraction": float(n_leaves / global_stats["taxa_count"]) if global_stats["taxa_count"] else np.nan,
            "subtree_balance": subtree_balance(clade),
            "mean_child_branch_length": float(np.mean(child_bl)) if child_bl else 0.0,
            "std_child_branch_length": float(np.std(child_bl)) if child_bl else 0.0,
            "n_children": float(len(clade.clades)),
            "clade_size_log": float(np.log1p(n_leaves)),
            "relative_branch_length": float(branch_length / max_depth) if max_depth > 0 else 0.0,
            "depth_ratio": float(depth / max_depth) if max_depth > 0 else 0.0,
            **global_stats, **clade_stats
        })
        node_idx += 1
    return pd.DataFrame(rows)

FEATURE_COLUMNS = [
    "branch_length", "depth", "subtree_fraction", "subtree_balance",
    "mean_child_branch_length", "std_child_branch_length", "n_children",
    "clade_size_log", "relative_branch_length", "depth_ratio",
    "taxa_count", "alignment_length", "gap_fraction_global",
    "gc_mean_global", "gc_std_global", "variable_site_fraction_global",
    "gap_fraction_clade", "gc_mean_clade", "gc_std_clade", "mean_pairwise_pdist_clade"
]

def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df["target_bootstrap"].copy()
    groups = df["dataset_id"].copy()
    
    X = X.fillna(X.median())
    
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y))
    fold_metrics = []
    
    model_template = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
        min_samples_leaf=5, random_state=42
    )
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = model_template.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        oof_pred[test_idx] = np.clip(preds, 0.0, 100.0)
        
        fold_metrics.append({
            "fold": fold+1,
            "mae": mean_absolute_error(y_te, oof_pred[test_idx]),
            "rmse": np.sqrt(mean_squared_error(y_te, oof_pred[test_idx])),
            "r2": r2_score(y_te, oof_pred[test_idx])
        })
        print(f"Fold {fold+1}: MAE={fold_metrics[-1]['mae']:.3f}, RMSE={fold_metrics[-1]['rmse']:.3f}, R²={fold_metrics[-1]['r2']:.3f}")

    agg = pd.DataFrame(fold_metrics).mean()
    
    final_model = model_template.fit(X, y)
    importance = final_model.feature_importances_
    feat_imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance}).sort_values("importance", ascending=False)
    
    pred_df = X.copy()
    pred_df["y_true"] = y.values
    pred_df["y_pred"] = oof_pred
    pred_df["dataset_id"] = groups.values
    
    metrics = {
        "n_rows_total": int(len(df)),
        "n_datasets_total": int(groups.nunique()),
        "cv_mae": float(agg["mae"]),
        "cv_rmse": float(agg["rmse"]),
        "cv_r2": float(agg["r2"]),
        "feature_columns": FEATURE_COLUMNS,
        "fold_metrics": [m for m in fold_metrics]
    }
    
    return final_model, metrics, feat_imp, pred_df

ALIGNMENT_EXTS = {".nex", ".nexus", ".nexorg", ".fasta", ".fa", ".fas", ".txt"}
TREE_HINTS = ("tree", "bootstrap", "nwk", "newick", "con_", "majrule")

def detect_alignment_file(ds_path: Path) -> Path | None:
    candidates = [f for f in ds_path.iterdir() if f.is_file() and f.suffix.lower() in ALIGNMENT_EXTS]
    candidates.sort(key=lambda p: (p.suffix.lower() not in {".nex", ".nexus", ".nexorg"}, p.name.lower()))
    return candidates[0] if candidates else None

def detect_tree_file(ds_path: Path, alignment_file: Path | None) -> Path | None:
    files = [f for f in ds_path.iterdir() if f.is_file()]
    if alignment_file is not None: files = [f for f in files if f.resolve() != alignment_file.resolve()]
    scored = [(sum(3 for h in TREE_HINTS if h in f.name.lower()) + (3 if f.suffix.lower() in {".nwk",".newick",".tree"} else 0), f) for f in files]
    scored.sort(key=lambda x: (-x[0], x[1].name.lower()))
    for _, f in scored:
        try: load_tree_file(f); return f
        except: continue
    return None

def build_feature_table(datasets_root: str | Path):
    datasets_root = Path(datasets_root)
    all_frames, audit = [], []
    for ds_path in sorted(datasets_root.iterdir()):
        if not ds_path.is_dir(): continue
        dataset_id = ds_path.name
        alignment_file = detect_alignment_file(ds_path)
        tree_file = detect_tree_file(ds_path, alignment_file)
        status = {"dataset_id": dataset_id, "alignment_file": str(alignment_file), "tree_file": str(tree_file), "ok": False, "error": None}
        try:
            if not alignment_file: raise FileNotFoundError("Не найден alignment-файл")
            if not tree_file: raise FileNotFoundError("Не найден tree-файл")
            seqs = load_alignment_file(alignment_file)
            tree, _ = load_tree_file(tree_file)
            df = extract_node_features(tree, seqs, dataset_id)
            if df.empty: raise ValueError("Нет узлов с bootstrap")
            all_frames.append(df)
            status["ok"] = True; status["n_rows"] = len(df)
        except Exception as e: status["error"] = str(e)
        audit.append(status)
    if not all_frames: raise RuntimeError("Не собрано ни одной строки. Проверь audit.")
    return pd.concat(all_frames, ignore_index=True), audit

def main():
    project_root = Path.cwd()
    datasets_root = project_root / "datasets"
    output_dir = project_root / "ml_outputss"
    output_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] datasets root: {datasets_root}")
    df, audit = build_feature_table(datasets_root)
    df.to_csv(output_dir / "node_dataset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(output_dir / "dataset_audit.csv", index=False, encoding="utf-8-sig")
    
    print("\n🔹 Запуск GroupKFold тренировки...")
    model, metrics, feat_imp, pred_df = train_model(df)
    
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(model, output_dir / "model_v3_stable.pkl")
    feat_imp.to_csv(output_dir / "feature_importance_v3.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "model_metadata_v3.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    print("\n=== CV RESULTS ===")
    print(f"R²   : {metrics['cv_r2']:.4f}")
    print(f"MAE  : {metrics['cv_mae']:.4f}")
    print(f"RMSE : {metrics['cv_rmse']:.4f}")
    print("\nTop features:")
    print(feat_imp.head(10).to_string(index=False))
    print(f"\nФайл предсказаний сохранён в: {output_dir / 'test_predictions.csv'}")

if __name__ == "__main__":
    main()