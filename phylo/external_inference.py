from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Bio import AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree


class ExternalToolError(RuntimeError):
    pass


@dataclass
class ExternalInferenceResult:
    tree: Tree
    log_likelihood: float | None = None
    support_label: str | None = None
    support_kind: str | None = None
    metadata: dict | None = None


_IQTREE_CANDIDATES = (
    os.environ.get("PHYLO_IQTREE_BIN"),
    "iqtree3",
    "iqtree2",
    "iqtree",
)
_MRBAYES_CANDIDATES = (
    os.environ.get("PHYLO_MRBAYES_BIN"),
    "mb",
    "mrbayes",
)


def _resolve_binary(candidates: Iterable[str | None], tool_name: str) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return str(Path(candidate).resolve())
    raise ExternalToolError(
        f"Не найден исполняемый файл {tool_name}. Установите его и добавьте в PATH "
        f"или задайте переменную окружения для пути к бинарнику."
    )


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        raise ExternalToolError(details) from exc


def _sanitize_alignment_ids(alignment: MultipleSeqAlignment) -> MultipleSeqAlignment:
    safe = MultipleSeqAlignment([])
    used: set[str] = set()
    for idx, record in enumerate(alignment, start=1):
        copied = record[:]
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.id or f"taxon_{idx}")
        if not base:
            base = f"taxon_{idx}"
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        copied.id = candidate
        copied.name = candidate
        copied.description = candidate
        safe.append(copied)
        used.add(candidate)
    return safe


def _write_alignment_fasta(alignment: MultipleSeqAlignment, path: Path) -> None:
    AlignIO.write(_sanitize_alignment_ids(alignment), str(path), "fasta")


def _write_alignment_nexus(alignment: MultipleSeqAlignment, path: Path) -> None:
    safe = _sanitize_alignment_ids(alignment)
    nchar = safe.get_alignment_length() if len(safe) else 0
    lines = [
        "#NEXUS",
        "begin data;",
        f"dimensions ntax={len(safe)} nchar={nchar};",
        "format datatype=dna missing=? gap=- interleave=no;",
        "matrix",
    ]
    for record in safe:
        lines.append(f"{record.id} {str(record.seq)}")
    lines.extend([";", "end;"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_tree_any(path: Path) -> Tree:
    parse_order = ["newick", "nexus"]
    if path.suffix.lower() in {".nex", ".nexus", ".tre", ".con", ".tree"}:
        parse_order = ["nexus", "newick"]
    last_exc: Exception | None = None
    for fmt in parse_order:
        try:
            tree = Phylo.read(str(path), fmt)
            for clade in tree.find_clades():
                if clade.branch_length is None or clade.branch_length < 0:
                    clade.branch_length = max(float(clade.branch_length or 0.0), 0.0)
            return tree
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise ExternalToolError(f"Не удалось прочитать дерево из {path.name}: {last_exc}")


def _extract_log_likelihood(text: str) -> float | None:
    patterns = [
        r"Log-likelihood of the tree:\s*(-?\d+(?:\.\d+)?)",
        r"BEST SCORE FOUND\s*:\s*(-?\d+(?:\.\d+)?)",
        r"lnL\s*=\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def run_iqtree_ml(alignment: MultipleSeqAlignment) -> ExternalInferenceResult:
    binary = _resolve_binary(_IQTREE_CANDIDATES, "IQ-TREE")
    with tempfile.TemporaryDirectory(prefix="phylo_iqtree_") as tmpdir:
        workdir = Path(tmpdir)
        aln_path = workdir / "alignment.fasta"
        prefix = workdir / "iqtree_run"
        _write_alignment_fasta(alignment, aln_path)
        cmd = [
            binary,
            "-s",
            str(aln_path),
            "-m",
            "MFP",
            "-B",
            "1000",
            "-T",
            "AUTO",
            "--prefix",
            str(prefix),
            "--redo",
        ]
        result = _run_command(cmd, cwd=workdir)
        tree_path = prefix.with_suffix(".contree")
        if not tree_path.exists():
            tree_path = prefix.with_suffix(".treefile")
        if not tree_path.exists():
            raise ExternalToolError("IQ-TREE завершился, но не создал итоговое дерево (.contree/.treefile).")
        report_path = prefix.with_suffix(".iqtree")
        report_text = report_path.read_text(encoding="utf-8", errors="ignore") if report_path.exists() else result.stdout
        tree = _read_tree_any(tree_path)
        tree.rooted = False
        return ExternalInferenceResult(
            tree=tree,
            log_likelihood=_extract_log_likelihood(report_text),
            support_label="Ultrafast bootstrap",
            support_kind="ml_ultrafast_bootstrap",
            metadata={
                "engine": "IQ-TREE",
                "engine_binary": Path(binary).name,
                "model_selection": "MFP",
                "bootstrap_replicates": 1000,
            },
        )


_MRBAYES_BLOCK = """
begin mrbayes;
    set autoclose=yes nowarn=yes;
    lset nst=6 rates=gamma;
    mcmcp ngen=50000 samplefreq=100 printfreq=100 diagnfreq=1000 nchains=4 nruns=1 burninfrac=0.25 temp=0.2 stoprule=yes stopval=0.01;
    mcmc;
    sump;
    sumt;
end;
""".strip()


def run_mrbayes(alignment: MultipleSeqAlignment) -> ExternalInferenceResult:
    binary = _resolve_binary(_MRBAYES_CANDIDATES, "MrBayes")
    with tempfile.TemporaryDirectory(prefix="phylo_mrbayes_") as tmpdir:
        workdir = Path(tmpdir)
        nexus_path = workdir / "alignment.nex"
        _write_alignment_nexus(alignment, nexus_path)
        with nexus_path.open("a", encoding="utf-8") as fh:
            fh.write("\n\n" + _MRBAYES_BLOCK + "\n")
        cmd = [binary, str(nexus_path)]
        result = _run_command(cmd, cwd=workdir)

        candidates = [
            workdir / "alignment.nex.con.tre",
            workdir / "alignment.nex.con",
            workdir / "alignment.con.tre",
            workdir / "alignment.con",
        ]
        tree_path = next((path for path in candidates if path.exists()), None)
        if tree_path is None:
            tre_files = sorted(workdir.glob("*.con*"))
            tree_path = tre_files[0] if tre_files else None
        if tree_path is None:
            raise ExternalToolError("MrBayes завершился, но не создал consensus tree (.con/.con.tre).")

        tree = _read_tree_any(tree_path)
        tree.rooted = True
        output_text = result.stdout + "\n" + result.stderr
        return ExternalInferenceResult(
            tree=tree,
            log_likelihood=_extract_log_likelihood(output_text),
            support_label="Posterior probability",
            support_kind="bayesian_posterior",
            metadata={
                "engine": "MrBayes",
                "engine_binary": Path(binary).name,
                "ngen": 50000,
                "samplefreq": 100,
                "burninfrac": 0.25,
            },
        )
