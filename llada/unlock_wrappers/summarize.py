"""Read speed_metrics.jsonl + lm_eval results_*.json across runs and
produce a compact (tag, accuracy, avg_nfe, tokens_per_nfe, wallclock) table.

Used by the launchers to make the NFE-reduction story legible at a glance
instead of hunting through a dozen JSON files.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln for ln in (line.strip() for line in fh) if ln]
    for ln in reversed(lines):
        try:
            return json.loads(ln)
        except json.JSONDecodeError:
            continue
    return None


def _latest_results_json(eval_output_path: Path) -> Optional[Path]:
    matches = sorted(eval_output_path.glob("*/results_*.json"))
    return matches[-1] if matches else None


def summarize_run(
    results_root: Path,
    run_tag: str,
    method_id: Optional[str] = None,
    task: str = "gsm8k",
    accuracy_metric: str = "flexible-extract",
) -> Dict[str, Any]:
    """Collect (accuracy, avg_nfe, tokens_per_nfe, avg_selected, tokens_per_sec,
    total_wallclock_sec) for one run. Best-effort — fields missing from the
    logs come back as None rather than raising."""
    out: Dict[str, Any] = {
        "tag": run_tag,
        "accuracy": None,
        "accuracy_stderr": None,
        "avg_nfe": None,
        "tokens_per_nfe": None,
        "avg_selected_per_step": None,
        "avg_candidate_pool": None,
        "tokens_per_sec": None,
        "total_wallclock_sec": None,
    }

    # Each run_tag lives under results_root/<run_tag>/<method_id>/
    run_dir = results_root / run_tag
    if not run_dir.exists():
        return out
    method_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
    if method_id is not None:
        method_dirs = [p for p in method_dirs if p.name == method_id]
    if not method_dirs:
        return out
    method_dir = method_dirs[0]

    # speed_metrics.jsonl — last line is most recent.
    speed = _last_jsonl(method_dir / "speed_metrics.jsonl")
    if speed is not None:
        out["avg_nfe"] = speed.get("Average NFE")
        out["tokens_per_nfe"] = speed.get("Tokens per NFE")
        out["avg_selected_per_step"] = speed.get("UNLOCK Avg Selected Count")
        out["avg_candidate_pool"] = speed.get("UNLOCK Avg Candidate Pool Size")
        out["tokens_per_sec"] = speed.get("Tokens per Second")
        out["total_wallclock_sec"] = speed.get("Total Time Taken")

    # Accuracy from lm_eval results JSON.
    eval_output_path = method_dir / "lm_eval_output"
    results_json = _latest_results_json(eval_output_path)
    if results_json is not None:
        payload = json.loads(results_json.read_text(encoding="utf-8"))
        task_metrics = payload.get("results", {}).get(task, {})
        out["accuracy"] = task_metrics.get(f"exact_match,{accuracy_metric}")
        out["accuracy_stderr"] = task_metrics.get(f"exact_match_stderr,{accuracy_metric}")

    return out


def render_table(rows: List[Dict[str, Any]], baseline_tag: Optional[str] = None) -> str:
    """Render a fixed-width summary. If baseline_tag is given and present in rows,
    append per-row Avg NFE ratio vs baseline (lower is better)."""
    def _fmt(v: Any, spec: str = ".4f") -> str:
        if v is None:
            return " -"
        try:
            return format(float(v), spec)
        except (TypeError, ValueError):
            return str(v)

    baseline_nfe: Optional[float] = None
    if baseline_tag is not None:
        for r in rows:
            if r["tag"] == baseline_tag and r.get("avg_nfe") is not None:
                baseline_nfe = float(r["avg_nfe"])
                break

    header = (
        f"{'tag':28s}  {'acc':>7s}  {'avg_nfe':>9s}  "
        f"{'tok/nfe':>8s}  {'sel/step':>9s}  {'tok/s':>8s}  {'wall(s)':>8s}"
    )
    if baseline_nfe is not None:
        header += f"  {'nfe/base':>9s}"

    lines = [header, "-" * len(header)]
    for r in rows:
        line = (
            f"{str(r['tag'])[:28]:28s}  "
            f"{_fmt(r.get('accuracy'), '.3f'):>7s}  "
            f"{_fmt(r.get('avg_nfe'), '.2f'):>9s}  "
            f"{_fmt(r.get('tokens_per_nfe'), '.2f'):>8s}  "
            f"{_fmt(r.get('avg_selected_per_step'), '.2f'):>9s}  "
            f"{_fmt(r.get('tokens_per_sec'), '.1f'):>8s}  "
            f"{_fmt(r.get('total_wallclock_sec'), '.1f'):>8s}"
        )
        if baseline_nfe is not None:
            nfe = r.get("avg_nfe")
            ratio = float(nfe) / baseline_nfe if nfe is not None and baseline_nfe > 0 else None
            line += f"  {_fmt(ratio, '.3f'):>9s}"
        lines.append(line)
    return "\n".join(lines)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "tag", "accuracy", "accuracy_stderr",
        "avg_nfe", "tokens_per_nfe",
        "avg_selected_per_step", "avg_candidate_pool",
        "tokens_per_sec", "total_wallclock_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def summarize_many(
    results_root: Path,
    run_tags: List[str],
    task: str = "gsm8k",
    accuracy_metric: str = "flexible-extract",
) -> List[Dict[str, Any]]:
    return [summarize_run(results_root, t, task=task, accuracy_metric=accuracy_metric) for t in run_tags]


if __name__ == "__main__":
    # Ad-hoc CLI: python -m unlock_wrappers.summarize <results_root> <tag> [<tag> ...]
    if len(sys.argv) < 3:
        sys.stderr.write(
            "usage: python -m unlock_wrappers.summarize <results_root> <tag> [<tag> ...]\n"
        )
        raise SystemExit(2)
    root = Path(sys.argv[1]).expanduser().resolve()
    tags = sys.argv[2:]
    rows = summarize_many(root, tags)
    print(render_table(rows, baseline_tag=tags[0]))
