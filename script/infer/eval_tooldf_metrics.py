"""
eval_tooldf_metrics.py
======================

Compute ACC / Balanced ACC / F1 (macro) on a Tool-SFT/GRPO eval jsonl produced
by `evaluate_tooldf.py`.

Each line carries pre-parsed labels under `pred.answer` / `gt.answer`
("real" | "fake" | null). We trust those over re-parsing `response` / `labels`,
but fall back to the text fields if the structured ones are missing.

Null handling:
  - default     : drop (skip) — match eval_metrics.py behavior
  - --strict    : count null preds as wrong (mapped to the opposite of GT)

Usage:
    python script/eval_tooldf_metrics.py result/tooldf/tooldf_sft_c3.jsonl
    python script/eval_tooldf_metrics.py result/tooldf/tooldf_sft_*.jsonl
    python script/eval_tooldf_metrics.py result/tooldf/ --strict
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

LABEL2ID = {"real": 0, "fake": 1}


def _norm(label):
    if label is None:
        return None
    s = str(label).strip().lower()
    return s if s in LABEL2ID else None


def _from_text(text):
    """Fallback parser when pred/gt structured fields are absent."""
    if not text:
        return None
    t = text.strip().lower()
    if "<answer>" in t and "</answer>" in t:
        s = t.find("<answer>") + len("<answer>")
        e = t.find("</answer>")
        c = t[s:e].strip()
        if c in LABEL2ID:
            return c
    if t in LABEL2ID:
        return t
    c = t.replace(" ", "")
    if c in LABEL2ID:
        return c
    for k in LABEL2ID:
        if k in t.split():
            return k
    return None


def extract_pair(d):
    """Return (gt_label, pred_label) — either may be None."""
    gt = _norm((d.get("gt") or {}).get("answer")) or _from_text(d.get("labels", ""))
    pr = _norm((d.get("pred") or {}).get("answer")) or _from_text(d.get("response", ""))
    return gt, pr


def load_results(path, strict=False):
    y_true, y_pred = [], []
    skipped_no_gt = 0
    skipped_null_pred = 0
    null_pred_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                skipped_no_gt += 1
                continue

            gt, pr = extract_pair(d)

            if gt not in LABEL2ID:
                skipped_no_gt += 1
                continue

            if pr not in LABEL2ID:
                null_pred_count += 1
                if strict:
                    # Worst-case: count as the opposite class.
                    pr = "fake" if gt == "real" else "real"
                else:
                    skipped_null_pred += 1
                    continue

            y_true.append(LABEL2ID[gt])
            y_pred.append(LABEL2ID[pr])

    return (
        np.array(y_true),
        np.array(y_pred),
        {
            "skipped_no_gt": skipped_no_gt,
            "skipped_null_pred": skipped_null_pred,
            "null_pred_count": null_pred_count,
        },
    )


def evaluate(path, strict=False):
    y_true, y_pred, info = load_results(path, strict=strict)
    n = len(y_true)
    if n == 0:
        print(f"[{path}] no valid samples (info={info})")
        return None

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cls = Counter(y_true.tolist())

    tag = os.path.join(
        os.path.basename(os.path.dirname(path)), os.path.basename(path)
    )
    mode = "strict (nulls=wrong)" if strict else "loose (nulls=skipped)"
    print(f"\n=== {tag}  [{mode}] ===")
    print(f"  N={n}  class_dist={{real:{cls[0]}, fake:{cls[1]}}}")
    print(f"  null_preds={info['null_pred_count']}  "
          f"skipped(no_gt)={info['skipped_no_gt']}  "
          f"skipped(null_pred)={info['skipped_null_pred']}")
    print(f"  ACC       : {acc:.4f}")
    print(f"  BACC      : {bacc:.4f}")
    print(f"  F1 macro  : {f1m:.4f}")
    print(f"  Confusion (rows=true [real,fake], cols=pred):")
    print(f"    [[{cm[0,0]:>6d} {cm[0,1]:>6d}]")
    print(f"     [{cm[1,0]:>6d} {cm[1,1]:>6d}]]")

    return {
        "file": path,
        "n": n,
        "acc": acc,
        "bacc": bacc,
        "f1_macro": f1m,
        "null_preds": info["null_pred_count"],
    }


def expand_paths(paths):
    files = []
    for q in paths:
        if any(c in q for c in "*?["):
            files.extend(sorted(glob.glob(q)))
        elif os.path.isdir(q):
            files.extend(sorted(glob.glob(os.path.join(q, "**", "*.jsonl"), recursive=True)))
        else:
            files.append(q)
    # Drop per-shard intermediates from a glob over a directory — they are
    # already represented in the merged file.
    files = [f for f in files if ".shard" not in os.path.basename(f)]
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+",
                   help="tooldf eval jsonl file(s), glob(s), or directories")
    p.add_argument("--strict", action="store_true",
                   help="Count null predictions as wrong (default: skip).")
    args = p.parse_args()

    files = expand_paths(args.paths)
    if not files:
        print("no files matched")
        return

    rows = []
    for f in files:
        r = evaluate(f, strict=args.strict)
        if r:
            rows.append(r)

    if len(rows) > 1:
        print("\n=== summary ===")
        hdr = f"{'file':<60s}  {'N':>6s}  {'null':>5s}  {'ACC':>7s}  {'BACC':>7s}  {'F1m':>7s}"
        print(hdr)
        for r in rows:
            tag = "/".join(r["file"].split("/")[-2:])
            print(f"{tag:<60s}  {r['n']:>6d}  {r['null_preds']:>5d}  "
                  f"{r['acc']:>7.4f}  {r['bacc']:>7.4f}  {r['f1_macro']:>7.4f}")


if __name__ == "__main__":
    main()
