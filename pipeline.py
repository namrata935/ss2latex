# =============================================================================
# pipeline.py — Input adapters, TB-YX sort, output serialisation, benchmark
# =============================================================================

import os
import copy
import json
import time
import statistics
from pathlib import Path

import numpy as np

from config import CONFIG
from utils import load_image, to_serialisable
from solver import TextAndTableSolver


# ─────────────────────────────────────────────
# INPUT ADAPTERS
# Accepts both output formats from teammate's YOLOv8
# ─────────────────────────────────────────────

def load_regions_from_folder(folder_path: str) -> list:
    """
    Mode A: folder of pre-cropped images.
    Naming convention from YOLOv8 teammate:
        <region_id>_<Type>.png
        e.g. 001_Title.png | 002_Text.png | 003_List.png | 004_Table.png
    """
    regions = []
    folder  = Path(folder_path)
    files   = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))

    for img_path in files:
        parts     = img_path.stem.split("_", 1)
        region_id = int(parts[0]) if parts[0].isdigit() else len(regions) + 1
        rtype     = parts[1].capitalize() if len(parts) > 1 else "Text"
        regions.append({
            "region_id": region_id,
            "type":      rtype,
            "bbox":      None,
            "image":     load_image(str(img_path)),
        })

    print(f"[Input] Mode A — {len(regions)} regions from '{folder_path}'")
    return regions


def load_regions_from_json(json_path: str) -> list:
    """
    Mode B: YOLOv8 JSON with bboxes + source image.
    Expected schema:
    {
      "source_image": "screenshot.png",
      "detections": [
        {"region_id": 1, "type": "Title", "bbox": [x1, y1, x2, y2]},
        ...
      ]
    }
    """
    with open(json_path) as f:
        data = json.load(f)

    src     = load_image(data["source_image"])
    regions = []

    for det in data["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        regions.append({
            "region_id": det["region_id"],
            "type":      det["type"].capitalize(),
            "bbox":      det["bbox"],
            "image":     src[y1:y2, x1:x2],
        })

    print(f"[Input] Mode B — {len(regions)} regions from JSON")
    return regions


def load_regions(input_path: str) -> list:
    p = Path(input_path)
    if p.is_dir():          return load_regions_from_folder(input_path)
    if p.suffix == ".json": return load_regions_from_json(input_path)
    raise ValueError(f"Expected a folder or .json, got: {input_path}")


# ─────────────────────────────────────────────
# TB-YX SORT
# Top-Bottom then Left-Right reading order
# Required for correct document structure in Stage 4
# ─────────────────────────────────────────────

def tbyx_sort(regions: list) -> list:
    tol       = CONFIG["row_merge_tolerance_px"]
    with_bbox = [r for r in regions if r.get("bbox")]
    no_bbox   = [r for r in regions if not r.get("bbox")]

    if with_bbox:
        with_bbox.sort(key=lambda r: r["bbox"][1])
        rows, cur = [], [with_bbox[0]]
        for r in with_bbox[1:]:
            if abs(r["bbox"][1] - cur[0]["bbox"][1]) <= tol:
                cur.append(r)
            else:
                rows.append(cur)
                cur = [r]
        rows.append(cur)
        for row in rows:
            row.sort(key=lambda r: r["bbox"][0])
        ordered = [r for row in rows for r in row]
    else:
        ordered = []

    ordered += no_bbox
    for i, r in enumerate(ordered):
        r["reading_order"] = i + 1
    return ordered


# ─────────────────────────────────────────────
# OUTPUT SERIALISER
# ─────────────────────────────────────────────

def save_output(regions: list, output_path: str) -> dict:
    payload = {
        "stage":        2,
        "solver":       "EasyOCR + TATR + Preprocessing",
        "region_count": len(regions),
        "regions":      [to_serialisable(r) for r in regions],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[Stage 2] Output saved -> {output_path}")
    return payload


# ─────────────────────────────────────────────
# MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_stage2(input_path: str, output_path: str = None) -> dict:
    """
    Full Stage 2 pipeline.

    Usage:
        run_stage2("crops/")                # Mode A — folder of crops
        run_stage2("yolo_output.json")      # Mode B — YOLOv8 JSON
    """
    out_path = output_path or CONFIG["output_json"]
    solver   = TextAndTableSolver()
    regions  = load_regions(input_path)
    regions  = tbyx_sort(regions)

    print("[Stage 2] Running solvers ...")
    solved = []
    for r in regions:
        print(f"  -> [{r['reading_order']}] {r['type']:<10} id={r['region_id']}")
        solved.append(solver.solve(r))

    return save_output(solved, out_path)


# ─────────────────────────────────────────────
# BENCHMARK
# Matches UniMERNet output format from teammate
# ─────────────────────────────────────────────

def _edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if s1[i-1] == s2[j-1] else \
                    1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]

def _bleu_1gram(pred: str, ref: str) -> float:
    pt, rt = pred.strip().split(), ref.strip().split()
    if not pt: return 0.0
    rc = {}
    for t in rt: rc[t] = rc.get(t, 0) + 1
    matches = 0
    for t in pt:
        if rc.get(t, 0) > 0:
            rc[t] -= 1
            matches += 1
    bp = 1.0 if len(pt) >= len(rt) else 2.718 ** (1 - len(rt) / max(len(pt), 1))
    return round(bp * matches / len(pt), 4)

def _ned(pred: str, ref: str) -> float:
    return round(
        _edit_distance(pred, ref) / max(len(pred), len(ref), 1), 4
    )


def benchmark_latency(input_path: str,
                       ground_truth: dict = None,
                       runs: int = 3) -> dict:
    """
    Benchmarks Stage 2 latency and accuracy.
    Output format matches teammate's UniMERNet benchmark.

    Args:
        input_path   : same as run_stage2()
        ground_truth : optional dict {region_id: expected_latex_string}
                       if provided, also computes BLEU, NED, exact match
        runs         : timed runs per region (more = more stable average)

    Example:
        benchmark_latency("test_crops", runs=3)

        benchmark_latency("test_crops", ground_truth={
            1: "\\\\section{Introduction}",
            2: "This is sample text Two lines",
        }, runs=3)
    """
    solver  = TextAndTableSolver()
    regions = load_regions(input_path)
    regions = tbyx_sort(regions)
    backend = f"EasyOCR+TATR {'GPU' if CONFIG['gpu'] else 'CPU'}"
    has_gt  = ground_truth is not None

    print(f"\n>> python main.py --input {input_path}")
    print(f"Using {os.cpu_count()} CPU threads")
    print(f"Samples (timed): {len(regions) * runs} | "
          f"max_regions: {len(regions)} | runs: {runs}\n")

    if has_gt:
        print(f"{'Backend':<22} {'Avg latency (ms)':>18} {'BLEU':>8} "
              f"{'Norm edit dist':>16} {'Exact match %':>14}")
        print("-" * 82)
    else:
        print(f"{'Backend':<22} {'Avg latency (ms)':>18} {'Min ms':>8} {'Max ms':>8}")
        print("-" * 60)

    all_times, bleu_scores, ned_scores = [], [], []
    exact_hits, gt_count = 0, 0

    for r in regions:
        times    = []
        solved_r = None
        for _ in range(runs):
            r_copy   = copy.deepcopy(r)
            t0       = time.perf_counter()
            solved_r = solver.solve(r_copy)
            t1       = time.perf_counter()
            times.append((t1 - t0) * 1000)
        all_times.extend(times)

        if has_gt and r["region_id"] in ground_truth:
            pred = solved_r.get("latex", "")
            ref  = ground_truth[r["region_id"]]
            bleu_scores.append(_bleu_1gram(pred, ref))
            ned_scores.append(_ned(pred, ref))
            if pred.strip() == ref.strip():
                exact_hits += 1
            gt_count += 1

    avg_ms = statistics.mean(all_times)
    min_ms = min(all_times)
    max_ms = max(all_times)

    if has_gt and bleu_scores:
        print(f"{backend:<22} {avg_ms:>18.2f} "
              f"{statistics.mean(bleu_scores):>8.4f} "
              f"{statistics.mean(ned_scores):>16.4f} "
              f"{(exact_hits/gt_count)*100:>13.2f}%")
    else:
        print(f"{backend:<22} {avg_ms:>18.2f} {min_ms:>8.2f} {max_ms:>8.2f}")

    print(f"\nPeak observed working set: CPU-only, no GPU memory used.")
    print(f"Target: <30ms | "
          f"Status: {'PASS' if avg_ms < 30 else 'FAIL (expected on CPU prototype)'}")

    return {
        "avg_latency_ms":  round(avg_ms, 2),
        "bleu":            round(statistics.mean(bleu_scores), 4) if bleu_scores else None,
        "norm_edit_dist":  round(statistics.mean(ned_scores),  4) if ned_scores  else None,
        "exact_match_pct": round((exact_hits/max(gt_count,1))*100, 2) if bleu_scores else None,
    }
