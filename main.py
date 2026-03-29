# =============================================================================
# main.py — Entry point for Stage 2: Text + Table Solver
#
# Screenshot-to-LaTeX Pipeline — Stage 2
# Lightweight alternative to PDF2LaTeX (2020) baseline
#
# Models used:
#   OCR    : EasyOCR (CRAFT detector + CRNN ResNet recogniser)
#   Tables : Microsoft Table Transformer (TATR) — PubTables-1M pretrained
#   Preprocess: Deskew (Hough) + Denoise (NLM) + Contrast (PIL)
#
# Usage:
#   python main.py                          # runs quick synthetic test
#   python main.py --input crops/           # Mode A: folder of crops
#   python main.py --input yolo.json        # Mode B: YOLOv8 JSON
#   python main.py --benchmark              # latency benchmark
#   python main.py --finetune paper.pdf     # generate fine-tune data
# =============================================================================

import os
import argparse
from PIL import Image, ImageDraw

from pipeline import run_stage2, benchmark_latency
from solver import generate_finetune_dataset


# ─────────────────────────────────────────────
# PDF → images helper (for fine-tuning)
# ─────────────────────────────────────────────

def convert_pdf_to_images(pdf_path: str,
                           output_dir: str = "pdf_pages",
                           dpi: int = 250) -> list:
    """
    Converts a PDF to page images at 250 DPI.
    Same resolution used by PDF2LaTeX (2020) for fair comparison.
    Requires: pip install pdf2image  +  brew install poppler (Mac)
    """
    from pdf2image import convert_from_path
    os.makedirs(output_dir, exist_ok=True)
    pages      = convert_from_path(pdf_path, dpi=dpi)
    page_paths = []
    for i, page in enumerate(pages):
        path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        page.save(path, "PNG")
        page_paths.append(path)
    print(f"[PDF] Converted {len(pages)} pages -> '{output_dir}/'")
    return page_paths


# ─────────────────────────────────────────────
# Tab2LaTeX (HF) helper
# ─────────────────────────────────────────────

def export_tab2latex(split: str = "test",
                     limit: int = 200,
                     output_dir: str = "latte_data") -> tuple:
    """
    Downloads a subset of the Tab2LaTeX dataset from Hugging Face,
    exports images to a crops folder, and writes ground-truth LaTeX.

    Returns:
        (crops_dir, gt_json_path)
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    ds = load_dataset("lt-asset/tab2latex", split=split)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    gt = {}
    for i, ex in enumerate(ds):
        region_id = i + 1
        img = ex["image"]
        latex = ex.get("latex", "")
        fname = f"{region_id:05d}_Table.png"
        img.save(os.path.join(crops_dir, fname))
        gt[region_id] = latex

    gt_path = os.path.join(output_dir, "latte_gt.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        import json
        json.dump(gt, f, indent=2, ensure_ascii=False)

    print(f"[Tab2LaTeX] Exported {len(gt)} samples to '{crops_dir}/'")
    print(f"[Tab2LaTeX] Ground truth -> {gt_path}")
    return crops_dir, gt_path


def _load_gt_json(path: str) -> dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


# ─────────────────────────────────────────────
# Synthetic test
# ─────────────────────────────────────────────

def quick_test() -> dict:
    """
    Creates four synthetic crop images using Pillow and runs
    the full Stage 2 pipeline. Use this to verify the pipeline
    works before connecting to real YOLOv8 input.
    """
    test_dir = "test_crops"
    os.makedirs(test_dir, exist_ok=True)

    def _make_img(lines, fname, w=400, h=80):
        img  = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        y = 10
        for line in lines:
            draw.text((10, y), line, fill=(0, 0, 0))
            y += 25
        img.save(os.path.join(test_dir, fname))

    _make_img(["Introduction"],                       "001_Title.png", h=50)
    _make_img(["This is sample text.", "Two lines."], "002_Text.png",  h=80)
    _make_img(["* Item one", "    * Nested item",
               "* Item two"],                         "003_List.png",  h=100)

    tbl  = Image.new("RGB", (300, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(tbl)
    draw.rectangle([10, 10, 290, 110], outline=(0, 0, 0))
    draw.line([(150, 10), (150, 110)], fill=(0, 0, 0))
    draw.line([(10, 60),  (290, 60)],  fill=(0, 0, 0))
    for text, pos in [("Header A", (20, 30)), ("Header B", (160, 30)),
                      ("Cell 1",   (20, 75)), ("Cell 2",   (160, 75))]:
        draw.text(pos, text, fill=(0, 0, 0))
    tbl.save(os.path.join(test_dir, "004_Table.png"))

    print(f"[Test] Synthetic crops written to '{test_dir}'\n")
    result = run_stage2(test_dir, "stage2_test_output.json")

    print("\n── LaTeX Preview ──────────────────────────")
    for r in result["regions"]:
        print(f"\n[{r['reading_order']}] {r['type']}")
        print(r.get("latex", "(empty)"))
    return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Text + Table Solver for Screenshot-to-LaTeX"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to crop folder (Mode A) or YOLOv8 JSON (Mode B)"
    )
    parser.add_argument(
        "--output", type=str, default="stage2_output.json",
        help="Output JSON path (default: stage2_output.json)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark on --input"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of benchmark runs per region (default: 3)"
    )
    parser.add_argument(
        "--finetune", type=str, default=None,
        help="Path to PDF — converts to images and generates fine-tune dataset"
    )
    parser.add_argument(
        "--latte", action="store_true",
        help="Export Tab2LaTeX (HF) to latte_data/ and optionally benchmark"
    )
    parser.add_argument(
        "--latte-split", type=str, default="test",
        help="Tab2LaTeX split to export (default: test)"
    )
    parser.add_argument(
        "--latte-limit", type=int, default=200,
        help="Max samples to export (default: 200)"
    )
    parser.add_argument(
        "--latte-output", type=str, default="latte_data",
        help="Output folder for Tab2LaTeX export (default: latte_data)"
    )
    parser.add_argument(
        "--latte-benchmark", action="store_true",
        help="Run benchmark after exporting Tab2LaTeX samples"
    )
    args = parser.parse_args()

    # Fine-tune dataset generation
    if args.finetune:
        pages = convert_pdf_to_images(args.finetune)
        generate_finetune_dataset(pages, output_dir="finetune_data")
        return

    # Tab2LaTeX export
    if args.latte:
        crops_dir, gt_path = export_tab2latex(
            split=args.latte_split,
            limit=args.latte_limit,
            output_dir=args.latte_output,
        )
        if args.latte_benchmark:
            gt = _load_gt_json(gt_path)
            benchmark_latency(crops_dir, ground_truth=gt, runs=args.runs)
        return

    # Benchmark
    if args.benchmark:
        if not args.input:
            print("[ERROR] --benchmark requires --input")
            return
        benchmark_latency(args.input, runs=args.runs)
        return

    # Normal run or quick test
    if args.input:
        run_stage2(args.input, args.output)
    else:
        quick_test()


if __name__ == "__main__":
    main()
