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
    args = parser.parse_args()

    # Fine-tune dataset generation
    if args.finetune:
        pages = convert_pdf_to_images(args.finetune)
        generate_finetune_dataset(pages, output_dir="finetune_data")
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
