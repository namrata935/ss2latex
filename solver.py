# =============================================================================
# solver.py — TextAndTableSolver class
#
# Combines all three improvements over PDF2LaTeX (2020):
#   1. EasyOCR (CRAFT + CRNN) for text/title/list — fine-tuneable on
#      academic paper data, no numpy ABI issues, CPU-friendly
#   2. TATR (Microsoft Table Transformer) for tables — trained on
#      PubTables-1M, proper row/col detection vs PDF2LaTeX's missing tables
#   3. Preprocessing pipeline applied to every crop before inference
#
# IMPROVEMENT 1 — Fine-tuning helper:
#   generate_finetune_dataset() crops text lines from academic paper images
#   and saves them as EasyOCR-compatible training data.
# IMPROVEMENT 2 — TATR:
#   TATRTableSolver uses object detection to find rows/cols, then OCRs cells.
# =============================================================================

import os
import json

import numpy as np
import easyocr
import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from config import CONFIG, TATR_MODEL
from utils import (
    latex_escape,
    strip_bullet,
    html_table_to_grid,
    grid_to_tabular,
    clean_table_grid,
    clean_latex_table,
)
from preprocess import preprocess_image


# ─────────────────────────────────────────────
# IMPROVEMENT 2 — TATR Table Solver
# ─────────────────────────────────────────────

class TATRTableSolver:
    """
    Microsoft Table Transformer — structure recognition model.
    Trained on PubTables-1M (1 million real document tables).
    Detects rows and columns as object detections, then OCRs each cell.

    This directly replaces the PIL line detector which struggled with
    borderless tables, merged cells, and low-contrast grid lines.
    """

    def __init__(self, ocr_reader: easyocr.Reader):
        device = "cuda" if CONFIG["gpu"] and torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(TATR_MODEL)
        self.model     = TableTransformerForObjectDetection.from_pretrained(
            TATR_MODEL
        ).to(device)
        self.model.eval()
        self.ocr    = ocr_reader
        self.device = device

    def parse(self, img: np.ndarray) -> dict:
        pil_img = Image.fromarray(img)
        inputs  = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=CONFIG["tatr_confidence"],
            target_sizes=torch.tensor([pil_img.size[::-1]]),
        )[0]

        rows, cols = self._extract_rows_cols(results)

        if not rows or not cols:
            # Fallback to plain OCR if TATR finds no structure
            text = self._ocr_region(img)
            return {
                "latex":      f"% Table (no grid detected)\n{latex_escape(text)}",
                "table_grid": [[text]],
                "text":       text,
            }

        grid = self._build_grid(img, rows, cols)
        col_widths = [max(0.0, float(c[2] - c[0])) for c in cols]
        grid = clean_table_grid(
            grid,
            col_widths=col_widths,
            min_col_width_ratio=CONFIG.get("table_min_col_width_ratio", 0.06),
            merge_dollar_cols=CONFIG.get("table_merge_dollar_cols", True),
        )
        return {
            "latex":      self._grid_to_tabular(grid),
            "table_grid": grid,
            "text":       " | ".join(c for row in grid for c in row),
        }

    def _extract_rows_cols(self, results) -> tuple:
        id2label = self.model.config.id2label
        rows, cols = [], []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            name = id2label[label.item()].lower()
            if "row"    in name: rows.append(box.tolist())
            elif "column" in name: cols.append(box.tolist())
        rows.sort(key=lambda b: b[1])   # top to bottom
        cols.sort(key=lambda b: b[0])   # left to right
        return rows, cols

    def _build_grid(self, img: np.ndarray,
                    rows: list, cols: list) -> list:
        h, w  = img.shape[:2]
        grid  = []
        for row_box in rows:
            y1 = max(0, int(row_box[1]))
            y2 = min(h, int(row_box[3]))
            row_cells = []
            for col_box in cols:
                x1   = max(0, int(col_box[0]))
                x2   = min(w, int(col_box[2]))
                crop = img[y1:y2, x1:x2]
                row_cells.append(
                    latex_escape(self._ocr_region(crop).strip())
                    if crop.size > 0 else ""
                )
            if any(row_cells):
                grid.append(row_cells)
        return grid or [["(empty table)"]]

    def _ocr_region(self, img: np.ndarray) -> str:
        results = self.ocr.readtext(img, detail=1, paragraph=False)
        results.sort(key=lambda r: r[0][0][1])
        return " ".join(r[1] for r in results if r[2] > CONFIG["ocr_confidence"])

    def _grid_to_tabular(self, grid: list) -> str:
        if not grid:
            return ""
        col_count = max(len(row) for row in grid)
        col_spec  = "|" + "l|" * col_count
        lines     = [f"\\begin{{tabular}}{{{col_spec}}}", "\\hline"]
        for i, row in enumerate(grid):
            padded = row + [""] * (col_count - len(row))
            lines.append(" & ".join(padded) + " \\\\")
            lines.append("\\hline")
            if i == 0 and len(grid) > 1:
                lines.append("\\hline")
        lines.append("\\end{tabular}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# SLANet (PaddleOCR PP-StructureV3) Table Solver
# ─────────────────────────────────────────────

class SLANetTableSolver:
    """
    PaddleOCR PP-StructureV3 table recognition using SLANet for structure.
    Returns HTML, converts it to a LaTeX tabular.
    """

    def __init__(self):
        try:
            from paddleocr import PPStructureV3
        except Exception as e:
            raise ImportError(
                "PaddleOCR not installed. Install paddleocr and paddlepaddle."
            ) from e

        params = {}
        lang = CONFIG.get("ppstructure_lang")
        if lang:
            params["lang"] = lang
        rec_model = CONFIG.get("ppstructure_text_recognition_model_name")
        if rec_model:
            params["text_recognition_model_name"] = rec_model

        self.engine = PPStructureV3(**params)

    def parse(self, img: np.ndarray) -> dict:
        # PaddleOCR expects BGR images
        import cv2

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        try:
            result = self.engine.predict(bgr)
        except Exception:
            # older API fallback
            result = self.engine(bgr)

        html = self._extract_html(result)
        if not html:
            return {
                "latex":      "",
                "table_grid": [],
                "text":       "",
                "html":       "",
            }

        grid = html_table_to_grid(html)
        grid = clean_table_grid(
            grid,
            col_widths=None,
            min_col_width_ratio=CONFIG.get("table_min_col_width_ratio", 0.06),
            merge_dollar_cols=CONFIG.get("table_merge_dollar_cols", True),
        )
        return {
            "latex":      grid_to_tabular(grid),
            "table_grid": grid,
            "text":       " | ".join(c for row in grid for c in row),
            "html":       html,
        }

    def _extract_html(self, result) -> str:
        if not result:
            return ""
        for item in result:
            if isinstance(item, dict):
                res = item.get("res", item.get("result", item))
                if isinstance(res, dict):
                    if res.get("html"):
                        return res["html"]
                    if res.get("structure") and isinstance(res["structure"], list):
                        return "".join(res["structure"])
                if isinstance(item.get("html"), str):
                    return item["html"]
            else:
                # object-style result
                for attr in ("res", "result", "html"):
                    val = getattr(item, attr, None)
                    if isinstance(val, dict):
                        if val.get("html"):
                            return val["html"]
                        if val.get("structure") and isinstance(val["structure"], list):
                            return "".join(val["structure"])
                    if isinstance(val, str) and val.strip().startswith("<table"):
                        return val
        return ""



# ─────────────────────────────────────────────
# IMPROVEMENT 1 — Fine-tune dataset generator
# ─────────────────────────────────────────────

def generate_finetune_dataset(pdf_image_paths: list,
                               output_dir: str = "finetune_data") -> str:
    """
    Generates EasyOCR-compatible fine-tuning data from academic paper images.

    Uses EasyOCR's own detector to find text regions, keeps only high-
    confidence detections as ground truth, and saves each crop + label.

    Args:
        pdf_image_paths : list of PNG/JPG paths (one per PDF page)
                          use convert_pdf_to_images() in main.py to generate
        output_dir      : where to save crops and annotation JSON

    Returns:
        path to annotations.json
    """
    import numpy as np
    from PIL import Image as PILImage

    os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
    reader      = easyocr.Reader(CONFIG["lang"], gpu=CONFIG["gpu"], verbose=False)
    annotations = []
    crop_idx    = 0

    for page_path in pdf_image_paths:
        img     = np.array(PILImage.open(page_path).convert("RGB"))
        results = reader.readtext(img, detail=1, paragraph=False)

        for bbox, text, conf in results:
            if conf < 0.8 or len(text.strip()) < 3:
                continue
            pts      = np.array(bbox, dtype=np.int32)
            x1, y1   = max(0, pts[:,0].min()-2), max(0, pts[:,1].min()-2)
            x2, y2   = min(img.shape[1], pts[:,0].max()+2), \
                       min(img.shape[0], pts[:,1].max()+2)
            crop     = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname    = f"crop_{crop_idx:05d}.png"
            PILImage.fromarray(crop).save(
                os.path.join(output_dir, "crops", fname)
            )
            annotations.append({
                "filename": os.path.join("crops", fname),
                "text":     text.strip(),
                "conf":     round(conf, 4),
            })
            crop_idx += 1

    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"[Finetune] {len(annotations)} crops saved to '{output_dir}'")
    print(f"[Finetune] Annotation file: {ann_path}")
    return ann_path


# ─────────────────────────────────────────────
# MAIN SOLVER CLASS
# ─────────────────────────────────────────────

class TextAndTableSolver:
    """
    Stage 2 entry point. Routes each YOLOv8 region to the correct solver:
      Title    -> EasyOCR -> \section{}
      Text     -> EasyOCR -> escaped paragraph
      List     -> EasyOCR + indent analysis -> \begin{itemize}
      Table    -> TATR + EasyOCR -> \begin{tabular}
      Equation -> passthrough (Stage 3 fills this)
    """

    def __init__(self):
        print("[Stage 2] Loading EasyOCR ...")
        self.reader = easyocr.Reader(
            CONFIG["lang"],
            gpu=CONFIG["gpu"],
            verbose=False,
        )
        self.slanet = None
        if CONFIG.get("use_slanet"):
            try:
                print("[Stage 2] Loading SLANet (PP-StructureV3) ...")
                self.slanet = SLANetTableSolver()
            except Exception as e:
                print(f"[WARN] SLANet unavailable, falling back to TATR: {e}")
                self.slanet = None

        print("[Stage 2] Loading TATR ...")
        self.tatr = TATRTableSolver(self.reader)
        print("[Stage 2] All models ready.\n")

    def solve(self, region: dict) -> dict:
        dispatch = {
            "Title":    self._solve_title,
            "Text":     self._solve_text,
            "List":     self._solve_list,
            "Table":    self._solve_table,
            "Equation": self._passthrough,
        }
        solver = dispatch.get(region.get("type", "Text"), self._solve_text)
        try:
            region["image"] = preprocess_image(region["image"])
            return solver(region)
        except Exception as e:
            print(f"  [WARN] Region {region['region_id']} ({region['type']}) failed: {e}")
            region["latex"] = ""
            region["error"] = str(e)
            return region

    def _solve_title(self, region: dict) -> dict:
        lines           = self._run_ocr(region["image"])
        title_text      = " ".join(lines).strip()
        region["text"]  = title_text
        region["latex"] = f"\\section{{{latex_escape(title_text)}}}"
        return region

    def _solve_text(self, region: dict) -> dict:
        lines = self._run_ocr_with_coords(region["image"])
        if not lines:
            region["text"] = ""
            region["latex"] = ""
            return region

        # group by line using y-gap
        lines.sort(key=lambda l: l["y"])
        heights = [l["h"] for l in lines if l.get("h")]
        med_h = float(np.median(heights)) if heights else 12.0
        gap_thresh = max(6.0, 0.5 * med_h)

        grouped = []
        cur = []
        cur_ymax = None
        for l in lines:
            y_min = l["y"]
            y_max = l["y"] + l["h"]
            if cur and cur_ymax is not None and (y_min - cur_ymax) > gap_thresh:
                grouped.append(cur)
                cur = []
            cur.append(l)
            cur_ymax = max(cur_ymax or y_max, y_max)
        if cur:
            grouped.append(cur)

        out_lines = []
        for g in grouped:
            g.sort(key=lambda l: l["x"])
            out_lines.append(" ".join(l["text"] for l in g))

        raw_text = "\n".join(out_lines)
        region["text"] = raw_text
        latex_text = latex_escape(raw_text).replace("\n", " \\\\\n")
        region["latex"] = latex_text + "\n\n"
        return region

    def _solve_list(self, region: dict) -> dict:
        raw_lines = self._run_ocr_with_coords(region["image"])
        if not raw_lines:
            region["latex"] = ""
            region["text"]  = ""
            return region

        base_x = min(line["x"] for line in raw_lines)
        xs = sorted(set(line["x"] for line in raw_lines))
        diffs = [b - a for a, b in zip(xs, xs[1:]) if (b - a) > 2]
        dyn_thresh = min(diffs) if diffs else CONFIG["indent_threshold_px"]
        threshold = max(8, dyn_thresh)

        for line in raw_lines:
            line["level"] = max(0, int((line["x"] - base_x) // threshold))
            line["text"]  = latex_escape(strip_bullet(line["text"]))

        region["latex"]      = self._build_itemize(raw_lines)
        region["text"]       = " ".join(l["text"] for l in raw_lines)
        region["list_items"] = [
            {"text": l["text"], "indent_level": l["level"]} for l in raw_lines
        ]
        return region

    def _solve_table(self, region: dict) -> dict:
        # Optional upscale for better table OCR/structure
        scale = float(CONFIG.get("table_upscale", 1.0))
        if scale > 1.0:
            from PIL import Image as PILImage
            img = region["image"]
            h, w = img.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            region["image"] = np.array(
                PILImage.fromarray(img).resize(new_size, resample=PILImage.BICUBIC)
            )

        if self.slanet:
            parsed = self.slanet.parse(region["image"])
            if parsed.get("latex"):
                region["latex"] = parsed["latex"]
                region["table_grid"] = parsed["table_grid"]
                region["text"] = parsed["text"]
                return region
        parsed               = self.tatr.parse(region["image"])
        region["latex"]      = parsed["latex"]
        region["table_grid"] = parsed["table_grid"]
        region["text"]       = parsed["text"]
        return region

    def _passthrough(self, region: dict) -> dict:
        region.setdefault("latex", "")
        return region

    def _build_itemize(self, lines: list) -> str:
        if not lines:
            return ""
        out, cur_level = [], -1
        for line in lines:
            lvl, text = line["level"], line["text"]
            if lvl > cur_level:
                for _ in range(lvl - cur_level):
                    out.append("\\begin{itemize}")
                cur_level = lvl
            elif lvl < cur_level:
                for _ in range(cur_level - lvl):
                    out.append("\\end{itemize}")
                cur_level = lvl
            out.append(f"  \\item {text}")
        for _ in range(cur_level + 1):
            out.append("\\end{itemize}")
        return "\n".join(out)

    def _run_ocr(self, img: np.ndarray) -> list:
        results = self.reader.readtext(img, detail=1, paragraph=False)
        results.sort(key=lambda r: r[0][0][1])
        return [r[1] for r in results if r[2] > CONFIG["ocr_confidence"]]

    def _run_ocr_with_coords(self, img: np.ndarray) -> list:
        results = self.reader.readtext(img, detail=1, paragraph=False)
        results.sort(key=lambda r: r[0][0][1])
        return [
            {
                "text": r[1],
                "x": r[0][0][0],
                "y": min(p[1] for p in r[0]),
                "h": max(p[1] for p in r[0]) - min(p[1] for p in r[0]),
            }
            for r in results if r[2] > CONFIG["ocr_confidence"]
        ]
