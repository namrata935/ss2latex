# =============================================================================
# solver.py — TextAndTableSolver class
#
# Combines all three improvements over PDF2LaTeX (2020):
#   1. PaddleOCR v4 (default) or EasyOCR for text/title/list
#      — PaddleOCR en_PP-OCRv4_server_rec is significantly more accurate
#        for academic text than EasyOCR's CRNN
#   2. TATR (Microsoft Table Transformer) for tables — trained on
#      PubTables-1M, proper row/col detection vs PDF2LaTeX's missing tables
#   3. Preprocessing pipeline applied to every crop before inference
#
# OCR backend is controlled by CONFIG["ocr_backend"]:
#   "paddle"  — PaddleOCR v4 (recommended, more accurate)
#   "easyocr" — EasyOCR fallback (no paddleocr dependency)
# =============================================================================

import os
import json

import numpy as np
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
# OCR backend abstraction
# ─────────────────────────────────────────────

class _PaddleOCRBackend:
    """PaddleOCR v4 text recognition — more accurate than EasyOCR on academic text."""

    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,  # MPS/Apple Silicon not supported; set True only for CUDA
            rec_model_name=CONFIG.get("ppstructure_text_recognition_model_name",
                                      "en_PP-OCRv4_server_rec"),
            show_log=False,
        )

    def readtext(self, img: np.ndarray) -> list:
        """Returns list of (bbox, text, confidence) matching EasyOCR format."""
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = self._ocr.ocr(bgr, cls=True)
        out = []
        if not result or result[0] is None:
            return out
        for line in result[0]:
            if line is None:
                continue
            bbox, (text, conf) = line[0], line[1]
            out.append((bbox, text, conf))
        return out


class _EasyOCRBackend:
    """EasyOCR fallback backend."""

    def __init__(self):
        import easyocr
        self._reader = easyocr.Reader(
            CONFIG["lang"], gpu=CONFIG["gpu"], verbose=False
        )

    def readtext(self, img: np.ndarray) -> list:
        return self._reader.readtext(img, detail=1, paragraph=False)


def _build_ocr_backend():
    backend = CONFIG.get("ocr_backend", "easyocr")
    if backend == "paddle":
        try:
            b = _PaddleOCRBackend()
            print("[Stage 2] OCR backend: PaddleOCR v4")
            return b
        except Exception as e:
            print(f"[WARN] PaddleOCR unavailable ({e}), falling back to EasyOCR")
    import easyocr  # noqa: F401 — ensure importable
    b = _EasyOCRBackend()
    print("[Stage 2] OCR backend: EasyOCR")
    return b


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _nms_boxes(boxes: list, iou_thresh: float = 0.3) -> list:
    """
    Non-maximum suppression on column boxes [x1, y1, x2, y2].
    When two columns overlap heavily, keep the narrower one (more specific).
    Returns boxes sorted left-to-right.
    """
    if not boxes:
        return boxes
    # sort by width ascending so narrower (more specific) boxes survive
    indexed = sorted(enumerate(boxes), key=lambda ib: ib[1][2] - ib[1][0])
    suppressed = set()
    keep_idx = []
    for i, (orig_i, a) in enumerate(indexed):
        if orig_i in suppressed:
            continue
        keep_idx.append(orig_i)
        ax1, ax2 = a[0], a[2]
        a_w = ax2 - ax1 or 1e-6
        for j, (orig_j, b) in enumerate(indexed):
            if j <= i or orig_j in suppressed:
                continue
            bx1, bx2 = b[0], b[2]
            inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            b_w = bx2 - bx1 or 1e-6
            iou = inter / min(a_w, b_w)
            if iou > iou_thresh:
                suppressed.add(orig_j)
    # return in original left-to-right order
    result = [boxes[i] for i in keep_idx]
    result.sort(key=lambda b: b[0])
    return result


# ─────────────────────────────────────────────
# IMPROVEMENT 2 — TATR Table Solver
# ─────────────────────────────────────────────

class TATRTableSolver:
    """
    Microsoft Table Transformer — structure recognition model.
    Trained on PubTables-1M (1 million real document tables).
    Detects rows and columns as object detections, then OCRs each cell.
    """

    def __init__(self, ocr_backend):
        device = "cuda" if CONFIG["gpu"] and torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(TATR_MODEL)
        self.model     = TableTransformerForObjectDetection.from_pretrained(
            TATR_MODEL
        ).to(device)
        self.model.eval()
        self.ocr    = ocr_backend
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
            text = self._ocr_full(img)
            return {
                "latex":      f"% Table (no grid detected)\n{latex_escape(text)}",
                "table_grid": [[text]],
                "text":       text,
            }

        # Run OCR once on the full image, then assign tokens to cells.
        # Keep a raw (non-preprocessed) copy for cell-crop fallback — preprocessing
        # can destroy faint single digits like "0" in small cells.
        tokens = self._ocr_tokens(img)
        grid = self._assign_tokens_to_grid(tokens, rows, cols, img.shape, img=img)

        col_widths = [max(0.0, float(c[2] - c[0])) for c in cols]
        grid = clean_table_grid(
            grid,
            col_widths=col_widths,
            min_col_width_ratio=CONFIG.get("table_min_col_width_ratio", 0.04),
            merge_dollar_cols=CONFIG.get("table_merge_dollar_cols", True),
            merge_currency_prefix=CONFIG.get("table_merge_currency_prefix", True),
        )
        # Escape after merge so currency symbols are visible during cleanup
        grid = [[latex_escape(cell) for cell in row] for row in grid]
        return {
            "latex":      self._grid_to_tabular(grid),
            "table_grid": grid,
            "text":       " | ".join(c for row in grid for c in row),
        }

        col_widths = [max(0.0, float(c[2] - c[0])) for c in cols]
        grid = clean_table_grid(
            grid,
            col_widths=col_widths,
            min_col_width_ratio=CONFIG.get("table_min_col_width_ratio", 0.04),
            merge_dollar_cols=CONFIG.get("table_merge_dollar_cols", True),
            merge_currency_prefix=CONFIG.get("table_merge_currency_prefix", True),
        )
        # Escape after merge so currency symbols are visible during cleanup
        grid = [[latex_escape(cell) for cell in row] for row in grid]
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
            # Skip spanning/projected headers — they cover the full row width
            # and cause duplicate content when used as column bboxes
            if "projected" in name or "spanning" in name:
                continue
            if "row" in name:
                rows.append(box.tolist())
            elif "column" in name:
                cols.append(box.tolist())

        rows.sort(key=lambda b: b[1])   # top to bottom
        cols.sort(key=lambda b: b[0])   # left to right

        # Remove columns that are too wide (spanning headers misdetected as cols)
        # A real column shouldn't be wider than 60% of the total table width
        if cols:
            all_x1 = min(c[0] for c in cols)
            all_x2 = max(c[2] for c in cols)
            table_w = all_x2 - all_x1 or 1.0
            cols = [c for c in cols if (c[2] - c[0]) / table_w < 0.6]

        # NMS: remove columns that heavily overlap another (keep the one with
        # smaller width — more specific detection wins)
        cols = _nms_boxes(cols, iou_thresh=0.5)

        return rows, cols

    def _ocr_tokens(self, img: np.ndarray) -> list:
        """
        Run OCR once on the full table image.
        Returns list of {text, cx, cy, conf} for each detected word.
        Using the centre point of each bbox to assign to cells.
        """
        results = self.ocr.readtext(img)
        tokens = []
        for bbox, text, conf in results:
            if conf < CONFIG["ocr_confidence"]:
                continue
            pts = bbox  # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            tokens.append({
                "text": text.strip(),
                "cx": (min(xs) + max(xs)) / 2,
                "cy": (min(ys) + max(ys)) / 2,
                "conf": conf,
            })
        return tokens

    def _ocr_full(self, img: np.ndarray) -> str:
        """Fallback: full-image OCR joined as plain text."""
        tokens = self._ocr_tokens(img)
        tokens.sort(key=lambda t: t["cy"])
        return " ".join(t["text"] for t in tokens)

    def _assign_tokens_to_grid(self, tokens: list,
                                rows: list, cols: list,
                                shape: tuple,
                                img: np.ndarray = None) -> list:
        """
        Assign each OCR token to the cell whose row×col bbox contains
        the token's centre point. Tokens outside all cells are dropped.
        For cells that end up empty, falls back to direct cell-crop OCR
        with a lower confidence threshold to catch faint/small characters.
        """
        h, w = shape[:2]
        grid = [[[] for _ in cols] for _ in rows]

        for tok in tokens:
            cx, cy = tok["cx"], tok["cy"]
            row_idx = None
            for ri, rb in enumerate(rows):
                if rb[1] <= cy <= rb[3]:
                    row_idx = ri
                    break
            if row_idx is None:
                continue
            col_idx = None
            for ci, cb in enumerate(cols):
                if cb[0] <= cx <= cb[2]:
                    col_idx = ci
                    break
            if col_idx is None:
                continue
            grid[row_idx][col_idx].append(tok)

        # Sort tokens within each cell left-to-right, join
        result = []
        for ri, row in enumerate(grid):
            cells = []
            # Check if this row has ANY token from the full-image OCR
            row_has_tokens = any(bool(cell_tokens) for cell_tokens in row)
            for ci, cell_tokens in enumerate(row):
                cell_tokens.sort(key=lambda t: t["cx"])
                text = " ".join(t["text"] for t in cell_tokens)
                # Only run cell-crop fallback for data rows (not header row 0)
                # and only when the row has real content from full-image OCR
                if not text.strip() and img is not None and row_has_tokens and ri > 0:
                    text = self._ocr_cell_crop(img, rows[ri], cols[ci])
                cells.append(text)
            if any(c.strip() for c in cells):
                result.append(cells)

        return result or [["(empty table)"]]

    def _ocr_cell_crop(self, img: np.ndarray, row_box: list, col_box: list) -> str:
        """Direct OCR on a single cell crop — used as fallback for empty cells.
        Tries both the passed image and a raw (no-preprocess) version since
        preprocessing can destroy faint single digits."""
        h, w = img.shape[:2]
        pad = 4
        y1 = max(0, int(row_box[1]) - pad)
        y2 = min(h, int(row_box[3]) + pad)
        x1 = max(0, int(col_box[0]) - pad)
        x2 = min(w, int(col_box[2]) + pad)

        def _try_ocr(crop):
            if crop.size == 0:
                return ""
            ch, cw = crop.shape[:2]
            if ch < 120:
                scale = max(3, 360 // ch)
                crop = np.array(
                    Image.fromarray(crop).resize((cw * scale, ch * scale), resample=Image.BICUBIC)
                )
            results = self.ocr.readtext(crop)
            results.sort(key=lambda r: r[0][0][1])
            return " ".join(r[1] for r in results if r[2] > 0.1)

        text = _try_ocr(img[y1:y2, x1:x2])
        if text:
            return text

        # Retry without preprocessing — contrast/denoise can erase faint digits
        from preprocess import preprocess_image as _pre
        # invert the preprocessing by using the raw upscaled image stored on self
        raw = getattr(self, "_raw_img", None)
        if raw is not None:
            rh, rw = raw.shape[:2]
            ry1 = max(0, int(row_box[1]) - pad)
            ry2 = min(rh, int(row_box[3]) + pad)
            rx1 = max(0, int(col_box[0]) - pad)
            rx2 = min(rw, int(col_box[2]) + pad)
            text = _try_ocr(raw[ry1:ry2, rx1:rx2])
        return text

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
    Permanently disables itself after any failure to avoid repeated crashes.
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

        self.engine  = PPStructureV3(**params)
        self._failed = False  # permanently disabled after first error

    def parse(self, img: np.ndarray) -> dict:
        if self._failed:
            return {}

        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        try:
            try:
                result = self.engine.predict(bgr)
            except Exception:
                result = self.engine(bgr)
        except Exception as e:
            print(f"[WARN] SLANet failed, permanently disabling: {e}")
            self._failed = True
            return {}

        html = self._extract_html(result)
        if not html:
            return {}

        grid = html_table_to_grid(html)
        grid = clean_table_grid(
            grid,
            col_widths=None,
            min_col_width_ratio=CONFIG.get("table_min_col_width_ratio", 0.04),
            merge_dollar_cols=CONFIG.get("table_merge_dollar_cols", True),
            merge_currency_prefix=CONFIG.get("table_merge_currency_prefix", True),
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
    import easyocr as _easyocr
    reader      = _easyocr.Reader(CONFIG["lang"], gpu=CONFIG["gpu"], verbose=False)
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
      Title    -> OCR -> \section{}
      Text     -> OCR -> escaped paragraph
      List     -> OCR + indent analysis -> \begin{itemize}
      Table    -> TATR + OCR -> \begin{tabular}
      Equation -> passthrough (Stage 3 fills this)
    """

    def __init__(self):
        print("[Stage 2] Loading OCR backend ...")
        self.ocr = _build_ocr_backend()

        self.slanet = None
        if CONFIG.get("use_slanet"):
            try:
                print("[Stage 2] Loading SLANet (PP-StructureV3) ...")
                self.slanet = SLANetTableSolver()
            except Exception as e:
                print(f"[WARN] SLANet unavailable, falling back to TATR: {e}")
                self.slanet = None

        print("[Stage 2] Loading TATR ...")
        self.tatr = TATRTableSolver(self.ocr)
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
            # Keep original before preprocessing so table solver can use it
            # for cell-crop fallback (preprocessing can erase faint digits)
            region["_raw_orig"] = region["image"].copy()
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
        raw_orig = region.get("_raw_orig")
        if scale > 1.0:
            from PIL import Image as PILImage
            img = region["image"]
            h, w = img.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            region["image"] = np.array(
                PILImage.fromarray(img).resize(new_size, resample=PILImage.BICUBIC)
            )
            # Upscale raw original to same dimensions for cell-crop fallback
            if raw_orig is not None:
                self.tatr._raw_img = np.array(
                    PILImage.fromarray(raw_orig).resize(new_size, resample=PILImage.BICUBIC)
                )
            else:
                self.tatr._raw_img = region["image"].copy()
        else:
            self.tatr._raw_img = raw_orig.copy() if raw_orig is not None else region["image"].copy()

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
        results = self.ocr.readtext(img)
        results.sort(key=lambda r: r[0][0][1])
        return [r[1] for r in results if r[2] > CONFIG["ocr_confidence"]]

    def _run_ocr_with_coords(self, img: np.ndarray) -> list:
        results = self.ocr.readtext(img)
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
