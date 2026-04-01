# =============================================================================
# utils.py — Shared helpers: image loading, LaTeX escaping, serialisation
# =============================================================================

import re
import json
from html import unescape
from html.parser import HTMLParser
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load any image file -> RGB numpy array via Pillow. No cv2 needed."""
    return np.array(Image.open(path).convert("RGB"))


# ─────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────

_BULLET_RE = re.compile(
    r"^(\d+[\.\)]|\([ivxlcdmIVXLCDM]+\)|[a-zA-Z][\.\)]"
    r"|[\u2022\u2023\u25E6\u2043\-\*\+>])\s*",
    re.IGNORECASE
)

_LATEX_SPECIAL = str.maketrans({
    "&":  r"\&",  "%":  r"\%",  "$":  r"\$",
    "#":  r"\#",  "_":  r"\_",  "{":  r"\{",
    "}":  r"\}",  "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
})

def latex_escape(text: str) -> str:
    """Escapes LaTeX special characters so output compiles cleanly."""
    return text.translate(_LATEX_SPECIAL)

def strip_bullet(text: str) -> str:
    """Removes leading bullet/number markers from list item text."""
    return _BULLET_RE.sub("", text).strip()


# ─────────────────────────────────────────────
# HTML table -> grid + LaTeX
# ─────────────────────────────────────────────

class _HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._cur_row = []
        self._cur_cell = []
        self._in_cell = False
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._cur_row = []
        if tag in ("td", "th"):
            self._in_cell = True
            self._cur_cell = []
            self._colspan = 1
            for k, v in attrs:
                if k == "colspan":
                    try:
                        self._colspan = max(1, int(v))
                    except ValueError:
                        self._colspan = 1

    def handle_data(self, data):
        if self._in_cell:
            self._cur_cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            text = unescape("".join(self._cur_cell)).strip()
            self._cur_row.append(text)
            for _ in range(self._colspan - 1):
                self._cur_row.append("")
            self._in_cell = False
        if tag == "tr":
            if self._cur_row:
                self.rows.append(self._cur_row)
            self._cur_row = []


def html_table_to_grid(html: str) -> list:
    if not html:
        return []
    parser = _HTMLTableParser()
    parser.feed(html)
    return parser.rows


def grid_to_tabular(grid: list) -> str:
    if not grid:
        return ""
    col_count = max(len(row) for row in grid)
    col_spec = "|" + "l|" * col_count
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\hline"]
    for row in grid:
        padded = row + [""] * (col_count - len(row))
        escaped = [latex_escape(c.strip()) for c in padded]
        lines.append(" & ".join(escaped) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def clean_table_grid(grid: list,
                     col_widths: list | None = None,
                     min_col_width_ratio: float = 0.06,
                     merge_dollar_cols: bool = True,
                     merge_currency_prefix: bool = True) -> list:
    """
    Heuristic cleanup:
      - merge very narrow columns into neighbors
      - merge columns that only contain "$"/currency symbols or are empty
      - prepend currency prefix cells onto the next column's values
    """
    if not grid:
        return grid

    col_count = max(len(r) for r in grid)
    rows = [r + [""] * (col_count - len(r)) for r in grid]

    # Currency/dollar symbols that should be prefixes, not standalone columns
    _CURRENCY = {"$", r"\$", "£", "€", "¥", "₹", "USD", "GBP", "EUR"}

    # Decide which columns to merge
    merge_cols = set()
    if merge_dollar_cols:
        for c in range(col_count):
            # Check data rows only (skip header row 0) for currency detection
            data_rows = range(1, len(rows)) if len(rows) > 1 else range(len(rows))
            vals = [rows[r][c].strip() for r in data_rows]
            non_empty = [v for v in vals if v]
            if not non_empty:
                continue
            currency_count = sum(1 for v in non_empty if v in _CURRENCY)
            # Merge if ≥75% of data values are currency symbols
            # (tolerates occasional OCR misread like "S" for "$")
            if currency_count / len(non_empty) >= 0.75:
                merge_cols.add(c)
                # Fix misread currency symbols in this column
                for r in data_rows:
                    v = rows[r][c].strip()
                    if v and v not in _CURRENCY and len(v) == 1:
                        rows[r][c] = "$"

    if col_widths and len(col_widths) == col_count:
        max_w = max(col_widths) or 1.0
        for c, w in enumerate(col_widths):
            if (w / max_w) < min_col_width_ratio:
                merge_cols.add(c)

    # Merge columns: currency prefix cols prepend to right neighbor
    for c in sorted(merge_cols):
        target = c + 1 if c + 1 < col_count else c - 1
        if target < 0:
            continue
        for r in range(len(rows)):
            cell = rows[r][c].strip()
            if cell:
                neighbor = rows[r][target].strip()
                if merge_currency_prefix and cell in _CURRENCY and neighbor:
                    # "$ value" with space (standard notation)
                    rows[r][target] = f"{cell} {neighbor}"
                elif neighbor:
                    rows[r][target] = f"{neighbor} {cell}"
                else:
                    rows[r][target] = cell
            rows[r][c] = ""

    # Drop fully empty columns
    keep = []
    for c in range(col_count):
        if any(rows[r][c].strip() for r in range(len(rows))):
            keep.append(c)

    cleaned = []
    for r in range(len(rows)):
        cleaned.append([rows[r][c] for c in keep])
    return cleaned


def clean_latex_table(latex: str) -> str:
    """
    Best-effort cleanup for model-generated LaTeX tables:
      - remove stray \\textbackslash tokens
      - strip \\cellcolor/\\rowcolor
      - normalize row column counts to match tabular spec
    """
    if not latex or "\\begin{tabular" not in latex:
        return latex

    text = latex.replace("\\textbackslash", "").strip()
    text = re.sub(r"\\cellcolor\\[[^\\]]+\\]\\s*", "", text)
    text = re.sub(r"\\rowcolor\\[[^\\]]+\\]\\s*", "", text)
    text = re.sub(r"\\textbf\\{([^}]*)\\}", r"\\1", text)
    text = re.sub(r"\\texttt\\{([^}]*)\\}", r"\\1", text)

    m = re.search(r"\\begin\\{tabular\\}\\{([^}]*)\\}", text)
    if not m:
        return text
    spec = m.group(1)
    # simplify column spec: keep only count, use left-aligned columns

    # count columns from spec
    col_count = 0
    i = 0
    while i < len(spec):
        ch = spec[i]
        if ch in "lcrX":
            col_count += 1
        elif ch in "pmb":
            # p{..}, m{..}, b{..}
            col_count += 1
            if i + 1 < len(spec) and spec[i + 1] == "{":
                depth = 1
                i += 2
                while i < len(spec) and depth > 0:
                    if spec[i] == "{":
                        depth += 1
                    elif spec[i] == "}":
                        depth -= 1
                    i += 1
                continue
        i += 1
    col_count = max(col_count, 1)
    simple_spec = "|" + "l|" * col_count

    # extract body
    body = re.sub(r"^.*?\\begin\\{tabular\\}\\{[^}]*\\}", "", text, flags=re.S)
    body = re.sub(r"\\end\\{tabular\\}.*$", "", body, flags=re.S).strip()

    # split rows
    raw_rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
    rows = []
    for r in raw_rows:
        r = r.replace("\\hline", "").strip()
        if not r:
            continue
        cells = [c.strip() for c in r.split("&")]
        if len(cells) > col_count:
            head = cells[:col_count - 1]
            tail = " ".join(c for c in cells[col_count - 1:] if c)
            cells = head + [tail]
        elif len(cells) < col_count:
            cells += [""] * (col_count - len(cells))
        rows.append(cells)

    # rebuild
    lines = [f"\\begin{{tabular}}{{{simple_spec}}}", "\\hline"]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# JSON serialisation
# ─────────────────────────────────────────────

def to_serialisable(region: dict) -> dict:
    """
    Strips the numpy image array from a region dict
    and converts numpy types to native Python for JSON serialisation.
    """
    out = {k: v for k, v in region.items() if k != "image"}

    def _conv(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj

    return json.loads(json.dumps(out, default=_conv))
