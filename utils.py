# =============================================================================
# utils.py — Shared helpers: image loading, LaTeX escaping, serialisation
# =============================================================================

import re
import json
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
