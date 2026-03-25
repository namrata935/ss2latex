# =============================================================================
# preprocess.py — Screenshot preprocessing pipeline
#
# IMPROVEMENT 3 over PDF2LaTeX (2020):
# The 2020 paper listed "noise removal and deskewing" as future work they
# never completed (Section 6). This module implements exactly that for
# screenshot-specific noise: anti-aliasing, DPI variance, sub-pixel noise.
#
# Three stages:
#   1. Deskew   — corrects slight rotation from screen capture angle
#   2. Denoise  — removes JPEG/screen compression artifacts
#   3. Contrast — enhances text visibility on low-contrast backgrounds
# =============================================================================

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from config import CONFIG


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Applies all enabled preprocessing steps to a screenshot crop
    before passing it to EasyOCR or TATR.

    Steps are individually togglable in config.py.
    """
    result = img.copy()

    if CONFIG["preprocess_deskew"]:
        result = _deskew(result)

    if CONFIG["preprocess_denoise"]:
        result = cv2.fastNlMeansDenoisingColored(
            result, None,
            h=10, hColor=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    if CONFIG["preprocess_contrast"]:
        pil      = Image.fromarray(result)
        pil      = ImageEnhance.Contrast(pil).enhance(1.5)
        result   = np.array(pil)

    return result


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Detects skew angle using Hough line transform and corrects it.
    Only corrects angles between 0.5° and 10° to avoid over-rotation
    on naturally tilted content like italics.
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return img

    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if 0.5 < abs(angle) < 10:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    h, w         = img.shape[:2]
    M            = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
