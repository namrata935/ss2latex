# =============================================================================
# config.py — Central configuration for Stage 2 pipeline
# =============================================================================

CONFIG = {
    "lang":                    ["en"],   # EasyOCR language list
    "indent_threshold_px":     20,       # pixel offset = one indent level
    "row_merge_tolerance_px":  10,       # TB-YX row grouping tolerance
    "output_json":             "stage2_output.json",
    "gpu":                     False,    # set True if NVIDIA GPU available
    "preprocess_deskew":       True,     # correct screenshot rotation
    "preprocess_denoise":      True,     # remove compression artifacts
    "preprocess_contrast":     True,     # enhance text visibility
    "tatr_confidence":         0.7,      # TATR detection threshold
    "ocr_confidence":          0.3,      # EasyOCR minimum confidence
}

# TATR pretrained model — Microsoft, trained on PubTables-1M
TATR_MODEL = "microsoft/table-transformer-structure-recognition"
