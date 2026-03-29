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
    "preprocess_contrast_factor": 2.2,  # contrast strength
    "preprocess_binarize":     False,     # adaptive thresholding
    "preprocess_sharpen":      True,     # unsharp mask
    "tatr_confidence":         0.7,      # TATR detection threshold
    "ocr_confidence":          0.3,      # EasyOCR minimum confidence
    "use_slanet":              True,    # use PaddleOCR SLANet table solver
    "table_upscale":           1.25,     # upscale factor for tables before parsing
    "table_min_col_width_ratio": 0.06,  # merge very narrow columns
    "table_merge_dollar_cols": True,    # merge columns that only contain "$"
    "ppstructure_lang":        "en",    # PaddleOCR PP-StructureV3 language
    "ppstructure_text_recognition_model_name": "en_PP-OCRv4_mobile_rec",
}

# TATR pretrained model — Microsoft, trained on PubTables-1M
TATR_MODEL = "microsoft/table-transformer-structure-recognition"
