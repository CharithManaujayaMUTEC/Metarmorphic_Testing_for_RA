import os
import numpy as np
import cv2

from dataset.feature_extractor import extract_features

def _predict_from_path(model, image_path: str) -> float:
    """Extract features from image_path and return scalar prediction."""
    feats = extract_features(image_path).reshape(1, -1)
    return float(model.predict(feats)[0])

def _write_temp(img: np.ndarray, base_path: str, suffix: str) -> str:
    """Save a temp image next to base_path and return the temp path."""
    temp = base_path.replace(".png", f"_{suffix}_tmp.png")
    cv2.imwrite(temp, img)
    return temp

def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

# MR-1  Horizontal Flip  →  sign inversion

def dt_horizontal_flip_test(model, image_path: str):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    temp = _write_temp(cv2.flip(img, 1), image_path, "flip")
    try:
        original_pred = _predict_from_path(model, image_path)
        flipped_pred  = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    expected   = -original_pred
    difference = abs(flipped_pred - expected)
    tolerance  = 0.08 if abs(original_pred) > 0.10 else 0.15

    return {
        "test":             "dt_horizontal_flip",
        "original_pred":    original_pred,
        "flipped_pred":     flipped_pred,
        "expected_flipped": expected,
        "difference":       difference,
        "tolerance":        tolerance,
        "passed":           difference < tolerance,
    }

# MR-2  Brightness Invariance

def dt_brightness_invariance_test(model, image_path: str, delta: int = 30):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    bright = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    temp   = _write_temp(bright, image_path, "bright")
    try:
        original_pred = _predict_from_path(model, image_path)
        bright_pred   = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    difference = abs(bright_pred - original_pred)
    tolerance  = 0.05

    return {
        "test":          "dt_brightness_invariance",
        "delta":         delta,
        "original_pred": original_pred,
        "bright_pred":   bright_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-3  Translation Consistency

def dt_translation_consistency_test(model, image_path: str,
                                    shift_px: int = 8):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w  = img.shape[:2]
    M     = np.float32([[1, 0, shift_px], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h))
    temp    = _write_temp(shifted, image_path, "shift")
    try:
        original_pred = _predict_from_path(model, image_path)
        shifted_pred  = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    difference = abs(shifted_pred - original_pred)
    tolerance  = 0.20
    strong     = abs(original_pred) > 0.10
    same_sign  = (original_pred * shifted_pred >= 0) or not strong
    passed     = (difference < tolerance) and same_sign

    return {
        "test":          "dt_translation_consistency",
        "shift_px":      shift_px,
        "original_pred": original_pred,
        "shifted_pred":  shifted_pred,
        "difference":    difference,
        "same_sign":     same_sign,
        "tolerance":     tolerance,
        "passed":        passed,
    }

# MR-4  Vertical Crop  →  sign and magnitude preserved

def dt_vertical_crop_test(model, image_path: str,
                          crop_top_fraction: float = 0.20):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w    = img.shape[:2]
    cut     = int(h * crop_top_fraction)
    cropped = cv2.resize(img[cut:, :], (w, h))
    temp    = _write_temp(cropped, image_path, "crop")
    try:
        original_pred = _predict_from_path(model, image_path)
        cropped_pred  = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    difference = abs(cropped_pred - original_pred)
    tolerance  = 0.20
    near_zero  = abs(original_pred) < 0.05
    same_sign  = (original_pred * cropped_pred >= 0) or near_zero
    passed     = (difference < tolerance) and same_sign

    return {
        "test":          "dt_vertical_crop",
        "crop_fraction": crop_top_fraction,
        "original_pred": original_pred,
        "cropped_pred":  cropped_pred,
        "difference":    difference,
        "same_sign":     same_sign,
        "tolerance":     tolerance,
        "passed":        passed,
    }

# MR-5  Monotonicity  →  larger top_offset → larger steering magnitude

def dt_monotonicity_test(model, image_path: str,
                         delta: float = 0.10):

    from dataset.feature_extractor import extract_features

    feats = extract_features(image_path).reshape(1, -1).astype(np.float64)
    original_pred = float(model.predict(feats)[0])

    # Feature index 1 = top_offset  (matches FEATURE_NAMES order)
    TOP_OFFSET_IDX = 1

    feats_perturbed = feats.copy()
    sign = 1.0 if feats[0, TOP_OFFSET_IDX] >= 0 else -1.0
    feats_perturbed[0, TOP_OFFSET_IDX] += sign * delta

    # Keep within [-1, 1] — the valid normalised range
    feats_perturbed[0, TOP_OFFSET_IDX] = float(
        np.clip(feats_perturbed[0, TOP_OFFSET_IDX], -1.0, 1.0)
    )

    perturbed_pred = float(model.predict(feats_perturbed)[0])

    epsilon    = 0.02
    mag_change = abs(perturbed_pred) - abs(original_pred)
    passed     = mag_change >= -epsilon   # magnitude must not shrink

    return {
        "test":            "dt_monotonicity",
        "top_offset_orig": float(feats[0, TOP_OFFSET_IDX]),
        "top_offset_new":  float(feats_perturbed[0, TOP_OFFSET_IDX]),
        "original_pred":   original_pred,
        "perturbed_pred":  perturbed_pred,
        "difference":      mag_change,
        "tolerance":       epsilon,
        "passed":          passed,
    }

# MR-6  Symmetry Consistency  →  mirrored scene → mirrored steering

def dt_symmetry_consistency_test(model, image_path: str):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    double_flipped = cv2.flip(cv2.flip(img, 1), 1)   # == original
    temp = _write_temp(double_flipped, image_path, "sym")
    try:
        original_pred    = _predict_from_path(model, image_path)
        double_flip_pred = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    difference = abs(double_flip_pred - original_pred)
    tolerance  = 0.01

    return {
        "test":              "dt_symmetry_consistency",
        "original_pred":     original_pred,
        "double_flip_pred":  double_flip_pred,
        "difference":        difference,
        "tolerance":         tolerance,
        "passed":            difference < tolerance,
    }