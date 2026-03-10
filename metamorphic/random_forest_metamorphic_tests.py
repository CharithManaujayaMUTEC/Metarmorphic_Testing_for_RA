import os
import numpy as np
import cv2

from dataset.feature_extractor import extract_features

# Helpers (identical to decision_tree_metamorphic_tests.py) 

def _predict_from_path(model, image_path):
    feats = extract_features(image_path).reshape(1, -1)
    return float(model.predict(feats)[0])

def _write_temp(img, base_path, suffix):
    temp = base_path.replace(".png", f"_{suffix}_tmp.png")
    cv2.imwrite(temp, img)
    return temp

def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

# MR-1  Horizontal Flip  →  steering sign inversion

def rf_horizontal_flip_test(model, image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    temp = _write_temp(cv2.flip(img, 1), image_path, "rf_flip")
    try:
        original_pred = _predict_from_path(model, image_path)
        flipped_pred  = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    expected   = -original_pred
    difference = abs(flipped_pred - expected)
    tolerance  = 0.08 if abs(original_pred) > 0.10 else 0.15

    return {
        "test":             "rf_horizontal_flip",
        "original_pred":    original_pred,
        "flipped_pred":     flipped_pred,
        "expected_flipped": expected,
        "difference":       difference,
        "tolerance":        tolerance,
        "passed":           difference < tolerance,
    }

# MR-2  Brightness Invariance

def rf_brightness_invariance_test(model, image_path, delta=30):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    bright = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    temp   = _write_temp(bright, image_path, "rf_bright")
    try:
        original_pred = _predict_from_path(model, image_path)
        bright_pred   = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    difference = abs(bright_pred - original_pred)
    tolerance  = 0.05

    return {
        "test":          "rf_brightness_invariance",
        "delta":         delta,
        "original_pred": original_pred,
        "bright_pred":   bright_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-3  Translation Consistency

def rf_translation_consistency_test(model, image_path, shift_px=8):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w     = img.shape[:2]
    M        = np.float32([[1, 0, shift_px], [0, 1, 0]])
    shifted  = cv2.warpAffine(img, M, (w, h))
    temp     = _write_temp(shifted, image_path, "rf_shift")
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
        "test":          "rf_translation_consistency",
        "shift_px":      shift_px,
        "original_pred": original_pred,
        "shifted_pred":  shifted_pred,
        "difference":    difference,
        "same_sign":     same_sign,
        "tolerance":     tolerance,
        "passed":        passed,
    }

# MR-4  Vertical Crop Consistency

def rf_vertical_crop_test(model, image_path, crop_top_fraction=0.20):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w    = img.shape[:2]
    cut     = int(h * crop_top_fraction)
    cropped = cv2.resize(img[cut:, :], (w, h))
    temp    = _write_temp(cropped, image_path, "rf_crop")
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
        "test":          "rf_vertical_crop",
        "crop_fraction": crop_top_fraction,
        "original_pred": original_pred,
        "cropped_pred":  cropped_pred,
        "difference":    difference,
        "same_sign":     same_sign,
        "tolerance":     tolerance,
        "passed":        passed,
    }

# MR-5  Feature-Space Monotonicity

def rf_monotonicity_test(model, image_path, delta=0.10):

    feats = extract_features(image_path).reshape(1, -1).astype(np.float64)
    original_pred = float(model.predict(feats)[0])

    TOP_OFFSET_IDX = 1
    feats_p = feats.copy()
    sign    = 1.0 if feats[0, TOP_OFFSET_IDX] >= 0 else -1.0
    feats_p[0, TOP_OFFSET_IDX] = float(
        np.clip(feats[0, TOP_OFFSET_IDX] + sign * delta, -1.0, 1.0)
    )

    perturbed_pred = float(model.predict(feats_p)[0])
    mag_change     = abs(perturbed_pred) - abs(original_pred)
    epsilon        = 0.02

    return {
        "test":            "rf_monotonicity",
        "top_offset_orig": float(feats[0, TOP_OFFSET_IDX]),
        "top_offset_new":  float(feats_p[0, TOP_OFFSET_IDX]),
        "original_pred":   original_pred,
        "perturbed_pred":  perturbed_pred,
        "difference":      mag_change,
        "tolerance":       epsilon,
        "passed":          mag_change >= -epsilon,
    }

# MR-6  Ensemble Consistency  (RF-specific)

def rf_ensemble_consistency_test(model, image_path,
                                 subset_fraction=0.5):

    feats = extract_features(image_path).reshape(1, -1).astype(np.float64)

    full_pred = float(model.predict(feats)[0])

    # Random half of trees
    rng      = np.random.default_rng(seed=0)   # fixed seed for reproducibility
    n        = len(model._trees)
    half_idx = rng.choice(n, size=n // 2, replace=False)

    half_preds = np.array([
        float(model._trees[i].predict(feats)[0]) for i in half_idx
    ])
    half_pred  = float(half_preds.mean())
    difference = abs(half_pred - full_pred)
    tolerance  = 0.05

    return {
        "test":          "rf_ensemble_consistency",
        "full_pred":     full_pred,
        "half_pred":     half_pred,
        "n_trees_used":  len(half_idx),
        "original_pred": full_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-7  Uncertainty Bounds  (RF-specific)

def rf_uncertainty_bounds_test(model, image_path,
                               max_std_threshold=0.05):

    feats = extract_features(image_path).reshape(1, -1).astype(np.float64)

    pred = float(model.predict(feats)[0])
    std  = float(model.predict_std(feats)[0])

    passed = std < max_std_threshold

    return {
        "test":          "rf_uncertainty_bounds",
        "original_pred": pred,
        "prediction_std": std,
        "difference":    std,          # reuse 'difference' key for runner
        "tolerance":     max_std_threshold,
        "passed":        passed,
    }