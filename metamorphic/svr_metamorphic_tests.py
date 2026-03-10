import os
import numpy as np
import cv2

from dataset.feature_extractor import extract_features

# Shared helpers  (identical to dt/rf metamorphic tests) 

def _predict_from_path(model, image_path: str) -> float:
    feats = extract_features(image_path).reshape(1, -1)
    return float(model.predict(feats)[0])

def _write_temp(img: np.ndarray, base_path: str, suffix: str) -> str:
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

def svr_horizontal_flip_test(model, image_path: str):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    temp = _write_temp(cv2.flip(img, 1), image_path, "svr_flip")
    try:
        original_pred = _predict_from_path(model, image_path)
        flipped_pred  = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    expected   = -original_pred
    difference = abs(flipped_pred - expected)
    tolerance  = 0.08 if abs(original_pred) > 0.10 else 0.15

    return {
        "test":             "svr_horizontal_flip",
        "original_pred":    original_pred,
        "flipped_pred":     flipped_pred,
        "expected_flipped": expected,
        "difference":       difference,
        "tolerance":        tolerance,
        "passed":           difference < tolerance,
    }

# MR-2  Brightness Invariance

def svr_brightness_invariance_test(model, image_path: str, delta: int = 30):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    bright = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    temp   = _write_temp(bright, image_path, "svr_bright")
    try:
        original_pred = _predict_from_path(model, image_path)
        bright_pred   = _predict_from_path(model, temp)
    finally:
        _cleanup(temp)

    difference = abs(bright_pred - original_pred)
    tolerance  = 0.05

    return {
        "test":          "svr_brightness_invariance",
        "delta":         delta,
        "original_pred": original_pred,
        "bright_pred":   bright_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-3  Translation Consistency

def svr_translation_consistency_test(model, image_path: str, shift_px: int = 8):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w    = img.shape[:2]
    M       = np.float32([[1, 0, shift_px], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h))
    temp    = _write_temp(shifted, image_path, "svr_shift")
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
        "test":          "svr_translation_consistency",
        "shift_px":      shift_px,
        "original_pred": original_pred,
        "shifted_pred":  shifted_pred,
        "difference":    difference,
        "same_sign":     same_sign,
        "tolerance":     tolerance,
        "passed":        passed,
    }

# MR-4  Vertical Crop Consistency

def svr_vertical_crop_test(model, image_path: str,
                           crop_top_fraction: float = 0.20):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w    = img.shape[:2]
    cut     = int(h * crop_top_fraction)
    cropped = cv2.resize(img[cut:, :], (w, h))
    temp    = _write_temp(cropped, image_path, "svr_crop")
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
        "test":          "svr_vertical_crop",
        "crop_fraction": crop_top_fraction,
        "original_pred": original_pred,
        "cropped_pred":  cropped_pred,
        "difference":    difference,
        "same_sign":     same_sign,
        "tolerance":     tolerance,
        "passed":        passed,
    }

# MR-5  Feature-Space Monotonicity

def svr_monotonicity_test(model, image_path: str, delta: float = 0.10):

    feats = extract_features(image_path).reshape(1, -1).astype(np.float64)
    original_pred = float(model.predict(feats)[0])

    TOP_OFFSET_IDX = 1    # index in FEATURE_NAMES
    feats_p = feats.copy()
    sign    = 1.0 if feats[0, TOP_OFFSET_IDX] >= 0 else -1.0
    feats_p[0, TOP_OFFSET_IDX] = float(
        np.clip(feats[0, TOP_OFFSET_IDX] + sign * delta, -1.0, 1.0)
    )

    perturbed_pred = float(model.predict(feats_p)[0])
    mag_change     = abs(perturbed_pred) - abs(original_pred)
    epsilon        = 0.02

    return {
        "test":            "svr_monotonicity",
        "top_offset_orig": float(feats[0, TOP_OFFSET_IDX]),
        "top_offset_new":  float(feats_p[0, TOP_OFFSET_IDX]),
        "original_pred":   original_pred,
        "perturbed_pred":  perturbed_pred,
        "difference":      mag_change,
        "tolerance":       epsilon,
        "passed":          mag_change >= -epsilon,
    }

# MR-6  Epsilon-Tube Consistency  (SVR-specific)

def svr_epsilon_tube_test(model, image_path: str,
                          noise_scale: float = 0.005):
    
    feats = extract_features(image_path).reshape(1, -1).astype(np.float64)
    original_pred = float(model.predict(feats)[0])

    rng         = np.random.default_rng(seed=7)   # fixed for reproducibility
    noise       = rng.normal(0.0, noise_scale, size=feats.shape)
    feats_noisy = np.clip(feats + noise, -2.0, 2.0)

    noisy_pred = float(model.predict(feats_noisy)[0])
    difference = abs(noisy_pred - original_pred)
    tolerance  = model.epsilon + 0.02

    return {
        "test":          "svr_epsilon_tube",
        "original_pred": original_pred,
        "noisy_pred":    noisy_pred,
        "noise_scale":   noise_scale,
        "epsilon":       model.epsilon,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-7  Kernel Interpolation Consistency  (SVR-specific)

def svr_kernel_interpolation_test(model, image_path_a: str,
                                  image_path_b: str,
                                  alpha: float = 0.5):

    feats_a = extract_features(image_path_a).reshape(1, -1).astype(np.float64)
    feats_b = extract_features(image_path_b).reshape(1, -1).astype(np.float64)
    feats_m = alpha * feats_a + (1.0 - alpha) * feats_b

    pred_a = float(model.predict(feats_a)[0])
    pred_b = float(model.predict(feats_b)[0])
    pred_m = float(model.predict(feats_m)[0])

    lo        = min(pred_a, pred_b)
    hi        = max(pred_a, pred_b)
    tolerance = 0.10
    passed    = (lo - tolerance) <= pred_m <= (hi + tolerance)
    # difference = how far pred_m is outside [lo, hi]
    difference = max(0.0, lo - tolerance - pred_m,
                         pred_m - hi - tolerance)

    return {
        "test":          "svr_kernel_interpolation",
        "pred_a":        pred_a,
        "pred_b":        pred_b,
        "pred_midpoint": pred_m,
        "alpha":         alpha,
        "original_pred": pred_a,   # for consistent runner display
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        passed,
    }