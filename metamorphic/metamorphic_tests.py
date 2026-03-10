import torch
import torchvision.transforms.functional as TF
import os
import numpy as np
import cv2
from dataset.feature_extractor import extract_features

# Original CNN test (for the SteeringRegression model) 
def horizontal_flip_test(model, image, steering):

    """Metamorphic test for the CNN SteeringRegression model."""
    flipped_image       = TF.hflip(image)
    flipped_prediction  = model(flipped_image.unsqueeze(0))
    original_prediction = model(image.unsqueeze(0))
    expected            = -original_prediction
    difference          = torch.abs(flipped_prediction - expected)
    return difference.item()

# Multiple-Regression metamorphic tests 

def _predict_from_path(model, image_path):
    feats = extract_features(image_path).reshape(1, -1)
    return float(model.predict(feats)[0])


#  MR-1: Brightness invariance 
def brightness_invariance_test(model, image_path, delta=30):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bright = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)

    temp_path = image_path.replace(".png", "_bright_temp.png")
    cv2.imwrite(temp_path, img_bright)

    original_pred = _predict_from_path(model, image_path)
    bright_pred   = _predict_from_path(model, temp_path)

    os.remove(temp_path)

    difference = abs(original_pred - bright_pred)
    tolerance  = 0.05  # tight — brightness should not affect lane geometry features

    return {
        "test":          "brightness_invariance",
        "original_pred": original_pred,
        "bright_pred":   bright_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-2: Horizontal flip 
def horizontal_flip_regression_test(model, image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    flipped   = cv2.flip(img, 1)
    temp_path = image_path.replace(".png", "_flip_temp.png")
    cv2.imwrite(temp_path, flipped)

    original_pred = _predict_from_path(model, image_path)
    flipped_pred  = _predict_from_path(model, temp_path)

    os.remove(temp_path)

    expected_flipped = -original_pred
    difference       = abs(flipped_pred - expected_flipped)
    tolerance        = 0.05

    return {
        "test":             "horizontal_flip_regression",
        "original_pred":    original_pred,
        "flipped_pred":     flipped_pred,
        "expected_flipped": expected_flipped,
        "difference":       difference,
        "tolerance":        tolerance,
        "passed":           difference < tolerance,
    }

# MR-3: Small translation 
def translation_invariance_test(model, image_path, shift_pixels=5):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w  = img.shape[:2]
    M     = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
    shifted   = cv2.warpAffine(img, M, (w, h))
    temp_path = image_path.replace(".png", "_shift_temp.png")
    cv2.imwrite(temp_path, shifted)

    original_pred = _predict_from_path(model, image_path)
    shifted_pred  = _predict_from_path(model, temp_path)

    os.remove(temp_path)

    difference = abs(shifted_pred - original_pred)
    tolerance  = 0.15

    return {
        "test":          "translation_invariance",
        "original_pred": original_pred,
        "shifted_pred":  shifted_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-4: Vertical crop (distance invariance) 
def vertical_crop_test(model, image_path, crop_top_fraction=0.2):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w    = img.shape[:2]
    cut_row = int(h * crop_top_fraction)
    cropped = img[cut_row:, :]
    cropped_resized = cv2.resize(cropped, (w, h))

    temp_path = image_path.replace(".png", "_crop_temp.png")
    cv2.imwrite(temp_path, cropped_resized)

    original_pred = _predict_from_path(model, image_path)
    cropped_pred  = _predict_from_path(model, temp_path)

    os.remove(temp_path)

    same_sign  = (np.sign(original_pred) == np.sign(cropped_pred)) or \
                 (abs(original_pred) < 0.01)          # near-zero: sign is irrelevant
    difference = abs(cropped_pred - original_pred)
    tolerance  = 0.20

    passed = same_sign and (difference < tolerance)

    return {
        "test":          "vertical_crop_consistency",
        "original_pred": original_pred,
        "cropped_pred":  cropped_pred,
        "same_sign":     same_sign,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        passed,
    }