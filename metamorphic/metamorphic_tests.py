import torch
import torchvision.transforms.functional as TF
import os
import numpy as np
import cv2
from dataset.feature_extractor import extract_features

# =========================
# HELPER
# =========================
def _predict_from_path(model, image_path):
    feats = extract_features(image_path).reshape(1, -1)
    return float(model.predict(feats)[0])


# =========================
# CNN TEST
# =========================
def cnn_horizontal_flip_test(model, image, tolerance=0.05):
    flipped = TF.hflip(image)

    f_pred = model(flipped.unsqueeze(0))
    o_pred = model(image.unsqueeze(0))

    expected = -o_pred
    diff = torch.abs(f_pred - expected).item()

    return {
        "test": "cnn_flip",
        "difference": diff,
        "passed": diff < tolerance
    }


# =========================
# REGRESSION TESTS
# =========================
def brightness_test(model, path):
    img = cv2.imread(path)
    bright = np.clip(img + 30, 0, 255).astype(np.uint8)

    temp = path.replace(".png", "_b.png")
    cv2.imwrite(temp, bright)

    o = _predict_from_path(model, path)
    b = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "brightness", "difference": abs(o-b), "passed": abs(o-b) < 0.05}


def flip_test(model, path):
    img = cv2.imread(path)
    flipped = cv2.flip(img, 1)

    temp = path.replace(".png", "_f.png")
    cv2.imwrite(temp, flipped)

    o = _predict_from_path(model, path)
    f = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "flip", "difference": abs(f+o), "passed": abs(f+o) < 0.05}


def translation_test(model, path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    M = np.float32([[1, 0, 5], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h))

    temp = path.replace(".png", "_t.png")
    cv2.imwrite(temp, shifted)

    o = _predict_from_path(model, path)
    s = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "translation", "difference": abs(o-s), "passed": abs(o-s) < 0.15}


def noise_test(model, path):
    img = cv2.imread(path)
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

    temp = path.replace(".png", "_n.png")
    cv2.imwrite(temp, noisy)

    o = _predict_from_path(model, path)
    n = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "noise", "difference": abs(o-n), "passed": abs(o-n) < 0.15}


def blur_test(model, path):
    img = cv2.imread(path)
    blur = cv2.GaussianBlur(img, (5,5), 0)

    temp = path.replace(".png", "_bl.png")
    cv2.imwrite(temp, blur)

    o = _predict_from_path(model, path)
    b = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "blur", "difference": abs(o-b), "passed": abs(o-b) < 0.15}


# =========================
# MASTER RUNNER
# =========================
def run_all_tests(model_reg, model_cnn, image_path, image_tensor):
    results = []

    # CNN
    results.append(cnn_horizontal_flip_test(model_cnn, image_tensor))

    # Regression
    tests = [brightness_test, flip_test, translation_test, noise_test, blur_test]

    for t in tests:
        try:
            results.append(t(model_reg, image_path))
        except Exception as e:
            results.append({"test": t.__name__, "error": str(e)})

    return results
