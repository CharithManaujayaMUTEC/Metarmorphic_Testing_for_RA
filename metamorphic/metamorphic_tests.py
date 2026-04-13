import torch
import torchvision.transforms.functional as TF
import os
import numpy as np
import cv2
from dataset.feature_extractor import extract_features

# =========================
# CNN MODEL TEST
# =========================
def horizontal_flip_test(model, image, steering):
    flipped_image       = TF.hflip(image)
    flipped_prediction  = model(flipped_image.unsqueeze(0))
    original_prediction = model(image.unsqueeze(0))
    expected            = -original_prediction
    difference          = torch.abs(flipped_prediction - expected)
    return difference.item()


# =========================
# HELPER FUNCTION
# =========================
def _predict_from_path(model, image_path):
    feats = extract_features(image_path).reshape(1, -1)
    return float(model.predict(feats)[0])


# =========================
# BASIC METAMORPHIC RELATIONS
# =========================

def brightness_invariance_test(model, image_path, delta=30):
    img = cv2.imread(image_path)
    img_bright = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)

    temp = image_path.replace(".png", "_bright.png")
    cv2.imwrite(temp, img_bright)

    o = _predict_from_path(model, image_path)
    b = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "brightness", "difference": abs(o - b), "passed": abs(o - b) < 0.05}


def horizontal_flip_regression_test(model, image_path):
    img = cv2.imread(image_path)
    flipped = cv2.flip(img, 1)

    temp = image_path.replace(".png", "_flip.png")
    cv2.imwrite(temp, flipped)

    o = _predict_from_path(model, image_path)
    f = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "flip", "difference": abs(f + o), "passed": abs(f + o) < 0.05}


def translation_invariance_test(model, image_path, shift_pixels=5):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    M = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h))

    temp = image_path.replace(".png", "_shift.png")
    cv2.imwrite(temp, shifted)

    o = _predict_from_path(model, image_path)
    s = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "translation", "difference": abs(o - s), "passed": abs(o - s) < 0.15}


def vertical_crop_test(model, image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    cropped = img[int(h * 0.2):, :]
    cropped = cv2.resize(cropped, (w, h))

    temp = image_path.replace(".png", "_crop.png")
    cv2.imwrite(temp, cropped)

    o = _predict_from_path(model, image_path)
    c = _predict_from_path(model, temp)
    os.remove(temp)

    same_sign = (np.sign(o) == np.sign(c)) or abs(o) < 0.01
    diff = abs(o - c)

    return {"test": "crop", "difference": diff, "passed": same_sign and diff < 0.2}


# =========================
# ADVANCED REAL-WORLD MRs
# =========================

def pothole_noise_test(model, image_path):
    img = cv2.imread(image_path)

    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    temp = image_path.replace(".png", "_noise.png")
    cv2.imwrite(temp, noisy)

    o = _predict_from_path(model, image_path)
    n = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "pothole_noise", "difference": abs(o - n), "passed": abs(o - n) < 0.15}


def shadow_test(model, image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    shadow = img.copy()
    shadow[:, :w//2] = (shadow[:, :w//2] * 0.5).astype(np.uint8)

    temp = image_path.replace(".png", "_shadow.png")
    cv2.imwrite(temp, shadow)

    o = _predict_from_path(model, image_path)
    s = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "shadow", "difference": abs(o - s), "passed": abs(o - s) < 0.15}


def blur_test(model, image_path):
    img = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    temp = image_path.replace(".png", "_blur.png")
    cv2.imwrite(temp, blurred)

    o = _predict_from_path(model, image_path)
    b = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "blur", "difference": abs(o - b), "passed": abs(o - b) < 0.15}


def contrast_test(model, image_path):
    img = cv2.imread(image_path)
    contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    temp = image_path.replace(".png", "_contrast.png")
    cv2.imwrite(temp, contrast)

    o = _predict_from_path(model, image_path)
    c = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "contrast", "difference": abs(o - c), "passed": abs(o - c) < 0.15}


def occlusion_test(model, image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    occ = img.copy()
    cv2.rectangle(occ, (w//3, h//2), (w//2, h), (0, 0, 0), -1)

    temp = image_path.replace(".png", "_occ.png")
    cv2.imwrite(temp, occ)

    o = _predict_from_path(model, image_path)
    oc = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "occlusion", "difference": abs(o - oc), "passed": abs(o - oc) < 0.2}


def rain_test(model, image_path):
    img = cv2.imread(image_path)
    rain = img.copy()

    for _ in range(500):
        x = np.random.randint(0, rain.shape[1])
        y = np.random.randint(0, rain.shape[0])
        cv2.line(rain, (x, y), (x+1, y+5), (200, 200, 200), 1)

    temp = image_path.replace(".png", "_rain.png")
    cv2.imwrite(temp, rain)

    o = _predict_from_path(model, image_path)
    r = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "rain", "difference": abs(o - r), "passed": abs(o - r) < 0.2}


# =========================
# RUN ALL TESTS
# =========================

def run_all_tests(model, image_path):
    tests = [
        brightness_invariance_test,
        horizontal_flip_regression_test,
        translation_invariance_test,
        vertical_crop_test,
        pothole_noise_test,
        shadow_test,
        blur_test,
        contrast_test,
        occlusion_test,
        rain_test
    ]

    results = []
    for test in tests:
        try:
            results.append(test(model, image_path))
        except Exception as e:
            results.append({"test": test.__name__, "error": str(e)})

    return results
