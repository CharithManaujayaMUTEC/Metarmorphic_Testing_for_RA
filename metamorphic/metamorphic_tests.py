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
# BASIC TESTS
# =========================
def brightness_test(model, path):
    img = cv2.imread(path)
    bright = np.clip(img + 30, 0, 255).astype(np.uint8)

    temp = path.replace(".png", "_bright.png")
    cv2.imwrite(temp, bright)

    o = _predict_from_path(model, path)
    b = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "brightness", "difference": abs(o-b), "passed": abs(o-b) < 0.05}


def flip_test(model, path):
    img = cv2.imread(path)
    flipped = cv2.flip(img, 1)

    temp = path.replace(".png", "_flip.png")
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

    temp = path.replace(".png", "_shift.png")
    cv2.imwrite(temp, shifted)

    o = _predict_from_path(model, path)
    s = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "translation", "difference": abs(o-s), "passed": abs(o-s) < 0.15}


def crop_test(model, path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    cropped = img[int(h*0.2):, :]
    cropped = cv2.resize(cropped, (w, h))

    temp = path.replace(".png", "_crop.png")
    cv2.imwrite(temp, cropped)

    o = _predict_from_path(model, path)
    c = _predict_from_path(model, temp)
    os.remove(temp)

    same_sign = (np.sign(o) == np.sign(c)) or abs(o) < 0.01
    diff = abs(o - c)

    return {"test": "crop", "difference": diff, "passed": same_sign and diff < 0.2}


# =========================
# REAL-WORLD ROBUSTNESS TESTS
# =========================
def noise_test(model, path):
    img = cv2.imread(path)
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

    temp = path.replace(".png", "_noise.png")
    cv2.imwrite(temp, noisy)

    o = _predict_from_path(model, path)
    n = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "noise", "difference": abs(o-n), "passed": abs(o-n) < 0.15}


def shadow_test(model, path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    shadow = img.copy()
    shadow[:, :w//2] = (shadow[:, :w//2] * 0.5).astype(np.uint8)

    temp = path.replace(".png", "_shadow.png")
    cv2.imwrite(temp, shadow)

    o = _predict_from_path(model, path)
    s = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "shadow", "difference": abs(o-s), "passed": abs(o-s) < 0.15}


def blur_test(model, path):
    img = cv2.imread(path)
    blur = cv2.GaussianBlur(img, (5,5), 0)

    temp = path.replace(".png", "_blur.png")
    cv2.imwrite(temp, blur)

    o = _predict_from_path(model, path)
    b = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "blur", "difference": abs(o-b), "passed": abs(o-b) < 0.15}


def contrast_test(model, path):
    img = cv2.imread(path)
    contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    temp = path.replace(".png", "_contrast.png")
    cv2.imwrite(temp, contrast)

    o = _predict_from_path(model, path)
    c = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "contrast", "difference": abs(o-c), "passed": abs(o-c) < 0.15}


def occlusion_test(model, path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    occ = img.copy()
    cv2.rectangle(occ, (w//3, h//2), (w//2, h), (0,0,0), -1)

    temp = path.replace(".png", "_occ.png")
    cv2.imwrite(temp, occ)

    o = _predict_from_path(model, path)
    oc = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "occlusion", "difference": abs(o-oc), "passed": abs(o-oc) < 0.2}


def rain_test(model, path):
    img = cv2.imread(path)
    rain = img.copy()

    for _ in range(400):
        x = np.random.randint(0, rain.shape[1])
        y = np.random.randint(0, rain.shape[0])
        cv2.line(rain, (x,y), (x+1,y+5), (200,200,200), 1)

    temp = path.replace(".png", "_rain.png")
    cv2.imwrite(temp, rain)

    o = _predict_from_path(model, path)
    r = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "rain", "difference": abs(o-r), "passed": abs(o-r) < 0.2}


def rotation_test(model, path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1)
    rotated = cv2.warpAffine(img, M, (w, h))

    temp = path.replace(".png", "_rot.png")
    cv2.imwrite(temp, rotated)

    o = _predict_from_path(model, path)
    r = _predict_from_path(model, temp)
    os.remove(temp)

    return {"test": "rotation", "difference": abs(o-r), "passed": abs(o-r) < 0.15}


# =========================
# MASTER RUNNER
# =========================
def run_all_tests(model_reg, model_cnn, image_path, image_tensor):
    results = []

    # CNN
    results.append(cnn_horizontal_flip_test(model_cnn, image_tensor))

    # ALL regression tests
    tests = [
        brightness_test,
        flip_test,
        translation_test,
        crop_test,
        noise_test,
        shadow_test,
        blur_test,
        contrast_test,
        occlusion_test,
        rain_test,
        rotation_test
    ]

    for t in tests:
        try:
            results.append(t(model_reg, image_path))
        except Exception as e:
            results.append({"test": t.__name__, "error": str(e)})

    return results
