import numpy as np
import cv2


def extract_features(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.resize(img, (200, 66))
    h, w = img.shape

    # Threshold to isolate the white lane line
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    image_center = w / 2.0

    # Feature 1: Overall lane center offset
    col_sums = np.sum(binary, axis=0)
    if col_sums.max() > 0:
        col_indices = np.arange(w)
        lane_col = np.sum(col_indices * col_sums) / np.sum(col_sums)  # weighted mean
    else:
        lane_col = image_center
    lane_center_offset = (lane_col - image_center) / image_center  # [-1, 1]

    # Feature 2: Top-third offset (lookahead)
    top_third = binary[:h // 3, :]
    col_sums_top = np.sum(top_third, axis=0)
    if col_sums_top.max() > 0:
        col_indices = np.arange(w)
        top_col = np.sum(col_indices * col_sums_top) / np.sum(col_sums_top)
    else:
        top_col = image_center
    top_offset = (top_col - image_center) / image_center  # [-1, 1]

    # Feature 3: Bottom-third offset (current lane position) 
    bottom_third = binary[2 * h // 3:, :]
    col_sums_bot = np.sum(bottom_third, axis=0)
    if col_sums_bot.max() > 0:
        col_indices = np.arange(w)
        bot_col = np.sum(col_indices * col_sums_bot) / np.sum(col_sums_bot)
    else:
        bot_col = image_center
    bottom_offset = (bot_col - image_center) / image_center  # [-1, 1]

    # Feature 4: Curvature proxy (signed direction of curve) 
    curvature_proxy = top_offset - bottom_offset  # negative = left curve, positive = right

    # Feature 5: White pixel ratio 
    white_pixel_ratio = np.sum(binary > 0) / binary.size  # [0, 1]

    # Feature 6: Vertical spread of lane positions 
    row_positions = []
    for row in binary:
        nonzero = np.where(row > 0)[0]
        if len(nonzero) > 0:
            row_positions.append(np.mean(nonzero) / image_center - 1.0)
    if len(row_positions) > 1:
        vertical_spread = float(np.std(row_positions))
    else:
        vertical_spread = 0.0

    return np.array([
        lane_center_offset,
        top_offset,
        bottom_offset,
        curvature_proxy,
        white_pixel_ratio,
        vertical_spread
    ], dtype=np.float32)