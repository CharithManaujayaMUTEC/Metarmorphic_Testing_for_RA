import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ImageNet normalisation constants 
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _denorm(t: torch.Tensor) -> torch.Tensor:
    """Normalised tensor → [0, 1] float tensor."""
    return torch.clamp(t * _STD + _MEAN, 0.0, 1.0)

def _renorm(t: torch.Tensor) -> torch.Tensor:
    """[0, 1] float tensor → normalised tensor."""
    return (t - _MEAN) / _STD

def _infer(model: torch.nn.Module, tensor: torch.Tensor) -> float:
    """Run a single (3,H,W) normalised tensor through the model."""
    model.eval()
    with torch.no_grad():
        return float(model(tensor.unsqueeze(0)))

# MR-1  Horizontal Flip  →  steering sign inversion

def cnn_horizontal_flip_test(model, image: torch.Tensor):

    original_pred = _infer(model, image)
    flipped_pred  = _infer(model, TF.hflip(image))

    expected   = -original_pred
    difference = abs(flipped_pred - expected)
    tolerance  = 0.08 if abs(original_pred) > 0.1 else 0.15

    return {
        "test":             "cnn_horizontal_flip",
        "original_pred":    original_pred,
        "flipped_pred":     flipped_pred,
        "expected_flipped": expected,
        "difference":       difference,
        "tolerance":        tolerance,
        "passed":           difference < tolerance,
    }

# MR-2  Brightness shift  →  steering invariance

def cnn_brightness_invariance_test(model, image: torch.Tensor,
                                   brightness_factor: float = 1.4):

    img_01      = _denorm(image)
    bright_01   = TF.adjust_brightness(img_01, brightness_factor)
    bright_norm = _renorm(bright_01)

    original_pred = _infer(model, image)
    bright_pred   = _infer(model, bright_norm)

    difference = abs(bright_pred - original_pred)
    tolerance  = 0.08

    return {
        "test":              "cnn_brightness_invariance",
        "brightness_factor": brightness_factor,
        "original_pred":     original_pred,
        "bright_pred":       bright_pred,
        "difference":        difference,
        "tolerance":         tolerance,
        "passed":            difference < tolerance,
    }

# MR-3  Small rotation  →  bounded steering change

def cnn_rotation_consistency_test(model, image: torch.Tensor,
                                  angle_deg: float = 5.0):

    rotated_image = TF.rotate(image, angle_deg)
    original_pred = _infer(model, image)
    rotated_pred  = _infer(model, rotated_image)

    difference = abs(rotated_pred - original_pred)
    tolerance  = 0.20
    near_zero  = abs(original_pred) < 0.05
    same_sign  = (original_pred * rotated_pred >= 0) or near_zero
    passed     = (difference < tolerance) and same_sign

    return {
        "test":           "cnn_rotation_consistency",
        "angle_deg":      angle_deg,
        "original_pred":  original_pred,
        "rotated_pred":   rotated_pred,
        "difference":     difference,
        "same_sign":      same_sign,
        "tolerance":      tolerance,
        "passed":         passed,
    }

# MR-4  Gaussian blur  →  steering invariance

def cnn_blur_invariance_test(model, image: torch.Tensor,
                             kernel_size: int = 5):

    blurred_image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=2.0)
    original_pred = _infer(model, image)
    blurred_pred  = _infer(model, blurred_image)

    difference = abs(blurred_pred - original_pred)
    tolerance  = 0.08

    return {
        "test":          "cnn_blur_invariance",
        "kernel_size":   kernel_size,
        "original_pred": original_pred,
        "blurred_pred":  blurred_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-5  Horizontal translation  →  bounded proportional steering change

def cnn_translation_test(model, image: torch.Tensor,
                         shift_px: int = 10):

    _, H, W = image.shape
    padded  = F.pad(image.unsqueeze(0), (shift_px, 0, 0, 0), mode="replicate")
    shifted = padded[:, :, :, :W]
    shifted = F.interpolate(shifted, size=(H, W), mode="bilinear",
                            align_corners=False).squeeze(0)

    original_pred = _infer(model, image)
    shifted_pred  = _infer(model, shifted)

    difference = abs(shifted_pred - original_pred)
    tolerance  = 0.20

    return {
        "test":          "cnn_translation_consistency",
        "shift_px":      shift_px,
        "original_pred": original_pred,
        "shifted_pred":  shifted_pred,
        "difference":    difference,
        "tolerance":     tolerance,
        "passed":        difference < tolerance,
    }

# MR-6  Contrast adjustment  →  steering invariance

def cnn_contrast_invariance_test(model, image: torch.Tensor,
                                 contrast_factor: float = 1.5):

    img_01        = _denorm(image)
    contrast_01   = TF.adjust_contrast(img_01, contrast_factor)
    contrast_norm = _renorm(contrast_01)

    original_pred = _infer(model, image)
    contrast_pred = _infer(model, contrast_norm)

    difference = abs(contrast_pred - original_pred)
    tolerance  = 0.08

    return {
        "test":            "cnn_contrast_invariance",
        "contrast_factor": contrast_factor,
        "original_pred":   original_pred,
        "contrast_pred":   contrast_pred,
        "difference":      difference,
        "tolerance":       tolerance,
        "passed":          difference < tolerance,
    }