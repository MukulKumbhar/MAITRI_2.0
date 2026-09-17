"""
MAITRI 2.0 — Digital Image Processing (DIP) Enhancement Pipeline
Real-time illumination normalization, skin-tone invariance, and blur recovery.

Techniques implemented:
1. LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE) - lighting/shadow invariance
2. Adaptive Gamma Correction (LUT vectorized) - dark scene & melanin compensation
3. Spatial Domain Unsharp Masking - micro-expression edge recovery under motion blur
4. Laplacian Variance Quality Gating - prevents noisy predictions on severe blur
"""

from typing import Dict, Tuple
import cv2
import numpy as np


def compute_blur_metric(bgr_img: np.ndarray) -> float:
    """
    Computes the Laplacian variance of the image:
        Blur Index = Var(Laplacian(Grayscale(Image)))
    Higher values = sharp edges; values < 70.0 = substantial blur.
    """
    if bgr_img is None or bgr_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_adaptive_gamma(
    bgr_img: np.ndarray, target_mean: float = 120.0
) -> Tuple[np.ndarray, float]:
    """
    Dynamically adjusts gamma based on the image's mean luminance.
    If image is underexposed or in dark lighting (mean < 85),
    non-linearly lifts shadows without burning highlights.
    Uses pre-computed 256-entry lookup table (LUT) for sub-millisecond execution.
    """
    if bgr_img is None or bgr_img.size == 0:
        return bgr_img, 1.0

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    current_mean = float(np.mean(gray))

    # Only correct if notably dark (underexposed) or overly harsh highlights
    if current_mean < 85.0 and current_mean > 5.0:
        # Non-linear gamma calculation
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = float(np.clip(gamma, 0.45, 1.5))
    elif current_mean > 175.0:
        gamma = 1.25  # slight compression for overexposure
    else:
        gamma = 1.0   # normal lighting, no modification

    if abs(gamma - 1.0) < 0.05:
        return bgr_img, 1.0

    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    corrected = cv2.LUT(bgr_img, table)
    return corrected, gamma


def apply_clahe_lab(
    bgr_img: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) applied
    strictly to the L-channel (Luminance) in LAB color space.
    Preserves chromaticity (A, B channels) to guarantee skin-tone invariance
    and prevent unnatural color casts.
    """
    if bgr_img is None or bgr_img.size == 0:
        return bgr_img

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    bgr_img: np.ndarray, sigma: float = 1.0, strength: float = 1.2
) -> np.ndarray:
    """
    Spatial domain unsharp masking to recover fine facial contour boundaries
    (nasolabial folds, lip corners, eyelid margins) degraded by slight motion blur.
    Sharpened = (1 + strength) * Original - strength * GaussianBlur
    """
    if bgr_img is None or bgr_img.size == 0:
        return bgr_img

    blurred = cv2.GaussianBlur(bgr_img, (0, 0), sigma)
    sharpened = cv2.addWeighted(bgr_img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def enhance_face_patch(
    bgr_face: np.ndarray,
) -> Tuple[np.ndarray, float, bool, Dict[str, any]]:
    """
    Unified real-time DIP enhancement pipeline:
    1. Blur estimation via Laplacian variance
    2. Adaptive Gamma correction (for low light & shadow details)
    3. LAB-space CLAHE (for structural contrast & skin tone normalization)
    4. Unsharp masking (for micro-expression sharpening)

    Returns:
        enhanced_face (np.ndarray): The DIP-enhanced face image
        blur_score (float): Laplacian variance metric
        is_blurry (bool): True if blur exceeds acceptable threshold
        telemetry (dict): Diagnostic DIP telemetry metrics for live HUD
    """
    if bgr_face is None or bgr_face.size == 0:
        return bgr_face, 0.0, True, {"status": "empty"}

    # 1. Blur evaluation
    blur_score = compute_blur_metric(bgr_face)
    is_blurry = blur_score < 70.0

    # 2. Adaptive Gamma
    gamma_corrected, gamma_val = apply_adaptive_gamma(bgr_face, target_mean=120.0)

    # 3. CLAHE on L-channel in LAB space
    clahe_enhanced = apply_clahe_lab(gamma_corrected, clip_limit=2.5, tile_grid_size=(8, 8))

    # 4. Unsharp Masking
    final_enhanced = apply_unsharp_mask(clahe_enhanced, sigma=1.0, strength=1.1)

    telemetry = {
        "blur_score": round(blur_score, 1),
        "is_blurry": is_blurry,
        "gamma_applied": round(gamma_val, 2),
        "clahe_applied": True,
        "sharpen_applied": True,
    }

    return final_enhanced, blur_score, is_blurry, telemetry
