"""
MAITRI 2.0 — Digital Image Processing (DIP) Enhancement Pipeline
Real-time illumination normalization, skin-tone invariance, and blur recovery.

Techniques implemented:
1. LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE) - lighting/shadow invariance
2. Adaptive Gamma Correction (LUT vectorized) - dark scene & melanin compensation
3. Spatial Domain Unsharp Masking - micro-expression edge recovery under motion blur
4. Laplacian Variance Quality Gating - prevents noisy predictions on severe blur
5. Landmark-based Affine Face Alignment - normalizes head tilt via eye-center geometry
   (Largest single accuracy boost; ~1-2ms overhead. Applied when MP landmarks available.)
"""

from typing import Dict, Optional, Tuple
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


def apply_face_alignment(
    bgr_img: np.ndarray,
    left_eye_center: Tuple[float, float],
    right_eye_center: Tuple[float, float],
) -> np.ndarray:
    """
    Landmark-based affine alignment: rotates the face image so the eye line
    is horizontal. This normalizes head tilt and dramatically improves FER
    model accuracy on angled faces.

    Technique: compute rotation angle θ from eye-center vectors, then apply
    cv2.getRotationMatrix2D + cv2.warpAffine around the face centroid.
    Overhead: ~1-2ms per frame (well within 24-30 FPS budget).

    Args:
        bgr_img:          The face image to align (can be full frame or cropped ROI)
        left_eye_center:  (x, y) pixel coords of left eye center
        right_eye_center: (x, y) pixel coords of right eye center

    Returns:
        aligned (np.ndarray): The rotation-corrected image (same size as input)
    """
    if bgr_img is None or bgr_img.size == 0:
        return bgr_img

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]

    # Angle in degrees; positive = counter-clockwise correction
    angle = float(np.degrees(np.arctan2(dy, dx)))

    # Only apply if tilt is meaningful (>1°) and not extreme (skip if >30°, likely bad landmark)
    if abs(angle) < 1.0 or abs(angle) > 30.0:
        return bgr_img

    h, w = bgr_img.shape[:2]
    center_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
    center_y = (left_eye_center[1] + right_eye_center[1]) / 2.0
    center = (center_x, center_y)

    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned = cv2.warpAffine(bgr_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    return aligned


def extract_eye_centers_from_landmarks(landmarks, img_w: int, img_h: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Extract left and right eye center coordinates from MediaPipe face landmarks.
    Uses the 6 EAR landmark indices (same as eye_module).
    Returns (left_eye_center, right_eye_center) in pixel coords, or None if landmarks invalid.

    MediaPipe left eye indices:  [362, 385, 387, 263, 373, 380]
    MediaPipe right eye indices: [33,  160, 158, 133, 153, 144]
    """
    _LEFT  = [362, 385, 387, 263, 373, 380]
    _RIGHT = [33,  160, 158, 133, 153, 144]

    try:
        lx = np.mean([landmarks[i].x * img_w for i in _LEFT])
        ly = np.mean([landmarks[i].y * img_h for i in _LEFT])
        rx = np.mean([landmarks[i].x * img_w for i in _RIGHT])
        ry = np.mean([landmarks[i].y * img_h for i in _RIGHT])
        return (float(lx), float(ly)), (float(rx), float(ry))
    except Exception:
        return None


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
