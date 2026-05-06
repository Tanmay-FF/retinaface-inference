"""
ArcFace 5-point face alignment.

Mirrors insightface.utils.face_align (estimate_norm + norm_crop) bit-for-bit:
  - same canonical destination template (arcface_dst)
  - same image_size scaling rules (multiples of 112 or 128)
  - same closed-form 2D similarity transform (Umeyama 1991, identical to
    skimage.transform.SimilarityTransform which is what InsightFace calls)
  - same cv2.warpAffine call (default INTER_LINEAR, BORDER_CONSTANT, value 0)

Aligned crops produced here are drop-in for ArcFace / AdaFace / MagFace /
CosFace / any InsightFace-compatible recognizer.
"""

import cv2
import numpy as np


# Canonical InsightFace destination points for a 112x112 ArcFace crop.
# Exact bit-for-bit match with insightface/utils/face_align.py:arcface_dst.
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# Per-size cache for scaled destination templates so repeated calls don't
# re-multiply the constant template.
_DST_CACHE: dict = {}


def _get_arcface_dst(image_size: int) -> np.ndarray:
    """
    Scaled ArcFace template for a square output of size `image_size`.

    Replicates insightface.utils.face_align.estimate_norm:
      - image_size % 112 == 0  →  ratio = size/112, diff_x = 0
      - image_size % 128 == 0  →  ratio = size/128, diff_x = 8*ratio
        (centers the 96-wide ArcFace template inside a 128-wide canvas)
    """
    cached = _DST_CACHE.get(image_size)
    if cached is not None:
        return cached

    if image_size % 112 == 0:
        ratio = image_size / 112.0
        diff_x = 0.0
    elif image_size % 128 == 0:
        ratio = image_size / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError(
            f"image_size={image_size} is not a multiple of 112 or 128. "
            f"InsightFace's template scaling only defines these two families."
        )

    dst = ARCFACE_DST * ratio
    if diff_x:
        dst[:, 0] += diff_x
    dst.flags.writeable = False
    _DST_CACHE[image_size] = dst
    return dst


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Closed-form 2D similarity transform fitting src → dst (Umeyama 1991).
    Numerically identical to skimage.transform.SimilarityTransform.estimate,
    which is the routine InsightFace calls. Returns a 2x3 float32 affine.
    """
    src = src.astype(np.float64, copy=False)
    dst = dst.astype(np.float64, copy=False)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    n = src.shape[0]
    A = dst_demean.T @ src_demean / n
    U, S, Vt = np.linalg.svd(A)

    # Reflection fix: if the implied rotation flips chirality, negate the
    # last singular component. Equivalent to R = U @ diag(1, -1) @ Vt and
    # scale = (S[0] - S[1]) / var_src.
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, -1] *= -1.0
        s_signed = S[0] - S[1]
    else:
        s_signed = S[0] + S[1]

    R = U @ Vt
    var_src = (src_demean * src_demean).sum() / n
    scale = s_signed / var_src

    M = np.empty((2, 3), dtype=np.float32)
    M[:, :2] = scale * R
    M[:, 2] = dst_mean - scale * (R @ src_mean)
    return M


def estimate_norm(landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Return the 2x3 similarity matrix that maps the 5 detected landmarks onto
    the ArcFace canonical template at the requested image_size.

    Mirrors insightface.utils.face_align.estimate_norm.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    if landmarks.shape != (5, 2):
        raise ValueError(f"landmarks must be (5, 2) or length 10, got {landmarks.shape}")
    dst = _get_arcface_dst(image_size)
    return _umeyama_similarity(landmarks, dst)


def norm_crop(image: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Crop and align a face using the ArcFace 5-point similarity warp.
    Mirrors insightface.utils.face_align.norm_crop.

    Args:
        image: source BGR image (H, W, 3), uint8.
        landmarks: (5, 2) or length-10 (x, y) coords in the order the model
            emits (eye, eye, nose, mouth corner, mouth corner).
        image_size: side length of the square aligned crop. Must be a
            multiple of 112 (default) or 128.

    Returns:
        Aligned BGR crop, (image_size, image_size, 3), uint8.
    """
    M = estimate_norm(landmarks, image_size=image_size)
    # InsightFace uses the cv2 defaults (INTER_LINEAR, BORDER_CONSTANT, 0).
    return cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)


# Backwards-compatible alias for any caller still using `align_face`.
align_face = norm_crop
