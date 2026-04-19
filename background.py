import math
import numpy as np
import cv2
import mediapipe as mp

_mp_selfie = mp.solutions.selfie_segmentation
_segmenter = _mp_selfie.SelfieSegmentation(model_selection=1)


def segment_person(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = _segmenter.process(rgb)
    mask = result.segmentation_mask
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return np.clip(mask, 0.0, 1.0)


def make_dj_background(shape, rms: float, gesture_a: str | None = None, gesture_b: str | None = None) -> np.ndarray:
    """Center red glow fading to black at edges. Spotlight effect. Reactive to RMS."""
    h, w = shape[:2]

    bg = np.zeros((h, w, 3), dtype=np.float32)

    pulse = 1.0 + rms * 1.5

    # Radial red glow centered, fading outward to pure black
    cx, cy = w // 2, h // 2
    max_r = math.sqrt(cx * cx + cy * cy)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_r

    # Steep falloff so edges go true black
    core = np.power(np.clip(1.0 - dist * 1.1, 0, 1), 2.2)
    bg[:, :, 2] = core * 110 * pulse  # red channel

    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # Scanlines (darken every 3 rows slightly)
    bg[::3] = (bg[::3].astype(np.int16) - 4).clip(0, 255).astype(np.uint8)

    # Light noise to avoid banding
    noise = np.random.randint(-3, 4, size=(h, w, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg


def composite_person_on_bg(frame: np.ndarray, person_mask: np.ndarray, bg: np.ndarray) -> np.ndarray:
    mask3 = np.stack([person_mask] * 3, axis=-1)
    out = (frame.astype(np.float32) * mask3 + bg.astype(np.float32) * (1 - mask3)).astype(np.uint8)
    return out
