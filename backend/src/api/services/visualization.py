"""
src/api/services/visualization.py
==================================
Visualization helpers for SAR dual-polarization imagery, change masks,
and confidence heatmaps, with Base64 encoding for REST API responses.
"""

from __future__ import annotations

import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def array_to_base64_png(img_array: np.ndarray) -> str:
    """
    Convert a uint8 NumPy array (H, W), (H, W, 3), or (H, W, 4) to a base64 PNG string.
    """
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def sar_dualpol_to_rgb(sar_array: np.ndarray) -> np.ndarray:
    """
    Convert a 2-band normalized SAR array (2, H, W) in [0, 1] into a false-color RGB image (H, W, 3) uint8.
    
    Channels:
      - Red:   VV backscatter
      - Green: VH backscatter
      - Blue:  VV / (VH + 1e-3) ratio scaled
    """
    if sar_array.ndim != 3 or sar_array.shape[0] != 2:
        raise ValueError(f"Expected shape (2, H, W), got {sar_array.shape}")

    vv = np.nan_to_num(sar_array[0], nan=0.0, posinf=1.0, neginf=0.0)
    vh = np.nan_to_num(sar_array[1], nan=0.0, posinf=1.0, neginf=0.0)

    # Compute ratio channel for blue
    ratio = np.clip(vv / (vh + 0.1), 0.0, 3.0) / 3.0

    r = np.clip(vv * 255.0, 0, 255).astype(np.uint8)
    g = np.clip(vh * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(ratio * 255.0, 0, 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def generate_change_mask_image(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert binary mask (H, W) {0, 1} or bool into a high-contrast RGB image (H, W, 3).
    Change pixels are rendered in bright cyan-white or red.
    """
    h, w = binary_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Bright coral red for changes
    rgb[binary_mask > 0] = [255, 69, 58]
    return rgb


def generate_heatmap_image(prob_map: np.ndarray, colormap_name: str = "turbo") -> np.ndarray:
    """
    Render probability map [0.0, 1.0] (H, W) as an RGBA/RGB colormap heatmap.
    """
    prob_clean = np.clip(np.nan_to_num(prob_map, nan=0.0), 0.0, 1.0)
    cmap = plt.get_cmap(colormap_name)
    colored = cmap(prob_clean) # (H, W, 4) in [0, 1]
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return rgb


def generate_overlay_image(base_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a blended overlay of the change mask on top of the base SAR false-color image.
    """
    overlay = base_rgb.copy().astype(np.float32)
    mask_indices = binary_mask > 0

    # Highlight change in bright neon magenta/red [255, 50, 50]
    highlight_color = np.array([255, 50, 50], dtype=np.float32)
    overlay[mask_indices] = (1.0 - alpha) * overlay[mask_indices] + alpha * highlight_color
    return np.clip(overlay, 0, 255).astype(np.uint8)
