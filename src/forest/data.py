"""Loading the rasters, input features, and the spatial train/val/test split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
N_BLOCKS = 4  # the area is cut into a 4 x 4 grid of ~5.5 km blocks
# Spatial split. Neighbouring pixels are nearly identical, so a random pixel split would leak.
# Whole blocks are held out instead, spread across the area so test isn't one landscape type.
TEST_BLOCKS = {(0, 3), (1, 1), (3, 0)}
VAL_BLOCKS = {(2, 3), (3, 2)}
MARGIN = 16  # pixels at each block edge that are excluded from training patches


def read(name: str) -> np.ndarray:
    with rasterio.open(RAW / name) as src:
        return src.read()


def features(s2: np.ndarray) -> np.ndarray:
    """6 reflectance bands -> 8 channels: the bands plus two classic indices.

    NDVI = (NIR - red) / (NIR + red): living leaves reflect NIR strongly and absorb red.
    NBR  = (NIR - SWIR2) / (NIR + SWIR2): sensitive to burned and bare ground.
    """
    blue, green, red, nir, sw1, sw2 = s2
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    nbr = (nir - sw2) / (nir + sw2 + eps)
    return np.concatenate([s2, ndvi[None], nbr[None]]).astype(np.float32)


def valid_mask(s2: np.ndarray) -> np.ndarray:
    return (s2 > 0).all(0)


def forest_label(worldcover: np.ndarray) -> np.ndarray:
    """ESA WorldCover class 10 = tree cover. 0 = no data."""
    return (worldcover == 10).astype(np.float32)


def block_id(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.minimum(np.arange(h) * N_BLOCKS // h, N_BLOCKS - 1)
    cols = np.minimum(np.arange(w) * N_BLOCKS // w, N_BLOCKS - 1)
    return rows, cols


def split_masks(h: int, w: int) -> dict[str, np.ndarray]:
    rows, cols = block_id(h, w)
    br, bc = np.meshgrid(rows, cols, indexing="ij")
    which = np.full((h, w), "train", dtype=object)
    for r, c in TEST_BLOCKS:
        which[(br == r) & (bc == c)] = "test"
    for r, c in VAL_BLOCKS:
        which[(br == r) & (bc == c)] = "val"
    return {s: which == s for s in ("train", "val", "test")}


def block_bounds(h: int, w: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """(row, col) -> (y0, y1, x0, x1) pixel bounds of each block."""
    ys = [round(i * h / N_BLOCKS) for i in range(N_BLOCKS + 1)]
    xs = [round(i * w / N_BLOCKS) for i in range(N_BLOCKS + 1)]
    return {(r, c): (ys[r], ys[r + 1], xs[c], xs[c + 1]) for r in range(N_BLOCKS) for c in range(N_BLOCKS)}
