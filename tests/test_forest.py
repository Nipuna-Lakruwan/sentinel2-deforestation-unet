import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import change_scores, clean, hansen_reference  # noqa: E402
from forest.data import TEST_BLOCKS, VAL_BLOCKS, features, split_masks  # noqa: E402
from forest.unet import UNet  # noqa: E402


def test_splits_are_disjoint_and_cover_everything():
    s = split_masks(103, 97)
    total = s["train"].astype(int) + s["val"] + s["test"]
    assert (total == 1).all()
    assert not TEST_BLOCKS & VAL_BLOCKS


def test_ndvi_high_for_vegetation():
    s2 = np.zeros((6, 1, 2), np.float32)
    s2[2] = [[0.03, 0.15]]  # red: low for forest, high for bare soil
    s2[3] = [[0.35, 0.20]]  # NIR: high for forest
    ndvi = features(s2)[6, 0]
    assert ndvi[0] > 0.8 and ndvi[1] < 0.2


def test_unet_keeps_size():
    assert UNet(in_ch=8)(torch.zeros(2, 8, 64, 64)).shape == (2, 1, 64, 64)


def test_clean_drops_small_patches():
    m = np.zeros((20, 20), bool)
    m[0:10, 0:10] = True  # 100 px, kept
    m[15, 15] = True      # 1 px, dropped
    out = clean(m, min_px=50)
    assert out.sum() == 100


def test_hansen_reference_years():
    loss = np.array([[0, 19, 20, 24, 5]], np.uint8)  # 0 = no loss; codes are years since 2000
    tc = np.full_like(loss, 80)
    ref, forest = hansen_reference(loss, tc, 2019, 2024)
    assert ref.tolist() == [[False, False, True, True, False]]
    assert forest.tolist() == [[True, False, True, True, False]]  # lost in 2019 or 2005 -> not forest in 2019


def test_change_scores_tolerate_one_pixel_shift():
    ref = np.zeros((10, 10), bool)
    ref[2:6, 2:6] = True
    ours = np.roll(ref, 1, axis=1)
    s = change_scores(ours, ref, np.ones_like(ref))
    assert s["precision"] == 1.0 and s["recall"] == 1.0


class _Item:
    def __init__(self, **props):
        self.properties = props


def test_reflectance_offset_not_applied_twice():
    from download_data import reflectance_offset

    # Old processing: no offset in the data.
    assert reflectance_offset(_Item(**{"s2:processing_baseline": "02.13"})) == 0.0
    # New processing, offset still in the data: remove it.
    assert reflectance_offset(_Item(**{"s2:processing_baseline": "05.10"})) == 0.1
    # New processing, Earth Search already removed it: don't subtract again.
    assert reflectance_offset(_Item(**{"s2:processing_baseline": "05.10", "earthsearch:boa_offset_applied": True})) == 0.0
