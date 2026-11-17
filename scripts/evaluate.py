"""Forest maps for every year, change detection, and honest scores.

Usage:  uv run python scripts/evaluate.py

1. Forest segmentation (2021) on the 3 held-out TEST blocks, against ESA WorldCover 2021.
   Baseline: an NDVI threshold, with the threshold chosen on the training blocks.
2. Change detection: a pixel is "newly cleared" between two years if it was forest in the first
   and not forest in the second; patches smaller than 0.5 ha are dropped as noise.
   Checked against Hansen Global Forest Change loss years, an independent dataset (Landsat,
   30 m, different team and method) that played no part in training.
Writes data/processed/forest_prob_<year>.npy, data/processed/cleared_*.npy,
results/metrics.json and results/plots/*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from forest.data import block_bounds, features, forest_label, read, split_masks, valid_mask  # noqa: E402
from forest.grid import YEARS  # noqa: E402
from forest.unet import UNet  # noqa: E402
from train import device, load_inputs, predict_full  # noqa: E402

PROC = ROOT / "data" / "processed"
PLOTS = ROOT / "results" / "plots"
MIN_PATCH_PX = 50  # 0.5 ha at 10 m
INTERVALS = [(2019, 2021), (2021, 2024), (2019, 2024)]


def seg_scores(pred: np.ndarray, label: np.ndarray, mask: np.ndarray) -> dict:
    p, t = pred[mask], label[mask] > 0.5
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    return {"iou": round(float(tp / (tp + fp + fn)), 3), "precision": round(float(tp / (tp + fp)), 3),
            "recall": round(float(tp / (tp + fn)), 3), "accuracy": round(float((p == t).mean()), 3),
            "pixels": int(mask.sum())}


def clean(mask: np.ndarray, min_px: int = MIN_PATCH_PX) -> np.ndarray:
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def hansen_reference(lossyear: np.ndarray, tc2000: np.ndarray, y1: int, y2: int) -> tuple[np.ndarray, np.ndarray]:
    """Hansen 'cleared between y1 and y2' (loss in years y1+1..y2), and 'forest at y1' (>=30% cover in
    2000, no loss up to y1). Loss codes are years since 2000."""
    lo, hi = y1 - 2000 + 1, y2 - 2000
    forest_at_y1 = (tc2000 >= 30) & ((lossyear == 0) | (lossyear >= lo))
    return (lossyear >= lo) & (lossyear <= hi), forest_at_y1


def change_scores(ours: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    """Pixel agreement with 1-pixel tolerance (Hansen is 30 m, we are 10 m, edges never line up),
    plus how many of our detected patches overlap a Hansen loss patch."""
    ref_d = ndimage.binary_dilation(ref, iterations=1)
    ours_d = ndimage.binary_dilation(ours, iterations=1)
    o, r = ours & mask, ref & mask
    precision = (o & ref_d).sum() / max(o.sum(), 1)
    recall = (r & ours_d).sum() / max(r.sum(), 1)
    lab, n = ndimage.label(o)
    hit = ndimage.mean(ref_d, lab, range(1, n + 1)) if n else np.array([])
    return {"precision": round(float(precision), 3), "recall": round(float(recall), 3),
            "f1": round(float(2 * precision * recall / max(precision + recall, 1e-9)), 3),
            "our_area_ha": round(float(o.sum() / 100), 1), "hansen_area_ha": round(float(r.sum() / 100), 1),
            "our_patches": int(n), "patches_mostly_confirmed": round(float((hit >= 0.5).mean()), 3) if n else None}


def rgb(year: int, gain: float = 3.5) -> np.ndarray:
    s2 = read(f"s2_{year}.tif")
    return np.clip(np.stack([s2[2], s2[1], s2[0]], -1) * gain, 0, 1)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    dev = device()
    ck = torch.load(ROOT / "checkpoints" / "unet.pt", map_location="cpu")
    model = UNet(in_ch=8).to(dev)
    model.load_state_dict(ck["model"])

    wc = read("worldcover_2021.tif")[0]
    label = forest_label(wc)
    lossyear, tc2000 = read("hansen_lossyear.tif")[0], read("hansen_treecover2000.tif")[0]
    h, w = wc.shape
    splits = split_masks(h, w)

    prob, ndvi, valid = {}, {}, {}
    for y in YEARS:
        x, valid[y], _ = load_inputs(y, ck["stats"])
        prob[y] = predict_full(model, x, dev)
        np.save(PROC / f"forest_prob_{y}.npy", prob[y])
        ndvi[y] = features(read(f"s2_{y}.tif"))[6]
    all_valid = np.logical_and.reduce([valid[y] for y in YEARS])
    labelled = all_valid & (wc > 0)

    # ---- 1. segmentation, 2021 ----
    tr = splits["train"] & labelled
    thresholds = np.linspace(0.3, 0.9, 61)
    ious = [seg_scores(ndvi[2021] > t, label, tr)["iou"] for t in thresholds]
    t_ndvi = float(thresholds[int(np.argmax(ious))])
    seg = {}
    for split in ("train", "val", "test"):
        m = splits[split] & labelled
        seg[split] = {"unet": seg_scores(prob[2021] > 0.5, label, m),
                      f"ndvi>{t_ndvi:.2f}": seg_scores(ndvi[2021] > t_ndvi, label, m)}
    print("segmentation (2021 vs WorldCover):", json.dumps(seg["test"]))

    # ---- 2. change detection ----
    change = {}
    for y1, y2 in INTERVALS:
        ref, forest_y1 = hansen_reference(lossyear, tc2000, y1, y2)
        ours = clean((prob[y1] > 0.5) & (prob[y2] < 0.5) & all_valid)
        base = clean((ndvi[y1] > t_ndvi) & (ndvi[y2] <= t_ndvi) & all_valid)
        np.save(PROC / f"cleared_{y1}_{y2}.npy", ours)
        # Hansen keeps ONE loss year per pixel, so land it already marked as lost before y1 can never
        # be in its reference, even if regrowth there is cleared again. Scoring only on land Hansen
        # still considered forest at y1 asks a question both datasets can answer.
        intact = all_valid & forest_y1
        prior_loss = (lossyear >= 1) & (lossyear <= y1 - 2000)
        ref_d, ours_d = ndimage.binary_dilation(ref), ndimage.binary_dilation(ours)
        only_ours, only_ref = ours & ~ref_d & all_valid, ref & ~ours_d & all_valid
        change[f"{y1}-{y2}"] = {
            "unet_whole_area": change_scores(ours, ref, all_valid),
            "unet_on_hansen_intact_forest": change_scores(ours, ref, intact),
            "unet_test_blocks": change_scores(ours, ref, all_valid & splits["test"]),
            "ndvi_whole_area": change_scores(base, ref, all_valid),
            "ndvi_on_hansen_intact_forest": change_scores(base, ref, intact),
            "unet_only_share_on_previously_cleared_land": round(float(prior_loss[only_ours].mean()), 3),
            "hansen_only_share_by_loss_year": {str(2000 + int(v)): round(float((lossyear[only_ref] == v).mean()), 3)
                                               for v in range(y1 - 2000 + 1, y2 - 2000 + 1)},
            "hansen_only_mean_unet_forest_prob_at_y2": round(float(prob[y2][only_ref].mean()), 3),
        }
        print(f"change {y1}-{y2}:", json.dumps(change[f"{y1}-{y2}"]["unet_whole_area"]))

    # Forest area per year (U-Net) vs Hansen forest
    area = {y: round(float((prob[y] > 0.5)[all_valid].mean()), 3) for y in YEARS}
    hansen_area = {y: round(float(hansen_reference(lossyear, tc2000, y, y)[1][all_valid].mean()), 3) for y in YEARS}

    metrics = {"ndvi_threshold": t_ndvi, "segmentation_2021_vs_worldcover": seg, "change_vs_hansen": change,
               "forest_fraction_unet": area, "forest_fraction_hansen": hansen_area,
               "checkpoint_epoch": ck["epoch"], "min_patch_ha": MIN_PATCH_PX / 100}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("forest fraction U-Net:", area, " Hansen:", hansen_area)

    # ---- plots ----
    bounds = block_bounds(h, w)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    for row, (r, c) in enumerate([(0, 3), (1, 1)]):  # two of the three test blocks
        y0, y1_, x0, x1 = bounds[(r, c)]
        crop = (slice(y0, y1_), slice(x0, x1))
        err = np.zeros((y1_ - y0, x1 - x0, 3))
        p, t = prob[2021][crop] > 0.5, label[crop] > 0.5
        err[p & ~t] = [1, 0.3, 0.3]
        err[~p & t] = [0.3, 0.5, 1]
        err[p & t] = [0.85, 0.85, 0.85]
        for ax, img, title in zip(axes[row], [rgb(2021)[crop], t, p, err],
                                  ["Sentinel-2, Jul 2021", "WorldCover 2021 'tree cover'", "U-Net forest",
                                   "red: U-Net only, blue: WorldCover only\ngrey: both forest, black: both non-forest"]):
            ax.imshow(img, cmap="Greens" if img.ndim == 2 else None, interpolation="nearest")
            ax.set_title(f"{title} (test block {r},{c})", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS / "segmentation_test_blocks.png", dpi=110)
    plt.close(fig)

    ref, _ = hansen_reference(lossyear, tc2000, 2019, 2024)
    ours = np.load(PROC / "cleared_2019_2024.npy")
    overlay = np.zeros((h, w, 3))
    overlay[ours & ref] = [1, 1, 0]
    overlay[ours & ~ref] = [1, 0.2, 0.2]
    overlay[~ours & ref] = [0.2, 0.6, 1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, img, title in zip(axes, [rgb(2019), rgb(2024), overlay],
                              ["Sentinel-2, dry season 2019", "Sentinel-2, dry season 2024",
                               "Cleared 2019→2024. yellow: U-Net and Hansen agree,\nred: U-Net only, blue: Hansen only"]):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for (r, c), (y0, y1_, x0, x1) in bounds.items():
        if (r, c) in {(0, 3), (1, 1), (3, 0)}:
            axes[2].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1_ - y0, fill=False, ec="white", lw=1, ls="--"))
    fig.tight_layout()
    fig.savefig(PLOTS / "change_2019_2024.png", dpi=110)
    plt.close(fig)

    hist = json.loads((ROOT / "results" / "train_history.json").read_text())["history"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot([e["epoch"] for e in hist], [e["val_iou"] for e in hist])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation forest IoU")
    fig.tight_layout()
    fig.savefig(PLOTS / "val_curve.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
