"""Train the forest U-Net on 2021 Sentinel-2 imagery with ESA WorldCover 2021 labels.

Usage:  uv run python scripts/train.py [--epochs 40]

Training patches (128 x 128 px = 1.28 km) are sampled at random from the 11 training blocks,
staying MARGIN pixels away from block edges. The checkpoint with the best validation IoU
(2 validation blocks) is kept. The 3 test blocks are never read here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forest.data import (MARGIN, TEST_BLOCKS, VAL_BLOCKS, block_bounds, features, forest_label, read,  # noqa: E402
                         valid_mask)
from forest.grid import TRAIN_YEAR  # noqa: E402
from forest.unet import UNet  # noqa: E402

P = 128


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_inputs(year: int, stats: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    s2 = read(f"s2_{year}.tif")
    x = features(s2)
    valid = valid_mask(s2)
    if stats is None:
        stats = {"mean": [float(x[i][valid].mean()) for i in range(len(x))],
                 "std": [float(x[i][valid].std() + 1e-6) for i in range(len(x))]}
    x = (x - np.array(stats["mean"], dtype=np.float32)[:, None, None]) / np.array(stats["std"], dtype=np.float32)[:, None, None]
    x[:, ~valid] = 0
    return x.astype(np.float32), valid, stats


@torch.no_grad()
def predict_full(model: torch.nn.Module, x: np.ndarray, dev: torch.device, tile: int = 256, overlap: int = 32) -> np.ndarray:
    """Forest probability for a whole image, tile by tile, keeping only each tile's centre."""
    model.eval()
    _, h, w = x.shape
    out = np.zeros((h, w), dtype=np.float32)
    step = tile - 2 * overlap
    pad = np.pad(x, ((0, 0), (overlap, overlap + tile), (overlap, overlap + tile)), mode="reflect")
    for y in range(0, h, step):
        for xx in range(0, w, step):
            t = torch.from_numpy(pad[:, y:y + tile, xx:xx + tile][None]).to(dev)
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            core = p[overlap:overlap + step, overlap:overlap + step]
            hh, ww = min(step, h - y), min(step, w - xx)
            out[y:y + hh, xx:xx + ww] = core[:hh, :ww]
    return out


def iou(pred: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    p, t = pred[mask], label[mask] > 0.5
    return float((p & t).sum() / max((p | t).sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patches", type=int, default=400, help="random patches per epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x, valid, stats = load_inputs(TRAIN_YEAR)
    y = forest_label(read("worldcover_2021.tif")[0])
    has_label = read("worldcover_2021.tif")[0] > 0
    usable = valid & has_label
    _, h, w = x.shape
    bounds = block_bounds(h, w)
    train_blocks = [b for b in bounds if b not in TEST_BLOCKS | VAL_BLOCKS]
    val_mask = np.zeros((h, w), bool)
    for b in VAL_BLOCKS:
        y0, y1, x0, x1 = bounds[b]
        val_mask[y0:y1, x0:x1] = True
    val_mask &= usable
    print(f"grid {h}x{w}; forest fraction {y[usable].mean():.1%}; {len(train_blocks)} train blocks")

    dev = device()
    model = UNet(in_ch=x.shape[0]).to(dev)
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ckpt = ROOT / "checkpoints" / "unet.pt"
    ckpt.parent.mkdir(exist_ok=True)

    def sample(n: int):
        xs, ys, ms = [], [], []
        for _ in range(n):
            y0, y1, x0, x1 = bounds[train_blocks[rng.integers(len(train_blocks))]]
            r = rng.integers(y0 + MARGIN, y1 - MARGIN - P)
            c = rng.integers(x0 + MARGIN, x1 - MARGIN - P)
            xp, yp, mp = x[:, r:r + P, c:c + P], y[r:r + P, c:c + P], usable[r:r + P, c:c + P]
            k = rng.integers(4)  # random rotation and flip: forests have no "up"
            xp, yp, mp = (np.rot90(a, k, axes=(-2, -1)) for a in (xp, yp, mp))
            if rng.random() < 0.5:
                xp, yp, mp = xp[..., ::-1], yp[..., ::-1], mp[..., ::-1]
            # brightness jitter, so other years' slightly different light doesn't surprise the model
            xp = xp * rng.uniform(0.9, 1.1) + rng.normal(0, 0.05)
            xs.append(xp.copy()), ys.append(yp.copy()), ms.append(mp.copy())
        return (torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ys))[:, None].to(dev),
                torch.from_numpy(np.stack(ms))[:, None].float().to(dev))

    best, history, t0 = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.patches // 16):
            xb, yb, mb = sample(16)
            logits = model(xb)
            bce = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * mb).sum() / mb.sum()
            p = torch.sigmoid(logits) * mb
            dice = 1 - (2 * (p * yb).sum() + 1) / (p.sum() + (yb * mb).sum() + 1)
            loss = bce + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        prob = predict_full(model, x, dev)
        v = iou(prob > 0.5, y, val_mask)
        history.append({"epoch": epoch, "loss": total / (args.patches // 16), "val_iou": v})
        flag = ""
        if v > best:
            best, flag = v, " *"
            torch.save({"model": model.state_dict(), "stats": stats, "epoch": epoch}, ckpt)
        print(f"epoch {epoch:2d}  loss {total / (args.patches // 16):.3f}  val forest IoU {v:.3f}{flag}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    (ROOT / "results" / "train_history.json").write_text(
        json.dumps({"history": history, "best_val_iou": best, "minutes": (time.time() - t0) / 60}, indent=1))


if __name__ == "__main__":
    main()
