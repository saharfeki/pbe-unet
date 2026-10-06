import numpy as np
import torch
from medpy import metric


@torch.no_grad()
def fast_iou_dice(output, target, eps=1e-5):
    """Cheap per-batch IoU / Dice on the GPU (mean of per-image values). For training logs only."""
    pred = torch.sigmoid(output) > 0.5
    gt = target > 0.5
    pred_f, gt_f = pred.flatten(1), gt.flatten(1)
    inter = (pred_f & gt_f).sum(1).float()
    union = (pred_f | gt_f).sum(1).float()
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred_f.sum(1).float() + gt_f.sum(1).float() + eps)
    return iou.mean().item(), dice.mean().item()


@torch.no_grad()
def per_image_metrics(output, target, with_hd=True, eps=1e-6):
    """Full metrics for every image in the batch. Returns a list of dicts.

    HD95 is in PIXELS (at the resolution of the tensors passed in, i.e. 256x256).
    It is NaN when the prediction or the ground truth is empty, or when with_hd=False.
    Aggregate with a nan-aware mean and report how many images were NaN.
    """
    pred = (torch.sigmoid(output) > 0.5).cpu().numpy().astype(bool)
    gt = (target > 0.5).cpu().numpy().astype(bool)
    rows = []
    for p, g in zip(pred, gt):
        p, g = p[0], g[0]  # (H, W)
        tp = float((p & g).sum())
        fp = float((p & ~g).sum())
        fn = float((~p & g).sum())
        tn = float((~p & ~g).sum())

        iou = (tp + 1e-5) / (tp + fp + fn + 1e-5)
        dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        specificity = tn / (tn + fp + eps)
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * recall * precision / (recall + precision + eps)

        hd95 = np.nan
        if with_hd and p.any() and g.any():
            hd95 = float(metric.binary.hd95(p, g))

        rows.append(dict(iou=iou, dice=dice, recall=recall, precision=precision,
                         specificity=specificity, acc=acc, f1=f1, hd95=hd95))
    return rows


def summarize(rows):
    """Mean of every metric over images; HD95 uses only non-NaN images."""
    out = {}
    for k in ["iou", "dice", "recall", "precision", "specificity", "acc", "f1"]:
        out[k] = float(np.mean([r[k] for r in rows]))
    hd = [r["hd95"] for r in rows if not np.isnan(r["hd95"])]
    out["hd95"] = float(np.mean(hd)) if hd else float("nan")
    out["hd95_n_nan"] = int(len(rows) - len(hd))
    return out