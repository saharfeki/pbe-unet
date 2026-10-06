import os
import csv
import json
import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A

from dataloader.dataset import MedicalDataSets
import utils.losses_boundary as losses
from utils.metrics import fast_iou_dice, per_image_metrics, summarize
from network.PBEUNet import PBEUNet


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "busi"


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT),
                   help="BUSI folder containing images/ and masks/0/")
    p.add_argument("--train_list", default=None,
                   help="training case-list file (default: <data_root>/busi_train.txt)")
    p.add_argument("--val_list", default=None,
                   help="validation case-list file (default: <data_root>/busi_val.txt)")
    p.add_argument("--base_lr", type=float, default=1e-3)   # paper: 0.001
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lambda2", type=float, default=0.7)    # boundary loss weight (0 = no boundary supervision)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hd_every", type=int, default=10, help="compute HD95 every N epochs (and at the last)")
    p.add_argument("--out_dir", default="runs/pbeunet_seed41")
    p.add_argument("--resume", action="store_true", help="resume from <out_dir>/last.pth if it exists")
    args = p.parse_args()
    args.train_list = args.train_list or os.path.join(args.data_root, "busi_train.txt")
    args.val_list = args.val_list or os.path.join(args.data_root, "busi_val.txt")
    return args


def main():
    args = get_args()
    for required_dir in (os.path.join(args.data_root, "images"),
                         os.path.join(args.data_root, "masks", "0")):
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(
                "Dataset folder not found: {}. Set --data_root to the BUSI folder "
                "containing images/ and masks/0/.".format(required_dir)
            )
    for split_file in (args.train_list, args.val_list):
        if not os.path.isfile(split_file):
            raise FileNotFoundError("Split list not found: {}".format(split_file))

    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_tf = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(256, 256),
        A.Normalize(),
    ])
    val_tf = A.Compose([A.Resize(256, 256), A.Normalize()])

    train_set = MedicalDataSets(args.data_root, args.train_list, train_tf)
    val_set = MedicalDataSets(args.data_root, args.val_list, val_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = PBEUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("params: {:.2f} M (paper reports 4.26 M)".format(n_params / 1e6))

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    criterion = losses.MultiTaskLoss(alpha=args.lambda2)

    max_iter = len(train_loader) * args.epochs
    iter_num, start_epoch, best_iou = 0, 0, 0.0
    last_path = os.path.join(args.out_dir, "last.pth")
    best_path = os.path.join(args.out_dir, "best_val_iou_model.pth")
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    header = ["epoch", "lr", "train_loss", "train_iou", "train_dice", "val_loss", "val_iou", "val_dice",
              "recall", "precision", "f1", "specificity", "acc", "hd95", "hd95_n_nan", "time_s"]

    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, iter_num, best_iou = ckpt["epoch"], ckpt["iter_num"], ckpt["best_iou"]
        print("resumed from epoch", start_epoch)
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ---------------- train ----------------
        model.train()
        tr_loss = tr_iou = tr_dice = 0.0
        n_seen = 0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            lab = batch["label"].to(device, non_blocking=True)
            bnd = batch["boundary"].to(device, non_blocking=True)

            lr = args.base_lr * (1.0 - iter_num / max_iter) ** 0.9   # Poly decay, the only schedule
            for g in optimizer.param_groups:
                g["lr"] = lr

            out, bds = model(img)
            loss, _ = criterion(out, bds, lab, bnd)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iter_num += 1

            bs = img.size(0)
            iou, dice = fast_iou_dice(out.detach(), lab)
            tr_loss += loss.item() * bs
            tr_iou += iou * bs
            tr_dice += dice * bs
            n_seen += bs

        # ---------------- validate ----------------
        want_hd = ((epoch + 1) % args.hd_every == 0) or (epoch + 1 == args.epochs)
        model.eval()
        rows, val_loss, n_val = [], 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                lab = batch["label"].to(device)
                bnd = batch["boundary"].to(device)
                out, bds = model(img)
                loss, _ = criterion(out, bds, lab, bnd)
                val_loss += loss.item() * img.size(0)
                n_val += img.size(0)
                rows += per_image_metrics(out, lab, with_hd=want_hd)
        s = summarize(rows)
        val_loss /= n_val

        if s["iou"] > best_iou:   # kept for reference only; report the LAST epoch
            best_iou = s["iou"]
            torch.save(model.state_dict(), best_path)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "iter_num": iter_num, "best_iou": best_iou}, last_path)

        dt = time.time() - t0
        hd = s["hd95"] if want_hd else ""
        n_nan = s["hd95_n_nan"] if want_hd else ""
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, lr, tr_loss / n_seen, tr_iou / n_seen, tr_dice / n_seen,
                                    val_loss, s["iou"], s["dice"], s["recall"], s["precision"], s["f1"],
                                    s["specificity"], s["acc"], hd, n_nan, round(dt, 1)])
        print("epoch [{}/{}] lr {:.5f} | train loss {:.4f} dice {:.4f} | val loss {:.4f} iou {:.4f} dice {:.4f}"
              " | hd95 {} (nan imgs: {}) | {:.1f}s".format(
                  epoch + 1, args.epochs, lr, tr_loss / n_seen, tr_dice / n_seen, val_loss, s["iou"], s["dice"],
                  "{:.2f}".format(hd) if want_hd else "-", n_nan if want_hd else "-", dt))

    print("Training finished. Final model: {}".format(last_path))


if __name__ == "__main__":
    main()