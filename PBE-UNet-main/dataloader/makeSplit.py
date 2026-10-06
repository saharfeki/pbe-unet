"""Create the BUSI train/validation/test case lists."""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_BASE = Path(__file__).resolve().parents[1] / "data" / "busi"


def stratified_split(items, labels, test_size, seed):
    """Split items while preserving class proportions, using only the stdlib."""
    groups = defaultdict(list)
    for item, label in zip(items, labels):
        groups[label].append(item)

    target_size = math.ceil(len(items) * test_size)
    allocations = {}
    fractions = {}
    for label, group in groups.items():
        exact = len(group) * test_size
        allocations[label] = math.floor(exact)
        fractions[label] = exact - allocations[label]

    remaining = target_size - sum(allocations.values())
    for label in sorted(groups, key=lambda key: (-fractions[key], str(key)))[:remaining]:
        allocations[label] += 1

    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(groups, key=str):
        group = groups[label]
        if len(group) < 2:
            raise ValueError(f"Need at least two cases in each class; {label!r} has {len(group)}")
        rng.shuffle(group)
        count = min(max(allocations[label], 1), len(group) - 1)
        test.extend(group[:count])
        train.extend(group[count:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def save(base, name, items):
    path = base / name
    path.write_text("\n".join(items) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} cases to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help=f"BUSI data folder containing images/ (default: {DEFAULT_BASE})",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()
    base = args.base_dir.resolve()
    image_dir = base / "images"
    if not image_dir.is_dir():
        parser.error(f"image folder not found: {image_dir}")

    cases = sorted(path.stem for path in image_dir.glob("*.png") if not path.stem.startswith("normal"))
    labels = [0 if case.startswith("benign") else 1 for case in cases]
    if not cases:
        parser.error(f"no PNG cases found in {image_dir} (normal cases are excluded)")
    print(f"{len(cases)} cases: {labels.count(0)} benign, {labels.count(1)} malignant")

    # Protocol B: 70 / 10 / 20, stratified.
    trainval, test = stratified_split(cases, labels, test_size=0.2, seed=args.seed)
    trainval_labels = [0 if case.startswith("benign") else 1 for case in trainval]
    train, val = stratified_split(trainval, trainval_labels, test_size=0.125, seed=args.seed)
    save(base, "busi_train.txt", train)
    save(base, "busi_val.txt", val)
    save(base, "busi_test.txt", test)

    # Protocol A: paper's 8:2 train/validation split.
    train82, val82 = stratified_split(cases, labels, test_size=0.2, seed=args.seed)
    save(base, "busi_train82_new.txt", train82)
    save(base, "busi_val82_new.txt", val82)


if __name__ == "__main__":
    main()
