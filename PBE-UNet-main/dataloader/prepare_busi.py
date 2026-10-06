import os, glob, argparse, cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True)   # folder that contains benign/ and malignant/
p.add_argument("--out", required=True)
a = p.parse_args()

os.makedirs(f"{a.out}/images", exist_ok=True)
os.makedirs(f"{a.out}/masks/0", exist_ok=True)
n, multi, resized = 0, 0, 0
for cls in ["benign", "malignant"]:
    for img_path in sorted(glob.glob(f"{a.raw}/{cls}/*.png")):
        if "_mask" in os.path.basename(img_path):
            continue
        stem = os.path.splitext(img_path)[0]
        name = os.path.basename(stem)
        img = cv2.imread(img_path)
        m = np.zeros(img.shape[:2], np.uint8)
        mask_files = sorted(glob.glob(glob.escape(stem) + "_mask*.png"))
        multi += len(mask_files) > 1
        for mp in mask_files:
            mm = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mm.shape != m.shape:
                resized += 1
                mm = cv2.resize(mm, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = np.maximum(m, (mm > 127).astype(np.uint8) * 255)
        cv2.imwrite(f"{a.out}/images/{name}.png", img)
        cv2.imwrite(f"{a.out}/masks/0/{name}_mask.png", m)
        n += 1
print(f"{n} cases, {multi} with multiple masks, {resized} masks resized")