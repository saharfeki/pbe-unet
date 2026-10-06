import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class MedicalDataSets(Dataset):
    """BUSI loader.

    Expects:
        <data_root>/images/<case>.png
        <data_root>/masks/0/<case>_mask.png      (built by prepare_busi.py)
    and a text file with one case name per line (split_file).

    Boundary GT = 3x3 morphological gradient of the binary mask, computed AFTER
    augmentation + resize so it always matches the mask the model is trained on.
    """

    def __init__(self, data_root, split_file, transform=None):
        self.data_root = data_root
        self.transform = transform
        with open(split_file, "r") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]
        print("total {} samples from {}".format(len(self.sample_list), split_file))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = os.path.join(self.data_root, "images", case + ".png")
        mask_path = os.path.join(self.data_root, "masks", "0", case + "_mask.png")

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError("missing image or mask for case: " + case)
        # BGR -> RGB so the ImageNet mean/std in A.Normalize match the channel order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out = self.transform(image=image, mask=mask)
        # A.Normalize already outputs standardized float32 -> do NOT divide by 255 again
        image = out["image"].astype(np.float32).transpose(2, 0, 1)
        mask = (out["mask"] > 127).astype(np.uint8)  # (H, W), values {0, 1}

        k = np.ones((3, 3), np.uint8)
        boundary = cv2.dilate(mask, k) - cv2.erode(mask, k)

        return {
            "image": image,                                   # (3, H, W) float32
            "label": mask[None].astype(np.float32),           # (1, H, W) {0, 1}
            "boundary": boundary[None].astype(np.float32),    # (1, H, W) {0, 1}
            "case": case,
        }