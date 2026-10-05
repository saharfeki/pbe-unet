import os
import cv2
from torch.utils.data import Dataset
import numpy as np


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="",
            val_file_dir="",
    ):

        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:

                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in
                                self.sample_list]


        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = os.path.join(self._base_dir, 'images', case + '.png')
        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '_mask.png'), cv2.IMREAD_GRAYSCALE)[
            ..., None]
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(label, kernel, iterations=1)
        dilated = cv2.dilate(label, kernel, iterations=1)
        boundary = dilated - eroded
        boundary = boundary[..., None]

        augmented = self.transform(image=image, mask=label, boundary=boundary)
        image = augmented['image']
        label = augmented['mask']
        boundary = augmented['boundary']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        boundary = boundary.astype('float32') / 255
        boundary = boundary.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "boundary": boundary, 'case': case, 'image_path': image_path}
        return sample
