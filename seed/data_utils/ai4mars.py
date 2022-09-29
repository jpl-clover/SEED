"""
AI4Mars data directory structure:

ai4mars-dataset-merged-0.1
│   mer
└───msl
    └───images
        └───edr  <- these are called ECM in M2020
        │   *.JPG files <- images to run segmentation on
        │   NLA_397681455EDR_F0020000AUT_04096M1.JPG
        │
        └───mxy
        │   *.png files <- rover masks
        │   NLA_397681455MXY_F0020000AUT_04096M1.png
        │
        └───rng-30m
        │   *.png files <- 30m range masks
        │   NLA_397681455RNG_F0020000AUT_04096M1.png
        │
    └───labels
        └───test
            └───masked-gold-min1-100agree
            │   *.png files <- segmentation masks/labels
            │   NLA_602664178EDR_F0732958NCAM00271M1_merged.png                         
            │
            └───masked-gold-min2-100agree
            │   *.png files <- segmentation masks/labels
            │            
            └───masked-gold-min3-100agree
            │   *.png files <- segmentation masks/labels
        └───train
            │   *.png files <- segmentation masks/labels
            │   NLA_397681455EDR_F0020000AUT_04096M1.png     

"""

import os
import sys
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

missions = {"msl", "mer", "m2020", "m2020_ai4mars"}

classes = {
    "msl": {
        4: ["soil", "bedrock", "sand", "big_rock"],
        6: ["soil", "bedrock", "sand", "big_rock", "rover", "background"],
    },
    "m2020_ai4mars": ["soil", "bedrock", "sand", "big_rock"],
    "m2020": ["float_rock", "sand", "bedrock", "pebbles", "vein", "hill"],
    "msl_geology": [
        "ripple",
        "smooth",
        "smooth_w_rock",
        "murray_smooth",
        "rover_track",
        "murray_rough",
        "caprock",
    ],
}

default_test_sets = {
    "msl": "masked-gold-min3-100agree",
    "m2020": "masked-gold-min1-68agree",
    "m2020_ai4mars": "sols200-202",
}


class AI4MarsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        mission=None,
        images_with_labels_only=True,
        split="train",
        test_set="masked-gold-min3-100agree",
        image_ext="JPG",
        label_ext="png",
        png_null_value=255,  # this is the value in the PNG for unlabeled pixels
        return_labels=False,
        classes=None,
        transform=None,
        target_transform=None,
        train_pct=1.0,
        random_state=0,
    ):
        # if final folder in data_root is not a mission, then append mission to it:
        if mission is not None and os.path.split(data_root)[-1] not in missions:
            self.data_root = os.path.join(data_root, mission)
        else:
            self.data_root = data_root
        assert os.path.isdir(self.data_root), f"{self.data_root} is not a directory"

        (
            self.image_files,
            self.rover_mask_files,
            self.range_mask_files,
            self.label_files,
        ) = self.get_ai4mars_files(
            images_with_labels_only=images_with_labels_only,
            split=split,
            test_set=test_set,
            image_ext=image_ext,
            label_ext=label_ext,
            train_pct=train_pct,
            random_state=random_state,
        )

        self.mission = mission
        self.split = split
        self.test_set = test_set
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.png_null_value = png_null_value
        self.return_labels = return_labels
        self.transform = transform
        self.target_transform = target_transform

        if classes is None:
            self.classes = classes["ai4mars_six_classes"]  # defaults to 6 classes
        else:
            self.classes = classes
        self.num_classes = len(self.classes)

    ## Define properties for image-related paths
    @property
    def rover_mask_dir(self):
        return os.path.join(self.image_root, "mxy")

    @property
    def range_dir(self):
        return os.path.join(self.image_root, "rng-30m")

    @property
    def image_dir(self):
        return os.path.join(self.image_root, "edr")

    @property
    def image_root(self):
        """
        Image root contains the following subfolders:
        - mxy: rover mask images
        - rng-30m: range images
        - edr: actual images to use for segmentation
        Returns:
            str: Full path to the image root
        """
        return os.path.join(self.data_root, "images")

    ## Define properties for label-related paths
    # The labels directory is split into train and test. Test is further subdivided into:
    # masked-gold-min1-100agree, masked-gold-min2-100agree, masked-gold-min3-100agree
    @property
    def test_label_dir(self):
        return os.path.join(self.label_root, "test")

    @property
    def train_label_dir(self):
        return os.path.join(self.label_root, "train")

    @property
    def label_root(self):
        return os.path.join(self.data_root, "labels")

    def get_ai4mars_files(
        self,
        images_with_labels_only=True,
        split="both",
        test_set="masked-gold-min3-100agree",
        image_ext="JPG",
        label_ext="png",
        train_pct=1.0,
        random_state=0,
    ):
        images = self.get_image_files()
        rover_masks = self.get_rover_masks_files()
        range_masks = self.get_range_mask_files()
        labels = self.get_label_files(split, test_set)
        if images_with_labels_only:
            if train_pct < 1.0:
                # Shuffle and filter the labels, since we'll be looping through this
                # in order to get the images with labels
                rng = np.random.RandomState(random_state)
                rng.shuffle(labels)
                # TODO: think about whether to use len(labels) or len(images)
                labels = labels[: int(train_pct * len(labels))]
            image_files = []
            rover_mask_files = []
            range_mask_files = []
            label_files = []
            for label_file in tqdm(labels, desc="Filtering images with labels"):
                filename = os.path.basename(label_file).replace("_merged", "")
                image_file = os.path.join(
                    self.image_dir, filename.replace(label_ext, image_ext)
                )
                rover_mask_file = os.path.join(
                    self.rover_mask_dir, filename.replace("EDR", "MXY")
                )
                range_mask_file = os.path.join(
                    self.range_dir, filename.replace("EDR", "RNG")
                )
                if os.path.exists(image_file):
                    image_files.append(image_file)
                    rover_mask_files.append(
                        rover_mask_file if os.path.exists(rover_mask_file) else None
                    )
                    range_mask_files.append(
                        range_mask_file if os.path.exists(range_mask_file) else None
                    )
                    label_files.append(label_file)
            num_rover_masks = len(rover_mask_files) - sum(
                [x is None for x in rover_mask_files]
            )
            num_range_masks = len(range_mask_files) - sum(
                [x is None for x in range_mask_files]
            )
            print(
                f"Found {len(image_files)} images with labels from split {split};"
                f" {num_rover_masks} with rover masks and "
                f"{num_range_masks} with range masks"
            )
            return image_files, rover_mask_files, range_mask_files, label_files
        else:
            if train_pct < 1.0:
                rng = np.random.RandomState(random_state)
                rng.shuffle(images)
                images = images[: int(len(images) * train_pct)]
            return images, rover_masks, range_masks, labels

    def get_image_files(self):
        images = sorted(glob(os.path.join(self.image_dir, "*.JPG")))
        print(f"Found {len(images)} JPG images in {self.image_dir}")
        return images

    def get_rover_masks_files(self):
        rover_masks = sorted(glob(os.path.join(self.rover_mask_dir, "*.png")))
        print(f"Found {len(rover_masks)} PNG rover masks in {self.rover_mask_dir}")
        return rover_masks

    def get_range_mask_files(self):
        range_masks = sorted(glob(os.path.join(self.range_dir, "*.png")))
        print(f"Found {len(range_masks)} PNG range masks in {self.range_dir}")
        return range_masks

    def get_label_files(self, split="train", test_set="masked-gold-min3-100agree"):
        if split == "train":
            label_dir = self.train_label_dir
        elif split == "test":
            label_dir = os.path.join(self.test_label_dir, test_set)
            assert os.path.isdir(label_dir), f"{label_dir} is not a directory"
        elif split in ["both", "all"]:
            train_labels = self.get_label_files("train", test_set)
            test_labels = self.get_label_files("test", test_set)
            return train_labels + test_labels
        else:
            raise NotImplementedError(f'"{split}" is not a valid split')
        labels = sorted(glob(os.path.join(label_dir, "*.png")))
        print(f"Found {len(labels)} PNG labels in {label_dir}")
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx, return_labels=False):
        # TODO: check whether we need to load the label file (no need to load during pretraining)
        image_file = self.image_files[idx]
        rover_mask_file = self.rover_mask_files[idx]
        range_mask_file = self.range_mask_files[idx]
        label_file = self.label_files[idx]

        # load image file with cv2
        # ? Can we combine these two lines?
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            # convert to PIL image if not already
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = self.transform(image)

        if return_labels or self.return_labels:
            label = png_to_tensor(label_file)
            rover_mask = png_to_tensor(rover_mask_file) if rover_mask_file else None
            range_mask = png_to_tensor(range_mask_file) if range_mask_file else None

            if rover_mask is not None and "rover" in self.classes:
                # set pixels where rover is present to 4
                label[rover_mask == 1] = self.classes.index("rover")
            if range_mask is not None and "rover" in self.classes:
                # set pixels where rover is present to 5
                label[range_mask == 1] = self.classes.index("background")

            label[label == self.png_null_value] = len(self.classes)
            if self.target_transform is not None:
                label = self.target_transform(label)
                # rover_mask = None if rover_mask is None else self.target_transform(rover_mask)
                # range_mask = None if range_mask is None else self.target_transform(range_mask)
            return image, label  # , rover_mask, range_mask
        else:
            return image, None

    def plot_item(self, idx):
        image, label, rover_mask, range_mask = self.__getitem__(idx, return_labels=True)
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title("Image")
        if label is not None:
            ax[1].imshow(label)
            ax[1].set_title("Label")
        if rover_mask is not None:
            ax[2].imshow(rover_mask)
            ax[2].set_title("Rover Mask")
        if range_mask is not None:
            ax[3].imshow(range_mask)
            ax[3].set_title("Range Mask")
        fig.suptitle(
            f"{idx}: " f"{os.path.basename(self.image_files[idx]).replace('.JPG', '')}"
        )
        plt.show()


def png_to_tensor(file):
    """
    Read a png file as grayscale
    """
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    tensor = torch.from_numpy(image).type(torch.uint8)
    return tensor


# Example Usage
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from simclr.modules.transformations import TransformsSimCLR

    dataset = AI4MarsDataset(
        "/home/goh/Documents/CLOVER/ai4mars-dataset-merged-0.1/",
        split="both",
        return_labels=True,
        transform=TransformsSimCLR(size=256),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        num_workers=1,
    )
    for i, ((view_1, view_2), label, rover_mask, range_mask) in enumerate(data_loader):
        # if dataset.return_labels = False, do: for i, (view_1, view_2) in enumerate():
        print(f"Batch: {i+1}/{len(data_loader)}")
        print(f"View 1 shape: {view_1.shape}")
        print(f"View 2 shape: {view_2.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Rover mask shape: {rover_mask.shape}")
        print(f"Range mask shape: {range_mask.shape}")
        if i >= 4:
            break
