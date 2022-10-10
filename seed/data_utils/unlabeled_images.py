import math
import os
from typing import Any, Callable, List, Optional

import numpy as np
from PIL import Image, ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, DatasetFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


def robust_loader(path: str) -> Any:
    """
    Adapted from torchvision.datasets.folder.default_loader
    Randomly selects another image in the same folder if PIL fails to load the current path for any reason.
    """
    num_retries, max_retries = 0, 3
    failed_files = set()
    while num_retries <= max_retries:
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")
        except:
            # If image fails to load, remove it from list and try another one
            num_retries += 1
            failed_files.add(os.path.basename(path))
            folder = os.path.dirname(path)
            all_files = list(set(os.listdir(folder)) - failed_files)
            path = os.path.join(folder, np.random.choice(all_files))


class UnlabeledImageDataset(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    The default directory structure can be customized by overriding the :meth"`find_classes` method.

    This CLOVER implementation customizes torchvision's default ImageFolder class to allow for a subset of the classes to be returned.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        classes (list, optional): A list of classes to subset. If None, all classes, i.e., all subfolders are included.
        num_images (int, optional): The number of images to return. If -1 and train_pct is None, returns all images.
            num_images takes precedence over train_pct.
        train_pct (float, optional): The fraction of images to return. If None and num_images is None, returns all images.
        random_state (int, optional): The random seed to use when shuffling the dataset.
        stratify_by_instrument (bool, optional): Whether to return stratified samples by instrument. Defaults to False.

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    Example Usage:
    -------------
    >>> root = "~/clover_shared/datasets/msl_images"
    >>> dataset = UnlabeledImageDataset(root, transform=TransformsSimCLR(size=256), classes=["MAST_LEFT", "MAST_RIGHT"])
    >>> data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    >>> for ((view_1, view_2), instrument) in data_loader:
    >>>    embedding_1, embedding_2 = model(view_1, view_2)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = robust_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        classes: Optional[List[str]] = None,
        num_images: Optional[int] = -1,
        train_pct: Optional[float] = None,
        random_state: Optional[int] = 0,
        stratify_by_instrument: Optional[bool] = False,
        # uniform_sampling_across_instruments: Optional[bool] = False,  TODO: implement this
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        if classes is not None:
            class_to_idx = {
                _class: i for i, _class in enumerate(classes) if _class in self.classes
            }
            if len(class_to_idx) == 0:
                raise ValueError(
                    f"Specified classes {classes} do not exist in the dataset"
                )
            self.samples = [
                (s[0], class_to_idx[self.classes[s[1]]])
                for s in self.samples
                if self.classes[s[1]] in class_to_idx
            ]
            self.targets = [s[1] for s in self.samples]            
            self.classes = classes
            self.class_to_idx = class_to_idx

        # Shuffle and select a subset of the images based on num_images or train_pct
        rng = np.random.RandomState(random_state)
        rng.shuffle(self.samples)
        N = len(self.samples)
        if num_images > 0:
            num_images = min(num_images, N)
        elif num_images == -1 and train_pct is None:
            num_images = N
        elif train_pct is not None:
            num_images = int(N * train_pct)
        else:
            num_images = N
        train_pct = num_images / N
        # sub-sampling
        if num_images < N:
            if not stratify_by_instrument:
                self.samples = self.samples[:num_images]
                self.targets = [s[1] for s in self.samples]
            else:
                classes, class_freq = np.unique(self.targets, return_counts=True)
                # get number of images in each class based on train_pct
                # such that the sum = num_images
                num_images_each_class = np.array(class_freq * train_pct, dtype=int)
                while sum(num_images_each_class) != num_images:
                    if sum(num_images_each_class) < num_images:
                        num_images_each_class[np.argmin(num_images_each_class)] += 1
                    else:
                        num_images_each_class[np.argmax(num_images_each_class)] -= 1

                indices = []  # will be a list of arrays of len(classes)
                for c, n in zip(classes, num_images_each_class):
                    # find class indices, shuffle with seed, then take first N images
                    class_indices = np.where(self.targets == c)[0]
                    rng.shuffle(class_indices)
                    indices.append(class_indices[:n])
                indices = np.concatenate(indices)
                self.samples = [self.samples[i] for i in indices]
                self.targets = [self.targets[i] for i in indices]

        self.imgs = self.samples


if __name__ == "__main__":
    from collections import defaultdict

    root_dir = "~/clover_shared/datasets/msl_images"
    dataset = UnlabeledImageDataset(
        root_dir, classes=["MAST_LEFT", "MAST_RIGHT", "MAHLI"], num_images=5920
    )
    print(f"Number of images: {len(dataset)}")
    instrument_counts = defaultdict(int)
    for (img_path, inst) in dataset.samples:
        instrument_counts[dataset.classes[inst]] += 1
    print(f"Instrument counts: {instrument_counts}")
