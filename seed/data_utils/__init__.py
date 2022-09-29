import math
import os
from functools import partial
from warnings import warn

import numpy as np
import torch
from data_utils.ai4mars import AI4MarsDataset as AI4Mars
from data_utils.ai4mars import classes as ai4mars_classes
from data_utils.ai4mars import default_test_sets
from data_utils.hirise import HiRISEDataset, get_hirise_images_from_file_map
from data_utils.msl import MSLDataset, get_image_paths_from_file_map
from data_utils.unlabeled_images import UnlabeledImageDataset
from torchvision.datasets import CIFAR10, STL10, ImageNet


def get_dataset_from_name(args, **kwargs):
    dataset = args.dataset
    root = args.dataset_dir
    transform = kwargs.get("transform", None)
    test_transform = kwargs.get("test_transform", transform)
    train_file_map = args.train_file_map
    test_file_map = vars(args).get("test_file_map", "")  # defaults to ""

    if dataset == "STL10":
        # recycle the same function for train and test
        f = partial(STL10, root=root, download=True, transform=transform)
        train_ds, test_ds = f(split="train"), f(split="test")
        if args.train_pct < 1:
            idx = np.arange(len(train_ds))
            np.random.RandomState(args.seed).shuffle(idx)
            idx = idx[: int(len(idx) * args.train_pct)]
            train_ds = torch.utils.data.Subset(train_ds, idx)
    elif dataset.lower() in ["imagenet", "ilsvrc"]:
        train_ds = ImageNet(root, split="train", transform=transform)
        # test_ds = ImageNet(root, split="val", transform=transform)
        test_ds = None  # Edwin didn't have the val split available on Longhorn
        # take a stratified subset of the train set if train_pct < 1
        if 0 < args.num_images <= len(train_ds) and train_pct == 1:
            train_pct = args.num_images / len(train_ds)
        if train_pct < 1:
            classes, class_freq = np.unique(train_ds.targets, return_counts=True)
            indices_each_class = []  # will be a list of arrays of len(classes)
            for c, f in zip(classes, class_freq):
                # find class indices, shuffle with seed, then take first N images
                class_indices = np.where(train_ds.targets == c)[0]
                np.random.RandomState(args.seed).shuffle(class_indices)
                indices_each_class.append(class_indices[: round(train_pct * f)])
            indices = np.concatenate(indices_each_class)
            train_ds.samples = [train_ds.samples[i] for i in indices]
            train_ds.targets = [train_ds.targets[i] for i in indices]
            train_ds.imgs = train_ds.samples

    elif dataset == "CIFAR10":
        f = partial(CIFAR10, root=root, download=True, transform=transform)
        train_ds, test_ds = f(train=True), f(train=False)
        if args.train_pct < 1:
            idx = np.arange(len(train_ds))
            np.random.RandomState(args.seed).shuffle(idx)
            idx = idx[: int(len(idx) * args.train_pct)]
            train_ds = torch.utils.data.Subset(train_ds, idx)
    elif dataset == "MSL":
        msl_folder = os.path.join(root, "images")
        if train_file_map is None or not os.path.isfile(train_file_map):
            train_file_map = os.path.join(root, "train-set-v2.1.txt")
        train_files, train_labels = get_image_paths_from_file_map(
            msl_folder, train_file_map, args.train_pct, random_state=args.seed
        )
        train_ds = MSLDataset(
            train_files, train_labels, transform=transform, return_filename=False
        )
        if test_file_map == "":
            test_file_map = os.path.join(root, "test-set-v2.1.txt")
        test_images, test_labels = get_image_paths_from_file_map(
            msl_folder, test_file_map, 1.0, random_state=args.seed
        )
        test_ds = MSLDataset(
            test_images, test_labels, transform=test_transform, return_filename=False
        )
    elif dataset.upper() == "HIRISE":
        img_dir = os.path.join(root, "map-proj-v3_2")
        file_map = train_file_map  # make an alias
        if not os.path.isfile(file_map) or file_map == "":
            file_map = os.path.join(root, "labels-map-proj_v3_2_train_val_test.txt")
        if test_file_map == "":
            test_file_map = os.path.join(
                root, "labels-map-proj_v3_2_train_val_test.txt"
            )
        image_paths, train_labels = get_hirise_images_from_file_map(
            img_dir,
            file_map,
            split="train",
            train_pct=args.train_pct,
            random_state=args.seed,
        )
        train_ds = HiRISEDataset(image_paths, train_labels, transform=transform)
        image_paths, test_labels = get_hirise_images_from_file_map(
            img_dir, test_file_map, split="test", train_pct=1.0, random_state=args.seed
        )
        test_ds = HiRISEDataset(image_paths, test_labels, transform=test_transform)
    elif dataset == "MSLUnlabeled":
        train_ds = UnlabeledImageDataset(
            root,
            classes=None,  # None means use all instruments
            num_images=args.num_images,
            transform=transform,
            stratify_by_instrument=True,
        )
        test_ds = None
    elif dataset == "LROCUnlabeled":
        train_ds = UnlabeledImageDataset(
            root,
            num_images=args.num_images,
            random_state=args.seed,
            transform=transform,
        )
        # TODO @Kai: implement LROCUnlabeledDataset, referencing unlabeled_images.py
    elif dataset.upper() in ["MSL_AI4MARS", "AI4MARS"]:
        train_args = get_ai4mars_kwargs(args, kwargs, transform, "msl")
        test_args = get_ai4mars_kwargs(args, kwargs, test_transform, "msl")
        train_ds = AI4Mars(root, split="train", train_pct=args.train_pct, **train_args)
        test_ds = AI4Mars(root, split="test", **test_args)
    elif dataset.upper() == "M2020_AI4MARS":
        train_args = get_ai4mars_kwargs(args, kwargs, transform, "m2020_ai4mars")
        test_args = get_ai4mars_kwargs(args, kwargs, test_transform, "m2020_ai4mars")
        train_ds = AI4Mars(root, split="train", train_pct=args.train_pct, **train_args)
        test_ds = AI4Mars(root, split="test", image_ext="jpeg", **train_args)
    elif dataset.upper() == "M2020_GEOLOGY":
        train_args = get_ai4mars_kwargs(args, kwargs, transform, "m2020")
        test_args = get_ai4mars_kwargs(args, kwargs, test_transform, "m2020")
        train_ds = AI4Mars(root, split="train", train_pct=args.train_pct, **train_args)
        test_ds = AI4Mars(root, split="test", **train_args)
    # TODO: MSL_GEOLOGY
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return train_ds, test_ds


def get_ai4mars_kwargs(args, kwargs, transform, subfolder=None, png_null_value=255):
    # Select default test set folder if not specified, e.g., masked-gold-min1-100agree
    if "ai4mars_test_set" not in vars(args) or args.ai4mars_test_set is None:
        args.ai4mars_test_set = default_test_sets[subfolder]

    # Find the right class definition if not specified
    if "classes" not in vars(args):
        if subfolder == "msl":
            args.classes = ai4mars_classes[subfolder][args.num_classes]
        else:
            args.classes = ai4mars_classes[subfolder]
    if len(args.classes) != args.num_classes:
        warn(
            f"Number of classes {args.num_classes} does not match the "
            f"number of classes in the dataset {args.classes}. "
            f"Setting args.num_classes = {len(args.classes)}"
        )
        args.num_classes = len(args.classes)

    ai4mars_kwargs = dict(
        mission=subfolder,
        images_with_labels_only=True,
        test_set=args.ai4mars_test_set,
        png_null_value=png_null_value,
        return_labels=True,
        classes=args.classes,
        transform=transform,
        target_transform=kwargs.get("target_transform", None),
        random_state=args.seed,
    )
    return ai4mars_kwargs
