import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.utils import _approximate_mode
from torchvision import transforms

torch.manual_seed(42)


def get_hirise_dataloader(
    hirise_folder, file_map, split="train", batch_size=32, shuffle=True, pin_memory=True
):

    image_paths, labels = get_hirise_images_from_file_map(
        hirise_folder, file_map, split
    )

    return torch.utils.data.DataLoader(
        HiRISEDataset(image_paths, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def get_hirise_images_from_file_map(
    hirise_folder, file_map, split, train_pct=1.0, random_state=0
):
    image_paths, labels = read_hirise_file_map(hirise_folder, file_map, split)
    print(
        f"Found {len(image_paths)} {split} HiRISE images from {file_map} "
        f"with extension {os.path.splitext(image_paths[0])[1]}. "
        f"Selecting {int(100 * train_pct)}% of images."
    )
    if split == "train" and train_pct < 1.0:
        # select random indices cumulatively
        shuffled_indices = np.random.RandomState(random_state).permutation(
            len(image_paths)
        )
        image_paths = np.array(image_paths)[shuffled_indices]
        labels = np.array(labels)[shuffled_indices]
        image_paths = image_paths[: int(len(image_paths) * train_pct)].tolist()
        labels = labels[: int(len(labels) * train_pct)].tolist()

    return image_paths, labels


def read_hirise_file_map(
    hirise_folder, file_map, split, return_split_col=False, keep_augmented=True
):
    # return all images if split == "all", else only return corresponding split
    check_split = lambda s: True if split == "all" else s == split
    with open(file_map, "r") as f:
        image_paths, labels, splits_ = zip(
            *(
                (os.path.join(hirise_folder, cols[0]), int(cols[1]), cols[2])
                for line in f.readlines()
                if check_split((cols := line.strip().split())[2])
                and (keep_augmented or not is_augmented(cols[0]))
            )
        )
    if return_split_col:
        return image_paths, labels, splits_
    else:
        return image_paths, labels


augmentations = [
    "-r90",  # 90 degrees clockwise rotation
    "-r180",  # 180 degrees clockwise rotation
    "-r270",  # 270 degrees clockwise rotation
    "-fh",  # horizontal flip
    "-fv",  # vertical flip
    "-brt",  # random brightness adjustment
]


def is_augmented(image_path):
    return any([a in image_path for a in augmentations])


class HiRISEDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index, force_tensor=True):
        file = self.image_paths[index]

        # load PNG file into Pytorch tensor
        image = Image.open(file).convert("RGB")

        # execute transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        elif not isinstance(image, torch.Tensor) and force_tensor:
            image = transforms.ToTensor()(image)

        return image, self.labels[index]

    def plot_sample(self, index):
        image, label = self[index]
        filename = os.path.basename(self.image_paths[index])
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"{filename} ({label})")
        plt.axis("off")
        plt.show()


def write_cumulative_splits_to_folder(
    file_map,
    split,
    pcts=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    keep_augmented=True,
    random_state=42,
):
    # Create output directory
    augmented = "_augmented" if keep_augmented else ""
    out_dir = os.path.join(
        os.path.dirname(file_map),
        f"cumulative_stratified_splits{augmented}_seed{random_state}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing splits to {out_dir}")
    # Get splits and write to text files
    splits = generate_cumulative_stratified_splits(file_map, split, pcts, random_state)
    for i, (image_files, labels, splits_) in enumerate(splits):
        out_file = os.path.join(
            out_dir, f"{int(pcts[i]*100)}pct{split.capitalize()}.txt"
        )
        with open(out_file, "w") as f:
            for img, label, split in zip(image_files, labels, splits_):
                f.write(f"{img}\t{label}\t{split}\n")


def generate_cumulative_stratified_splits(
    file_map, split, pcts=[0.1, 0.2, 0.5, 0.8], keep_augmented=True, random_state=42
):
    return [
        get_cumulative_stratified_split(
            file_map, split, pct, keep_augmented, random_state
        )
        for pct in pcts
    ]


def get_cumulative_stratified_split(
    file_map, split, pct, keep_augmented=True, random_state=42
):
    image_paths, labels, image_splits = read_hirise_file_map(
        "", file_map, split, return_split_col=True, keep_augmented=keep_augmented
    )
    # For each class, get the shuffled indices and number of samples based on class freq.
    rng = np.random.RandomState(random_state)
    shuffled_class_indices = shuffle_by_class(labels, rng)
    num_samples_per_class = get_stratified_num_samples(labels, pct, rng)

    # Get indices for each class and concatenate
    final_indices = np.array([], dtype=np.int32)
    for idx, num_samples in zip(shuffled_class_indices, num_samples_per_class):
        final_indices = np.concatenate((final_indices, idx[:num_samples]))

    # Index the subset and return
    final_indices = np.sort(final_indices)
    img_split = np.array(image_paths)[final_indices].tolist()
    label_split = np.array(labels)[final_indices].tolist()
    split_split = np.array(image_splits)[final_indices].tolist()
    return img_split, label_split, split_split


def shuffle_by_class(labels, rng=42):
    # get frequency of each label; taken from
    # https://github.com/scikit-learn/scikit-learn/blob/582fa30a31ffd1d2afc6325ec3506418e35b88c2/sklearn/model_selection/_split.py#L1935
    # [classes[i] for i in y_indices] == labels
    classes, y_indices, class_counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    sorted_indices = np.argsort(y_indices, kind="mergesort")
    split_at_idx = np.cumsum(class_counts)[:-1]
    class_indices = np.split(sorted_indices, split_at_idx)  # returns list of np.array

    # shuffle the indices for each class
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)

    shuffled_class_indices = [rng.permutation(indices) for indices in class_indices]
    return shuffled_class_indices


def get_stratified_num_samples(labels, pct, rng=42):
    _, class_counts = np.unique(labels, return_counts=True)
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)

    num_samples_per_class = _approximate_mode(class_counts, int(pct * len(labels)), rng)
    return num_samples_per_class


if __name__ == "__main__":
    # get the file map
    file_map = "/home/goh/Documents/CLOVER/data/hirise-map-proj-v3_2/labels-map-proj_v3_2_train_val_test.txt"

    # write the splits to files
    write_cumulative_splits_to_folder(file_map, "train")
    write_cumulative_splits_to_folder(file_map, "train", keep_augmented=False)
