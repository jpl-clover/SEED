import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

torch.manual_seed(42)


def get_msl_dataloader(
    images_folder, file_map, batch_size=32, shuffle=True, pin_memory=True
):

    image_paths, labels = get_image_paths_from_file_map(images_folder, file_map)
    return torch.utils.data.DataLoader(
        MSLDataset(image_paths, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def get_image_paths_from_file_map(
    images_folder, file_map, train_pct=1.0, random_state=0
):
    with open(file_map, "r") as f:
        lines = f.readlines()
    image_paths, labels = [], []

    # cumulative sampling of a certain percentage of training data
    np.random.RandomState(random_state).shuffle(lines)
    lines = lines[: int(len(lines) * train_pct)]

    for line in lines:
        cols = line.strip().split()
        filepath = os.path.join(images_folder, cols[0])
        label = int(cols[1])
        image_paths.append(filepath)
        labels.append(label)
    print(
        f"Found {len(image_paths)} images in {images_folder} "
        f"with extension {os.path.splitext(image_paths[0])[1]}"
        f" from file map {file_map}."
    )

    return image_paths, labels


class MSLDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None, return_filename=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_filename = return_filename

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
        if self.return_filename:
            return image, self.labels[index], file
        else:
            return image, self.labels[index]

    def plot_sample(self, index):
        image, label = self[index]
        filename = os.path.basename(self.image_paths[index])
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"{filename} ({label})")
        plt.axis("off")
        plt.show()


# read all images and labels, and write to an npz file
def generate_msl_npz(images_folder, file_map, out_file):
    with open(file_map, "r") as f:
        lines = f.readlines()

    image_files, labels = zip(*(line.strip().split() for line in lines))
    image_paths = [os.path.join(images_folder, file) for file in image_files]
    labels = [int(label) for label in labels]
    unique_labels = sorted(set(labels))
    if min(unique_labels) == 1:
        labels = [labels - 1 for labels in labels]  # convert to 0-indexed

    def load_image(path):
        # use PIL to open the image and convert to numpy
        image = Image.open(path).convert("RGB")
        return np.array(image)

    images = np.stack([load_image(path) for path in tqdm(image_paths)])

    print(
        f"Read {len(images)} images from {images_folder} "
        f"into a numpy array of shape {images.shape}."
        f"\nWriting to {out_file}."
    )
    # save to npz
    np.savez_compressed(out_file, images=images, labels=labels, filenames=image_files)


if __name__ == "__main__":

    # example usage:
    # python msl.py -i "path/to/msl-labeled-data-set-v2.1/images" -m "/path/to/msl-labeled-data-set-v2.1/train-set-v2.1.txt" -o "/path/to/msl-labeled-data-set-v2.1/train-set-v2.1.npz"

    args = argparse.ArgumentParser()
    args.add_argument("-i", "--images_folder", type=str, required=True)
    args.add_argument("-m", "--file_map", type=str, required=True)
    args.add_argument("-o", "--out_file", type=str, required=True)

    args = args.parse_args()

    generate_msl_npz(args.images_folder, args.file_map, args.out_file)
